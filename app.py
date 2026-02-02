import base64
import json
import io
import os
import re
import time
from flask import Flask, request, jsonify, render_template
from google import genai
from google.genai import types
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

app = Flask(__name__)

# --- 設定: APIキーとモデル設定 ---
API_KEY = os.getenv("GEMINI_API_KEY", "")
MODEL_ID = "models/gemini-2.5-flash"

client = genai.Client(
    api_key=API_KEY,
    http_options=types.HttpOptions(apiVersion="v1"),
)


def parse_json_from_text(text_response: str):
    text_response = (text_response or "").strip()
    if not text_response:
        return None

    json_match = re.search(r'\{.*\}', text_response, re.DOTALL)
    if json_match:
        return json.loads(json_match.group())

    return json.loads(text_response)


def build_fallback_payload():
    return {
        "format_type": "multi",
        "sections": []
    }


def build_prompt_accuracy() -> str:
    return """
        OCR抽出。JSONのみ。
        - みとりざん: 列1-4は原則7個、列5-8は原則8個を優先し、可変長で numbers を返す。
        - みとりざん results は {column, numbers, total} を返す。
        - かけざん: 1-6を必ず返す。左右2列(1-3,4-6)を混在させない。式は3桁×3桁として読む。
        - わりざん: 1-3を返す。
        - 読みにくい場合も、最も妥当な数字を1つ選ぶ。

        JSON:
        {
          "format_type":"multi",
          "sections":[
            {"title":"みとりざん","type":"table","results":[{"column":1,"numbers":[1,2,3,4,5,6,7],"total":28}]},
            {"title":"かけざん","type":"list","items":[{"number":"1","expression":"234×312=","answer":"73008"}]},
            {"title":"わりざん","type":"list","items":[{"number":"1","expression":"27864÷448=","answer":"63"}]}
          ]
        }
    """


def build_prompt_speed() -> str:
    return """
        OCRを高速に実行。JSONのみ。
        - みとりざん: 列1-4は7個、列5-8は8個を優先。resultsは {column, numbers, total}。
        - かけざん: 1-6を返す。左右列(1-3,4-6)を混在させない。式は3桁×3桁を優先。
        - わりざん: 1-3を返す。
        JSON:
        {"format_type":"multi","sections":[{"title":"みとりざん","type":"table","results":[{"column":1,"numbers":[1,2,3,4,5,6,7],"total":28}]},{"title":"かけざん","type":"list","items":[{"number":"1","expression":"234×312=","answer":"73008"}]},{"title":"わりざん","type":"list","items":[{"number":"1","expression":"27864÷448=","answer":"63"}]}]}
    """


def build_mitorizan_prompt() -> str:
    return """
        あなたは「みとりざん」表の転記専用OCRです。
        推測せず、画像に見える数字だけを列ごとに配列で返してください。
        各列は row_count を明示してください。
        第1〜4列と第5〜8列で行数が異なる場合でも、見えた行数をそのまま返してください。
        合計 total は numbers の合計で算出してください。

        出力はJSONのみ:
        {
          "format_type": "multi",
          "sections": [
            {
              "title": "みとりざん",
              "type": "table",
              "results": [
                {"column": 1, "numbers": [85, 77, 59, 23, 74, 80, 97], "row_count": 7, "total": 495}
              ]
            }
          ]
        }
    """


def generate_json_response(prompt: str, image_bytes: bytes, mime_type: str):
    try:
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=[
                prompt,
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
            ),
        )
    except Exception:
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=[
                prompt,
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            ],
        )
    return parse_json_from_text(response.text)


def get_mitorizan_section(parsed):
    if not isinstance(parsed, dict):
        return None
    for section in parsed.get("sections", []):
        title = str(section.get("title", ""))
        if "みとり" in title and section.get("type") == "table":
            return section
    return None


def mitorizan_score(section):
    if not section:
        return -1
    results = section.get("results", [])
    if not isinstance(results, list):
        return -1
    col_map = {}
    for r in results:
        try:
            c = int(r.get("column"))
        except Exception:
            continue
        nums = r.get("numbers", [])
        if not isinstance(nums, list):
            nums = []
        col_map[c] = len(nums)

    score = 0
    # Prefer exact row counts: col1-4=7, col5-8=8
    for c in range(1, 5):
        n = col_map.get(c, 0)
        score += max(0, 12 - abs(n - 7) * 4)
    for c in range(5, 9):
        n = col_map.get(c, 0)
        score += max(0, 14 - abs(n - 8) * 4)
    # Soft bonus for populated 1..8 columns
    score += sum(1 for c in range(1, 9) if col_map.get(c, 0) > 0)
    return score


def kakezan_score(section):
    if not section or section.get("type") != "list":
        return -1
    items = section.get("items", [])
    if not isinstance(items, list):
        return -1
    score = len(items)
    for item in items:
        expr = str(item.get("expression", ""))
        expr = expr.replace("×", "*").replace("x", "*").replace("X", "*")
        parts = [p for p in expr.split("*") if p]
        if len(parts) >= 2 and len(parts[0].strip()) == 3 and len(parts[1].strip()) == 3:
            score += 10
    return score


def best_mitorizan_from_variants(image_bytes: bytes, mime_type: str):
    prompt = build_mitorizan_prompt()
    img = Image.open(io.BytesIO(image_bytes))
    variants = [
        ("orig", img),
        ("rot90", img.rotate(90, expand=True)),
        ("rot270", img.rotate(270, expand=True)),
    ]
    best_section = None
    best_score = -1
    for _, variant in variants:
        buf = io.BytesIO()
        variant.save(buf, format="PNG")
        parsed = generate_json_response(prompt, buf.getvalue(), "image/png")
        section = get_mitorizan_section(parsed)
        score = mitorizan_score(section)
        if score > best_score:
            best_score = score
            best_section = section
    return best_section


def best_kakezan_from_variants(image_bytes: bytes):
    prompt = build_prompt_accuracy() + "\nかけざん・わりざんを優先して抽出せよ。"
    img = Image.open(io.BytesIO(image_bytes))
    w, h = img.size
    variants = [
        ("right45", img.crop((int(w * 0.45), int(h * 0.05), int(w * 0.98), int(h * 0.95))),
        ),
        ("right50", img.crop((int(w * 0.50), int(h * 0.05), int(w * 0.98), int(h * 0.95))),
        ),
    ]
    best_section = None
    best_score = -1
    for _, variant in variants:
        buf = io.BytesIO()
        variant.save(buf, format="PNG")
        parsed = generate_json_response(prompt, buf.getvalue(), "image/png")
        section = get_section(parsed, "かけざん") if isinstance(parsed, dict) else None
        score = kakezan_score(section)
        if score > best_score:
            best_score = score
            best_section = section
    return best_section


def enrich_mitorizan_answers(parsed):
    if not isinstance(parsed, dict):
        return parsed
    sections = parsed.get("sections", [])
    for section in sections:
        if section.get("type") != "table":
            continue
        title = str(section.get("title", ""))
        if "みとり" not in title:
            continue
        results = section.get("results", [])
        if not isinstance(results, list):
            continue
        for result in results:
            numbers = result.get("numbers", [])
            if not isinstance(numbers, list):
                numbers = []
            correct = 0
            for n in numbers:
                try:
                    correct += int(str(n).replace(",", ""))
                except Exception:
                    continue
            result["correct_answer"] = str(correct)
            if "student_answer" not in result:
                student = result.get("total", "")
                result["student_answer"] = str(student) if student is not None else ""
    return parsed


def resize_for_speed(image_bytes: bytes, max_side: int = 1700) -> tuple[bytes, str]:
    try:
        img = Image.open(io.BytesIO(image_bytes))
    except Exception:
        return image_bytes, "image/jpeg"

    w, h = img.size
    m = max(w, h)
    if m <= max_side:
        return image_bytes, "image/jpeg"

    scale = max_side / float(m)
    nw, nh = int(w * scale), int(h * scale)
    img = img.resize((nw, nh))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue(), "image/jpeg"


def get_section(parsed, title_hint: str):
    for section in parsed.get("sections", []):
        title = str(section.get("title", ""))
        if title_hint in title:
            return section
    return None


def needs_kakezan_rescan(parsed) -> bool:
    section = get_section(parsed, "かけざん")
    if not section:
        return True
    items = section.get("items", [])
    if not isinstance(items, list) or not items:
        return True
    numbers = {str(item.get("number", "")).strip() for item in items}
    required = {"1", "2", "3", "4", "5", "6"}
    return not required.issubset(numbers)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/ocr', methods=['POST'])
def perform_ocr():
    try:
        time_start = time.time()
        if not API_KEY:
            return jsonify({"error": "GEMINI_API_KEY is not set"}), 500

        data = request.json
        data_url = data['image']
        mode = str(data.get("mode", "speed")).lower()
        if mode not in {"accuracy", "speed"}:
            mode = "accuracy"
        header, encoded = data_url.split(',', 1)
        mime_type = header.split(';')[0].split(':')[1] if ':' in header else "image/jpeg"
        image_bytes = base64.b64decode(encoded)

        if mode == "speed":
            prompt = build_prompt_speed()
            image_bytes, mime_type = resize_for_speed(image_bytes, max_side=1700)
        else:
            prompt = build_prompt_accuracy()

        parsed = generate_json_response(prompt, image_bytes, mime_type)
        if not isinstance(parsed, dict):
            return jsonify(build_fallback_payload())

        if parsed.get("format_type") != "multi":
            parsed["format_type"] = "multi"
        if "sections" not in parsed or not isinstance(parsed["sections"], list):
            return jsonify(build_fallback_payload())

        parsed = enrich_mitorizan_answers(parsed)
        parsed["mode"] = mode
        parsed["processing_time_ms"] = int((time.time() - time_start) * 1000)
        return jsonify(parsed)

    except Exception as e:
        print(f"Server Error: {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(port=5000, debug=True)
