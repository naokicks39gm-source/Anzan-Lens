import base64
import json
import io
import os
import re
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


def build_prompt(spatial_focus: bool) -> str:
    spatial_rules = ""
    if spatial_focus:
        spatial_rules = """
        空間認識ルール:
        - 画像全体をスキャンし、見出し（みとりざん、かけざん、わりざん）の位置をまず特定せよ
        - かけざんセクションにおいて、左側(1-3番)だけでなく、その右側に配置されている(4-6番)を絶対に見落とすな
        - 画像右側のブロックも必ずスキャンし、見落としを防げ
        - 『4 234×312=』のような、問題番号、式、手書き解答のセットを漏れなく抽出せよ
        - JSONレスポンスには、読み取ったすべての問題番号を含めること
        """

    return f"""
        あなたは日本の学習プリント画像を解析するOCRアシスタントです。
        画像内の各セクション（見出し）を特定し、セクションごとに最適な抽出ルールを適用してください。

        対象例: 「みとりざん（表）」「かけざん（横式）」「わりざん（横式）」が同一画像に混在します。

        抽出ルール:
        - みとりざん: 15列の表として列ごとの7つの数値と合計を抽出
        - かけざん/わりざん: 問題番号、式、手書きの解答をセットで抽出
        - 読み取りにくい手書き数字（赤丸内など）は周囲の文脈から慎重に推論
        {spatial_rules}

        出力は以下のJSONのみ。説明文やMarkdownは禁止。

        {{
          "format_type": "multi",
          "sections": [
            {{
              "title": "みとりざん",
              "type": "table",
              "results": [
                {{"column": 1, "numbers": [85, 77, 59, 23, 74, 80, 97], "total": 495}}
                ... 15 columns total ...
              ]
            }},
            {{
              "title": "かけざん",
              "type": "list",
              "items": [
                {{"number": "1", "expression": "12×3=", "answer": "36"}},
                ...
              ]
            }},
            {{
              "title": "わりざん",
              "type": "list",
              "items": [
                {{"number": "1", "expression": "12÷3=", "answer": "4"}},
                ...
              ]
            }}
          ]
        }}

        追加ルール:
        - 数値が不明な場合は0または空文字にする
        - 必ずJSONのみを返す
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
    prompt = build_prompt(spatial_focus=True) + "\nかけざん・わりざんを優先して抽出せよ。"
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
        if not API_KEY:
            return jsonify({"error": "GEMINI_API_KEY is not set"}), 500

        data = request.json
        data_url = data['image']
        header, encoded = data_url.split(',', 1)
        mime_type = header.split(';')[0].split(':')[1] if ':' in header else "image/jpeg"
        image_bytes = base64.b64decode(encoded)

        prompt = build_prompt(spatial_focus=True)
        parsed = generate_json_response(prompt, image_bytes, mime_type)
        if not isinstance(parsed, dict):
            return jsonify(build_fallback_payload())

        if parsed.get("format_type") != "multi":
            parsed["format_type"] = "multi"
        if "sections" not in parsed or not isinstance(parsed["sections"], list):
            return jsonify(build_fallback_payload())

        if needs_kakezan_rescan(parsed):
            retry_prompt = build_prompt(spatial_focus=True) + "\n再走査: かけざんの4-6番を絶対に取りこぼさず補完してJSONのみ返せ。"
            retry_parsed = generate_json_response(retry_prompt, image_bytes, mime_type)
            if isinstance(retry_parsed, dict) and isinstance(retry_parsed.get("sections"), list):
                parsed = retry_parsed
                if parsed.get("format_type") != "multi":
                    parsed["format_type"] = "multi"

        best_mitori = best_mitorizan_from_variants(image_bytes, mime_type)
        if best_mitori is not None:
            sections = parsed.get("sections", [])
            replaced = False
            for i, s in enumerate(sections):
                title = str(s.get("title", ""))
                if "みとり" in title and s.get("type") == "table":
                    sections[i] = best_mitori
                    replaced = True
                    break
            if not replaced:
                sections.append(best_mitori)
            parsed["sections"] = sections

        best_kake = best_kakezan_from_variants(image_bytes)
        if best_kake is not None:
            sections = parsed.get("sections", [])
            for i, s in enumerate(sections):
                title = str(s.get("title", ""))
                if "かけざん" in title and s.get("type") == "list":
                    if kakezan_score(best_kake) > kakezan_score(s):
                        sections[i] = best_kake
                    break
            parsed["sections"] = sections

        return jsonify(parsed)

    except Exception as e:
        print(f"Server Error: {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(port=5000, debug=True)
