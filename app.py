import base64
import hashlib
import json
import io
import os
import re
import time
from flask import Flask, request, jsonify, render_template
from google import genai
from google.genai import types
from dotenv import load_dotenv
from PIL import Image, ImageOps

load_dotenv()

app = Flask(__name__)

# --- 設定: APIキーとモデル設定 ---
API_KEY = os.getenv("GEMINI_API_KEY", "")
MODEL_ID = "models/gemini-2.5-flash"
SPEED_MODEL_ID = "models/gemini-2.5-flash-lite"

client = genai.Client(
    api_key=API_KEY,
    http_options=types.HttpOptions(apiVersion="v1"),
)

# In-memory cache for repeated OCR on the same image+mode.
OCR_CACHE = {}
CACHE_MAX_ITEMS = 24


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


def make_cache_key(image_bytes: bytes, mode: str) -> str:
    digest = hashlib.sha256(image_bytes).hexdigest()
    return f"{mode}:{digest}"


def get_cached_payload(cache_key: str):
    payload = OCR_CACHE.get(cache_key)
    if not isinstance(payload, dict):
        return None
    # Return a copy so response mutations do not overwrite cache.
    return json.loads(json.dumps(payload))


def set_cached_payload(cache_key: str, payload: dict):
    if len(OCR_CACHE) >= CACHE_MAX_ITEMS:
        # Drop oldest inserted key to cap memory.
        first_key = next(iter(OCR_CACHE))
        OCR_CACHE.pop(first_key, None)
    OCR_CACHE[cache_key] = json.loads(json.dumps(payload))


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


def build_calc_prompt_speed() -> str:
    return """
        OCR抽出。JSONのみ。
        - かけざん: 1-6を必ず返す。左右列(1-3,4-6)を混在させない。式は3桁×3桁を優先。
        - わりざん: 1-3を返す。
        JSON:
        {"format_type":"multi","sections":[{"title":"かけざん","type":"list","items":[{"number":"1","expression":"234×312=","answer":"73008"}]},{"title":"わりざん","type":"list","items":[{"number":"1","expression":"27864÷448=","answer":"63"}]}]}
    """


def generate_json_response(
    prompt: str,
    image_bytes: bytes,
    mime_type: str,
    *,
    model_id: str = MODEL_ID,
    max_output_tokens: int | None = None,
    temperature: float | None = None,
):
    cfg_kwargs = {"response_mime_type": "application/json"}
    if max_output_tokens is not None:
        cfg_kwargs["max_output_tokens"] = max_output_tokens
    if temperature is not None:
        cfg_kwargs["temperature"] = temperature

    try:
        response = client.models.generate_content(
            model=model_id,
            contents=[
                prompt,
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            ],
            config=types.GenerateContentConfig(**cfg_kwargs),
        )
    except Exception:
        response = client.models.generate_content(
            model=model_id,
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


def resize_for_speed(image_bytes: bytes, max_side: int = 1400) -> tuple[bytes, str]:
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
    img = img.resize((nw, nh), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=78, optimize=True)
    return buf.getvalue(), "image/jpeg"


def normalize_exif_orientation(image_bytes: bytes) -> bytes:
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img = ImageOps.exif_transpose(img).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=92)
        return buf.getvalue()
    except Exception:
        return image_bytes


def transform_image(image_bytes: bytes, transform: str) -> bytes:
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return image_bytes

    if transform == "rot180":
        img = img.rotate(180, expand=True)
    elif transform == "flip_lr":
        img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    else:
        return image_bytes

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return buf.getvalue()


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


def replace_or_append_section(parsed, title_hint: str, replacement):
    if not isinstance(parsed, dict) or not isinstance(replacement, dict):
        return parsed
    sections = parsed.get("sections", [])
    if not isinstance(sections, list):
        parsed["sections"] = [replacement]
        return parsed
    for i, section in enumerate(sections):
        if title_hint in str(section.get("title", "")):
            sections[i] = replacement
            return parsed
    sections.append(replacement)
    return parsed


def mitori_has_8_columns(parsed) -> bool:
    sec = get_section(parsed, "みとり")
    if not sec:
        return False
    results = sec.get("results", [])
    if not isinstance(results, list):
        return False
    cols = set()
    for r in results:
        try:
            c = int(r.get("column"))
        except Exception:
            continue
        cols.add(c)
    return set(range(1, 9)).issubset(cols)


def mitori_layout_valid(parsed) -> bool:
    sec = get_section(parsed, "みとり")
    if not sec:
        return False
    results = sec.get("results", [])
    if not isinstance(results, list):
        return False
    col_map = {}
    for r in results:
        try:
            c = int(r.get("column"))
        except Exception:
            continue
        nums = r.get("numbers", [])
        col_map[c] = len(nums) if isinstance(nums, list) else 0

    # Speed mode validation: allow extra rows when OCR over-segments,
    # but require minimum expected rows for each column block.
    first_ok = all(col_map.get(c, 0) >= 7 for c in range(1, 5))
    latter_ok = all(col_map.get(c, 0) >= 8 for c in range(5, 9))
    return first_ok and latter_ok


def calc_sections_complete(parsed) -> bool:
    kake = get_section(parsed, "かけざん")
    wari = get_section(parsed, "わりざん")
    if not kake or not wari:
        return False
    k_items = kake.get("items", [])
    w_items = wari.get("items", [])
    if not isinstance(k_items, list) or not isinstance(w_items, list):
        return False
    k_nums = {str(item.get("number", "")).strip() for item in k_items}
    w_nums = {str(item.get("number", "")).strip() for item in w_items}
    return {"1", "2", "3", "4", "5", "6"}.issubset(k_nums) and {"1", "2", "3"}.issubset(w_nums)


def safe_generate_json_response(
    prompt: str,
    image_bytes: bytes,
    mime_type: str,
    *,
    model_id: str,
    max_output_tokens: int,
    temperature: float = 0.0,
):
    try:
        return generate_json_response(
            prompt,
            image_bytes,
            mime_type,
            model_id=model_id,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )
    except Exception:
        return None


def parsed_completeness_score(parsed) -> int:
    if not isinstance(parsed, dict):
        return -1
    score = 0
    if mitori_has_8_columns(parsed):
        score += 10
    if mitori_layout_valid(parsed):
        score += 10
    if calc_sections_complete(parsed):
        score += 10
    return score


def run_speed_pipeline(image_bytes: bytes, mime_type: str, *, allow_accuracy_fallback: bool):
    prompt = build_prompt_speed()
    speed_bytes, speed_mime = resize_for_speed(image_bytes, max_side=1400)
    parsed = safe_generate_json_response(
        prompt,
        speed_bytes,
        speed_mime,
        model_id=SPEED_MODEL_ID,
        max_output_tokens=1200,
        temperature=0.0,
    )
    if not isinstance(parsed, dict):
        parsed = safe_generate_json_response(
            prompt,
            speed_bytes,
            speed_mime,
            model_id=MODEL_ID,
            max_output_tokens=1500,
            temperature=0.0,
        )
    if not isinstance(parsed, dict):
        return None

    if not mitori_layout_valid(parsed):
        mitori_patch = safe_generate_json_response(
            build_mitorizan_prompt(),
            speed_bytes,
            speed_mime,
            model_id=SPEED_MODEL_ID,
            max_output_tokens=1000,
            temperature=0.0,
        )
        replacement = get_section(mitori_patch, "みとり") if isinstance(mitori_patch, dict) else None
        if replacement:
            parsed = replace_or_append_section(parsed, "みとり", replacement)

    if not mitori_layout_valid(parsed):
        mitori_patch = safe_generate_json_response(
            build_mitorizan_prompt(),
            speed_bytes,
            speed_mime,
            model_id=MODEL_ID,
            max_output_tokens=1100,
            temperature=0.0,
        )
        replacement = get_section(mitori_patch, "みとり") if isinstance(mitori_patch, dict) else None
        if replacement:
            parsed = replace_or_append_section(parsed, "みとり", replacement)

    if not calc_sections_complete(parsed):
        calc_patch = safe_generate_json_response(
            build_calc_prompt_speed(),
            speed_bytes,
            speed_mime,
            model_id=SPEED_MODEL_ID,
            max_output_tokens=700,
            temperature=0.0,
        )
        if isinstance(calc_patch, dict):
            kake = get_section(calc_patch, "かけざん")
            wari = get_section(calc_patch, "わりざん")
            if kake:
                parsed = replace_or_append_section(parsed, "かけざん", kake)
            if wari:
                parsed = replace_or_append_section(parsed, "わりざん", wari)

    if not calc_sections_complete(parsed):
        calc_patch = safe_generate_json_response(
            build_calc_prompt_speed(),
            speed_bytes,
            speed_mime,
            model_id=MODEL_ID,
            max_output_tokens=900,
            temperature=0.0,
        )
        if isinstance(calc_patch, dict):
            kake = get_section(calc_patch, "かけざん")
            wari = get_section(calc_patch, "わりざん")
            if kake:
                parsed = replace_or_append_section(parsed, "かけざん", kake)
            if wari:
                parsed = replace_or_append_section(parsed, "わりざん", wari)

    if allow_accuracy_fallback and (not mitori_has_8_columns(parsed) or not calc_sections_complete(parsed)):
        parsed = safe_generate_json_response(
            build_prompt_accuracy(),
            speed_bytes,
            speed_mime,
            model_id=MODEL_ID,
            max_output_tokens=1800,
            temperature=0.0,
        )
    return parsed


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
        image_bytes = normalize_exif_orientation(image_bytes)
        cache_key = make_cache_key(image_bytes, mode)
        cached = get_cached_payload(cache_key)
        if cached is not None:
            cached["mode"] = mode
            cached["cached"] = True
            cached["processing_time_ms"] = int((time.time() - time_start) * 1000)
            return jsonify(cached)

        if mode == "speed":
            parsed = run_speed_pipeline(image_bytes, mime_type, allow_accuracy_fallback=False)
            if parsed_completeness_score(parsed) < 20:
                candidates = [
                    run_speed_pipeline(transform_image(image_bytes, "rot180"), mime_type, allow_accuracy_fallback=False),
                    run_speed_pipeline(transform_image(image_bytes, "flip_lr"), mime_type, allow_accuracy_fallback=False),
                ]
                best = parsed
                best_score = parsed_completeness_score(parsed)
                for cand in candidates:
                    score = parsed_completeness_score(cand)
                    if score > best_score:
                        best = cand
                        best_score = score
                parsed = best
            if parsed_completeness_score(parsed) < 20:
                parsed = run_speed_pipeline(image_bytes, mime_type, allow_accuracy_fallback=True)
            if not isinstance(parsed, dict):
                return jsonify(build_fallback_payload())
        else:
            prompt = build_prompt_accuracy()
            parsed = generate_json_response(
                prompt,
                image_bytes,
                mime_type,
                model_id=MODEL_ID,
                max_output_tokens=2600,
            )
        if not isinstance(parsed, dict):
            return jsonify(build_fallback_payload())

        if parsed.get("format_type") != "multi":
            parsed["format_type"] = "multi"
        if "sections" not in parsed or not isinstance(parsed["sections"], list):
            return jsonify(build_fallback_payload())

        parsed = enrich_mitorizan_answers(parsed)
        set_cached_payload(cache_key, parsed)
        parsed["mode"] = mode
        parsed["cached"] = False
        parsed["processing_time_ms"] = int((time.time() - time_start) * 1000)
        return jsonify(parsed)

    except Exception as e:
        print(f"Server Error: {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(port=5000, debug=True)
