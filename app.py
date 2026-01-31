import base64
import json
import os
import re
import time
import io
from flask import Flask, request, jsonify, render_template
from google import genai
from google.genai import types
from dotenv import load_dotenv
from PIL import Image
import pytesseract

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
        読み取りルール（高速版）:
        - 画像は一度だけ全体を見て処理する
        - 問題番号（1,2,3…）をアンカーとして認識する
        - 番号の近くにある「式」と「手書き解答」を1セットとして扱う
        - 迷った場合は「番号に最も近い数字」を student_answer とする
        - 推測後の再検討は禁止

        レイアウトの最低限ルール:
        - かけ算は左右2列構成
        - 左列：1〜3番 / 右列：4〜6番
        - 右列（4〜6番）も必ず処理する
        - 左右の列を混在させない
        """

    return f"""
        あなたは高速・実用特化の手書き算数OCR＋採点AIです。
        対象: 小学生の暗算プリント画像（みとりざん/かけざん/わりざんが混在）。

        抽出ルール:
        - みとりざん: 15列の表として列ごとの7つの数値と合計を抽出
        - かけざん/わりざん: 問題番号、式、手書きの解答をセットで抽出
        - かけざん/わりざんは「正解値」を即計算し、手書き解答と一致するか判定する
        - 各問題に answer_confidence（0.0〜1.0）と needs_review（true/false）を付与する
        {spatial_rules}

        出力は以下のJSONのみ。説明文やMarkdownは禁止。

        {{
          "format_type": "multi",
          "processing_time_ms": 1850,
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
                {{"number": "1", "expression": "12×3=", "answer": "36", "correct_answer": "36", "is_correct": true, "mark": "○", "mark_position": "near_student_answer_top_right", "answer_confidence": 0.9, "needs_review": false}},
                ...
              ]
            }},
            {{
              "title": "わりざん",
              "type": "list",
              "items": [
                {{"number": "1", "expression": "12÷3=", "answer": "4", "correct_answer": "4", "is_correct": true, "mark": "○", "mark_position": "near_student_answer_top_right", "answer_confidence": 0.9, "needs_review": false}},
                ...
              ]
            }}
          ]
        }}

        追加ルール:
        - 数値が不明な場合は0または空文字にする
        - すべてのセクション（みとりざん、かけざん、わりざん）を漏れなく含める
        - is_correct に応じて mark を必ず付与（true→○, false→×）
        - mark_position は student_answer の近くでよい（例: \"near_student_answer_top_right\"）
        - answer_confidence は必ず数値で返す
        - needs_review は以下のいずれかで true:
          * answer_confidence < 0.75
          * is_correct が false かつ confidence が低い
          * 桁数が式の結果と不自然にズレている
        - 処理時間を processing_time_ms にミリ秒で必ず出力
        - 必ずJSONのみを返す
        """


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


def parse_number(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    text = text.replace(",", "")
    try:
        if "." in text:
            return float(text)
        return int(text)
    except ValueError:
        return None


def count_digits(value):
    if value is None:
        return 0
    text = str(value).strip()
    if not text:
        return 0
    text = text.replace("-", "").replace(".", "")
    return len(text)


def extract_ocr_tokens(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes))
    except Exception:
        return []

    config = "--psm 6 -c tessedit_char_whitelist=0123456789"
    data = pytesseract.image_to_data(image, config=config, output_type=pytesseract.Output.DICT)
    tokens = []
    for text, conf in zip(data.get("text", []), data.get("conf", [])):
        if text is None:
            continue
        cleaned = re.sub(r"\\D", "", str(text))
        if not cleaned:
            continue
        try:
            conf_val = float(conf)
        except Exception:
            continue
        if conf_val < 0:
            continue
        tokens.append({"text": cleaned, "conf": conf_val / 100.0})
    return tokens


def compute_correct_answer(expression: str):
    if not expression:
        return None
    expr = str(expression)
    expr = expr.replace("×", "*").replace("x", "*").replace("X", "*")
    expr = expr.replace("÷", "/")
    expr = expr.replace("＝", "=").replace(" ", "")
    expr = expr.rstrip("=")

    match = re.findall(r"-?\d+(?:\.\d+)?", expr)
    if len(match) < 2:
        return None

    a = parse_number(match[0])
    b = parse_number(match[1])
    if a is None or b is None:
        return None

    if "*" in expr:
        return a * b
    if "/" in expr:
        try:
            return a / b
        except ZeroDivisionError:
            return None
    return None


def enrich_scoring(parsed, ocr_tokens):
    sections = parsed.get("sections", [])
    token_map = {}
    for token in ocr_tokens or []:
        text = token.get("text")
        conf = token.get("conf")
        if not text or not isinstance(conf, (int, float)):
            continue
        token_map[text] = max(conf, token_map.get(text, 0.0))
    for section in sections:
        if section.get("type") == "table":
            results = section.get("results", [])
            if not isinstance(results, list):
                continue
            for result in results:
                numbers = result.get("numbers", [])
                total = result.get("total")
                correct_total = None
                if isinstance(numbers, list) and numbers:
                    try:
                        correct_total = sum([int(parse_number(n) or 0) for n in numbers])
                    except Exception:
                        correct_total = None
                if correct_total is not None:
                    result["correct_answer"] = str(correct_total)
                    result["student_answer"] = str(total) if total is not None else ""
                    total_num = parse_number(total)
                    result["is_correct"] = (total_num == correct_total)
                else:
                    result.setdefault("is_correct", False)

                total_key = re.sub(r"\\D", "", str(total or ""))
                confidence = token_map.get(total_key, 0.0) if total_key else 0.0
                result["answer_confidence"] = max(0.0, min(1.0, float(confidence)))

                if result.get("is_correct") is True:
                    result["mark"] = "○"
                else:
                    result["mark"] = "×"
                result.setdefault("mark_position", "near_student_answer_top_right")

                answer_digits = count_digits(total)
                correct_digits = count_digits(correct_total)
                digits_mismatch = abs(answer_digits - correct_digits) >= 2 and correct_digits > 0
                low_confidence = result["answer_confidence"] < 0.75
                incorrect_low = result.get("is_correct") is False and result["answer_confidence"] < 0.8
                result["needs_review"] = bool(low_confidence or incorrect_low or digits_mismatch)
            continue

        if section.get("type") != "list":
            continue
        items = section.get("items", [])
        if not isinstance(items, list):
            continue
        for item in items:
            expression = item.get("expression")
            answer = item.get("answer")
            correct_answer = item.get("correct_answer")
            confidence = item.get("answer_confidence")
            if correct_answer is None:
                correct_answer = compute_correct_answer(expression)
                if correct_answer is not None:
                    item["correct_answer"] = str(correct_answer)
            user_answer = parse_number(answer)
            correct_answer_num = parse_number(correct_answer)
            if correct_answer_num is not None and user_answer is not None:
                if isinstance(correct_answer_num, float) or isinstance(user_answer, float):
                    item["is_correct"] = abs(float(correct_answer_num) - float(user_answer)) < 1e-9
                else:
                    item["is_correct"] = int(correct_answer_num) == int(user_answer)
            else:
                item.setdefault("is_correct", False)

            if not isinstance(confidence, (int, float)):
                answer_key = re.sub(r"\\D", "", str(answer or ""))
                if answer_key and answer_key in token_map:
                    confidence = token_map[answer_key]
                else:
                    confidence = 0.0
            confidence = max(0.0, min(1.0, float(confidence)))
            item["answer_confidence"] = confidence

            if item.get("is_correct") is True:
                item["mark"] = "○"
            else:
                item["mark"] = "×"
            item.setdefault("mark_position", "near_student_answer_top_right")

            answer_digits = count_digits(answer)
            correct_digits = count_digits(correct_answer_num)
            digits_mismatch = abs(answer_digits - correct_digits) >= 2 and correct_digits > 0
            low_confidence = confidence < 0.75
            incorrect_low = item.get("is_correct") is False and confidence < 0.8
            item["needs_review"] = bool(low_confidence or incorrect_low or digits_mismatch)
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
        header, encoded = data_url.split(',', 1)
        mime_type = header.split(';')[0].split(':')[1] if ':' in header else "image/jpeg"
        image_bytes = base64.b64decode(encoded)

        mock_response = os.getenv("ANZAN_MOCK_RESPONSE")
        if mock_response:
            parsed = json.loads(mock_response)
            if not isinstance(parsed, dict):
                return jsonify(build_fallback_payload())
            parsed["format_type"] = "multi"
            parsed = enrich_scoring(parsed, [])
            return jsonify(parsed)

        prompt = build_prompt(spatial_focus=True)

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

        parsed = parse_json_from_text(response.text)
        if not isinstance(parsed, dict):
            return jsonify(build_fallback_payload())

        if parsed.get("format_type") != "multi":
            parsed["format_type"] = "multi"
        if "sections" not in parsed or not isinstance(parsed["sections"], list):
            return jsonify(build_fallback_payload())

        ocr_tokens = extract_ocr_tokens(image_bytes)
        parsed = enrich_scoring(parsed, ocr_tokens)
        processing_time_ms = int((time.time() - time_start) * 1000)
        parsed["processing_time_ms"] = processing_time_ms
        return jsonify(parsed)

    except Exception as e:
        print(f"Server Error: {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(port=5000, debug=True)
