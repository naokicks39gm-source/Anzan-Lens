import base64
import json
import os
import re
from flask import Flask, request, jsonify, render_template
from google import genai
from google.genai import types
from dotenv import load_dotenv

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

        if needs_kakezan_rescan(parsed):
            retry_prompt = build_prompt(spatial_focus=True) + "\n再走査: かけざんの4-6番を絶対に取りこぼさず補完してJSONのみ返せ。"
            try:
                retry = client.models.generate_content(
                    model=MODEL_ID,
                    contents=[
                        retry_prompt,
                        types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                    ],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                    ),
                )
            except Exception:
                retry = client.models.generate_content(
                    model=MODEL_ID,
                    contents=[
                        retry_prompt,
                        types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                    ],
                )

            retry_parsed = parse_json_from_text(retry.text)
            if isinstance(retry_parsed, dict) and isinstance(retry_parsed.get("sections"), list):
                parsed = retry_parsed
                if parsed.get("format_type") != "multi":
                    parsed["format_type"] = "multi"

        return jsonify(parsed)

    except Exception as e:
        print(f"Server Error: {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(port=5000, debug=True)
