import base64
import json
import re
from flask import Flask, request, jsonify, render_template
from google import genai
from google.genai import types

app = Flask(__name__)

# --- 設定: APIキーとモデル設定 ---
API_KEY = "AIzaSyCij-dUq3FPFUrB5iqf1W5sTs6HpE1-t-E"
MODEL_ID = "models/gemini-2.5-flash"

client = genai.Client(
    api_key=API_KEY,
    http_options=types.HttpOptions(apiVersion="v1"),
)


def build_fallback_table():
    results = []
    for i in range(1, 16):
        results.append({"column": i, "numbers": [0, 0, 0, 0, 0, 0, 0], "total": 0})
    return {"format_type": "table", "results": results}


def build_fallback_list():
    return {"format_type": "list", "items": []}


def parse_json_from_text(text_response: str):
    text_response = (text_response or "").strip()
    if not text_response:
        return None

    json_match = re.search(r'\{.*\}', text_response, re.DOTALL)
    if json_match:
        return json.loads(json_match.group())

    return json.loads(text_response)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/ocr', methods=['POST'])
def perform_ocr():
    try:
        data = request.json
        data_url = data['image']
        header, encoded = data_url.split(',', 1)
        mime_type = header.split(';')[0].split(':')[1] if ':' in header else "image/jpeg"
        image_bytes = base64.b64decode(encoded)

        prompt = """
        あなたは画像の構造を解析し、最適な形式でデータを抽出するOCRアシスタントです。

        手順:
        1) 画像が「15列の表形式」か「自由形式の問題/リスト形式」かを判断してください。
        2) 形式に応じて、以下のいずれかのJSONだけを出力してください（前後に説明文やMarkdownは不要）。

        A) 15列の表形式（format_type = "table"）:
        {
          "format_type": "table",
          "results": [
            {"column": 1, "numbers": [85, 77, 59, 23, 74, 80, 97], "total": 495},
            ... 15 columns total ...
          ]
        }

        B) 自由形式（format_type = "list"）:
        {
          "format_type": "list",
          "items": [
            {"question": "問題文", "answer": "解答"},
            ...
          ]
        }

        追加ルール:
        - 読み取り不能な数値は0にしてください。
        - 文字の読み取りが困難な場合は空文字で構いません。
        - どちらの形式でも必ずJSONのみを返してください。
        """

        response = client.models.generate_content(
            model=MODEL_ID,
            contents=[
                prompt,
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            ],
        )

        parsed = parse_json_from_text(response.text)
        if not isinstance(parsed, dict):
            return jsonify(build_fallback_list())

        format_type = parsed.get("format_type")
        if format_type == "table":
            return jsonify(parsed)
        if format_type == "list":
            return jsonify(parsed)

        # If model forgot format_type, infer best effort
        if "results" in parsed:
            parsed["format_type"] = "table"
            return jsonify(parsed)
        if "items" in parsed:
            parsed["format_type"] = "list"
            return jsonify(parsed)

        return jsonify(build_fallback_list())

    except Exception as e:
        print(f"Server Error: {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(port=5000, debug=True)
