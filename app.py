import os
import google.generativeai as genai
from flask import Flask, request, jsonify, render_template
from PIL import Image
import io
import base64
import json
import re

app = Flask(__name__)

# NOTE: keep your API key secure; consider using env vars instead of hardcoding
genai.configure(api_key=os.environ.get('GENAI_API_KEY', 'AIzaSyCij-dUq3FPFUrB5iqf1W5sTs6HpE1-t-E'))
model = genai.GenerativeModel('gemini-1.5-flash')


@app.route('/')
def index():
    return render_template('index.html')


def extract_json_from_text(text):
    # Try several strategies to locate a JSON object in the model response
    # 1) Find the first top-level curly-braced JSON block
    match = re.search(r"\{(?:[^{}]|(?R))*\}", text, re.DOTALL)
    if match:
        return match.group()

    # 2) Fallback: find first occurrence of '[' or '{' and attempt to balance
    start = None
    for i, ch in enumerate(text):
        if ch in '{[':
            start = i
            break
    if start is None:
        return None

    stack = []
    for i in range(start, len(text)):
        ch = text[i]
        if ch in '{[':
            stack.append(ch)
        elif ch in '}]':
            if not stack:
                return None
            open_ch = stack.pop()
            if (open_ch == '{' and ch != '}') or (open_ch == '[' and ch != ']'):
                return None
            if not stack:
                return text[start:i+1]
    return None


@app.route('/ocr', methods=['POST'])
def perform_ocr():
    try:
        data = request.get_json(force=True)
        if not data or 'image' not in data:
            return jsonify({"error": "リクエストに'image'フィールドがありません"}), 400

        # handle data URLs like data:image/png;base64,....
        image_field = data['image']
        if ',' in image_field:
            image_data = image_field.split(',', 1)[1]
        else:
            image_data = image_field

        try:
            image_bytes = base64.b64decode(image_data)
            img = Image.open(io.BytesIO(image_bytes))
        except Exception as e:
            return jsonify({"error": f"画像デコード失敗: {e}"}), 400

        prompt = """
        この画像は「みとりあんざん」の問題集です。
        各列の数字を抽出し、以下のJSON形式で出力してください。
        JSON以外のテキストは一切含めないでください。

        {
          "results": [
            {"column": 1, "numbers": [85, 77, 59], "total": 221}
          ]
        }
        """

        # Send prompt and image to the model
        response = model.generate_content([prompt, img])

        text = getattr(response, 'text', str(response))

        json_str = extract_json_from_text(text)
        if not json_str:
            return jsonify({"error": "JSONが見つかりませんでした", "raw": text}), 500

        try:
            parsed = json.loads(json_str)
        except Exception as e:
            return jsonify({"error": f"JSONパースエラー: {e}", "raw": json_str}), 500

        return jsonify(parsed)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(port=5000, debug=True)
import os
import google.generativeai as genai
from flask import Flask, request, jsonify, render_template
from PIL import Image
import io
import base64
import json

app = Flask(__name__)

# APIキー設定
genai.configure(api_key="AIzaSyCij-dUq3FPFUrB5iqf1W5sTs6HpE1-t-E")
model = genai.GenerativeModel('gemini-1.5-flash')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ocr', methods=['POST'])
def perform_ocr():
    try:
        data = request.json
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        img = Image.open(io.BytesIO(image_bytes))

        prompt = """
        この画像は「みとりあんざん（見取暗算）」の問題集です。
        各列に含まれる数字を正確に抽出し、以下のJSONフォーマットのみで出力してください。
        
        {
          "results": [
            {"column": 1, "numbers": [85, 77, 59, 23, 74, 80, 97], "total": 495}
          ]
        }
        ※数値計算も正確に行ってください。JSON以外のテキストは一切不要です。
        """
        response = model.generate_content([prompt, img])
        clean_json = response.text.replace('```json', '').replace('```', '').strip()
        return jsonify(json.loads(clean_json))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)
