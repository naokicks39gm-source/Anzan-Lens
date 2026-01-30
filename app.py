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


def build_fallback_results():
    results = []
    for i in range(1, 16):
        results.append({"column": i, "numbers": [0, 0, 0, 0, 0, 0, 0], "total": 0})
    return {"results": results}


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
        この画像は「みとりあんざん」の15列構成の計算シートです。
        画像は上段(1-5)、中段(6-10)、下段(11-15)の3段構成になっています。

        各列に含まれる7つの数字を正確に抽出し、合計値(total)と共に
        以下のJSONフォーマットのみで出力してください。
        読み取り不能な箇所は0で埋めて必ずJSONのみを返してください。

        {
          "results": [
            {"column": 1, "numbers": [85, 77, 59, 23, 74, 80, 97], "total": 495},
            ...
          ]
        }
        """

        response = client.models.generate_content(
            model=MODEL_ID,
            contents=[
                prompt,
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            ],
        )

        text_response = (response.text or "").strip()
        json_match = re.search(r'\{.*\}', text_response, re.DOTALL)
        if json_match:
            return jsonify(json.loads(json_match.group()))

        if text_response:
            try:
                return jsonify(json.loads(text_response))
            except Exception:
                pass

        # Fallback: return zeroed structure to avoid hard error
        return jsonify(build_fallback_results())

    except Exception as e:
        print(f"Server Error: {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(port=5000, debug=True)
