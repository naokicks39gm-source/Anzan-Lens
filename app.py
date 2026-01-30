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
