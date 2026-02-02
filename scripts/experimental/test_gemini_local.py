import base64
import json
import os
import re
import sys
import io
from pathlib import Path
from PIL import Image
from google import genai
from google.genai import types

sys.path.insert(0, os.path.dirname(__file__))
from app import build_prompt, preprocess_for_gemini  # noqa: E402


def run(image_path: str, out_json: str):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY is not set")

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(apiVersion="v1"),
    )

    img = Image.open(image_path)
    proc = preprocess_for_gemini(img)
    buf = Path(out_json).with_suffix(".png")
    proc.save(buf)

    b = io_bytes = None
    with open(buf, "rb") as f:
        b = f.read()

    prompt = build_prompt(spatial_focus=True)
    try:
        response = client.models.generate_content(
            model="models/gemini-2.5-flash",
            contents=[
                prompt,
                types.Part.from_bytes(data=b, mime_type="image/png"),
            ],
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )
    except Exception:
        response = client.models.generate_content(
            model="models/gemini-2.5-flash",
            contents=[
                prompt,
                types.Part.from_bytes(data=b, mime_type="image/png"),
            ],
        )

    text = (response.text or "").strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        text = match.group()

    data = json.loads(text)
    with open(out_json, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print("saved:", out_json)
    print("preprocessed image:", str(buf))


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python test_gemini_local.py /path/to/image /tmp/out.json")
        sys.exit(1)

    run(sys.argv[1], sys.argv[2])
