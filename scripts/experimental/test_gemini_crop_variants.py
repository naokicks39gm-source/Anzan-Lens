import json
import os
import re
import sys
from PIL import Image, ImageEnhance, ImageFilter
from google import genai
from google.genai import types

sys.path.insert(0, os.path.dirname(__file__))
from app import build_prompt  # noqa: E402


def preprocess_light(image, scale=2.0):
    img = image.convert("L")
    if scale != 1.0:
        w, h = img.size
        img = img.resize((int(w * scale), int(h * scale)))
    img = ImageEnhance.Contrast(img).enhance(1.3)
    img = img.filter(ImageFilter.SHARPEN)
    return img


def count_3x3(items):
    count = 0
    for item in items:
        expr = str(item.get("expression", ""))
        expr = expr.replace('×','*').replace('x','*').replace('X','*')
        parts = [p for p in expr.split('*') if p]
        if len(parts) >= 2 and len(parts[0].strip()) == 3 and len(parts[1].strip()) == 3:
            count += 1
    return count


def run_variant(client, image, crop_box, name, out_json):
    cropped = image.crop(crop_box)
    proc = preprocess_light(cropped, scale=2.0)
    png_path = out_json.replace('.json', '.png')
    proc.save(png_path)

    with open(png_path, 'rb') as f:
        b = f.read()

    prompt = build_prompt(spatial_focus=True)
    try:
        response = client.models.generate_content(
            model="models/gemini-2.5-flash",
            contents=[prompt, types.Part.from_bytes(data=b, mime_type="image/png")],
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )
    except Exception:
        response = client.models.generate_content(
            model="models/gemini-2.5-flash",
            contents=[prompt, types.Part.from_bytes(data=b, mime_type="image/png")],
        )

    text = (response.text or "").strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        text = match.group()
    data = json.loads(text)
    with open(out_json, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    kake = None
    for s in data.get('sections', []):
        if 'かけざん' in str(s.get('title', '')):
            kake = s
            break
    items = kake.get('items', []) if kake else []
    return len(items), count_3x3(items), png_path


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_gemini_crop_variants.py /path/to/image")
        sys.exit(1)

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY is not set")

    client = genai.Client(api_key=api_key, http_options=types.HttpOptions(apiVersion="v1"))
    image_path = sys.argv[1]
    image = Image.open(image_path)
    w, h = image.size

    # Crop variants focusing on right-side kakezan block
    variants = [
        ("right_40", (int(w * 0.60), int(h * 0.05), int(w * 0.98), int(h * 0.95))),
        ("right_45", (int(w * 0.55), int(h * 0.05), int(w * 0.98), int(h * 0.95))),
        ("right_50", (int(w * 0.50), int(h * 0.05), int(w * 0.98), int(h * 0.95))),
        ("right_55", (int(w * 0.45), int(h * 0.05), int(w * 0.98), int(h * 0.95))),
    ]

    results = []
    for name, box in variants:
        out_json = f"/tmp/gemini_crop_{name}.json"
        items_count, three_by_three, png_path = run_variant(client, image, box, name, out_json)
        results.append((name, items_count, three_by_three, out_json, png_path))
        print(f"{name}: items={items_count}  3x3={three_by_three}  json={out_json}  png={png_path}")

    best = sorted(results, key=lambda x: (x[2], x[1]), reverse=True)[0]
    print("\nBEST:", best)


if __name__ == "__main__":
    main()
