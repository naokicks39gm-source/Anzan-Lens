import json
import os
import re
import sys
from PIL import Image, ImageEnhance, ImageFilter
from google import genai
from google.genai import types

sys.path.insert(0, os.path.dirname(__file__))
from app import build_prompt  # noqa: E402


def preprocess_variant(image, variant):
    img = image.convert("L")
    if variant.get("scale", 1.0) != 1.0:
        w, h = img.size
        img = img.resize((int(w * variant["scale"]), int(h * variant["scale"])))
    if variant.get("contrast"):
        img = ImageEnhance.Contrast(img).enhance(variant["contrast"])
    if variant.get("sharp", False):
        img = img.filter(ImageFilter.SHARPEN)
    if variant.get("binarize"):
        thr = variant.get("threshold", 200)
        img = img.point(lambda x: 255 if x > thr else 0)
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


def run_variant(client, image_path, variant, out_json):
    img = Image.open(image_path)
    proc = preprocess_variant(img, variant)
    proc.save(out_json.replace('.json', '.png'))

    with open(out_json.replace('.json', '.png'), 'rb') as f:
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
    return len(items), count_3x3(items)


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_gemini_variants.py /path/to/image")
        sys.exit(1)

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY is not set")

    client = genai.Client(api_key=api_key, http_options=types.HttpOptions(apiVersion="v1"))

    variants = [
        {"name": "gray_scale2.0_contrast1.2", "scale": 2.0, "contrast": 1.2, "sharp": True, "binarize": False},
        {"name": "gray_scale2.0_contrast1.4", "scale": 2.0, "contrast": 1.4, "sharp": True, "binarize": False},
        {"name": "bin_thr220_scale1.5", "scale": 1.5, "contrast": 1.3, "sharp": True, "binarize": True, "threshold": 220},
    ]

    image_path = sys.argv[1]
    results = []
    for v in variants:
        out_json = f"/tmp/gemini_variant_{v['name']}.json"
        items_count, three_by_three = run_variant(client, image_path, v, out_json)
        results.append((v["name"], items_count, three_by_three))
        print(f"{v['name']}: items={items_count}  3x3={three_by_three}  json={out_json}")

    best = sorted(results, key=lambda x: (x[2], x[1]), reverse=True)[0]
    print("\nBEST:", best)


if __name__ == "__main__":
    main()
