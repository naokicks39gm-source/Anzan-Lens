import os
import re
import sys
from PIL import Image
import pytesseract

sys.path.insert(0, os.path.dirname(__file__))
from app import preprocess_for_ocr, preprocess_for_gemini  # noqa: E402


def count_three_digit_tokens(image, psm=11):
    config = f"--psm {psm} -c tessedit_char_whitelist=0123456789"
    data = pytesseract.image_to_data(image, config=config, output_type=pytesseract.Output.DICT)
    count = 0
    tokens = []
    for text, conf in zip(data.get("text", []), data.get("conf", [])):
        cleaned = re.sub(r"\D", "", str(text or ""))
        if len(cleaned) == 3:
            count += 1
            tokens.append((cleaned, conf))
    return count, tokens


def analyze(image, label):
    w, h = image.size
    right = image.crop((w // 2, 0, w, h))
    count, tokens = count_three_digit_tokens(right)
    print(f"{label}: 3-digit tokens in right half = {count}")
    if tokens:
        print("  sample:", tokens[:10])


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_preprocess_visual.py /path/to/image")
        sys.exit(1)

    path = sys.argv[1]
    img = Image.open(path)

    ocr_img = preprocess_for_ocr(img)
    gem_img = preprocess_for_gemini(img)

    ocr_out = "/tmp/ocr_preprocessed.png"
    gem_out = "/tmp/gemini_preprocessed.png"
    ocr_img.save(ocr_out)
    gem_img.save(gem_out)

    print("Saved:", ocr_out)
    print("Saved:", gem_out)

    analyze(ocr_img, "OCR preprocess")
    analyze(gem_img, "Gemini preprocess")
