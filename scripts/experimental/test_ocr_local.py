import os
import re
import sys
from PIL import Image
import pytesseract

# Reuse preprocessing from app.py without modifying existing logic
sys.path.insert(0, os.path.dirname(__file__))
from app import preprocess_for_ocr  # noqa: E402


def ocr_tokens(image, label):
    config = "--psm 6 -c tessedit_char_whitelist=0123456789"
    data = pytesseract.image_to_data(image, config=config, output_type=pytesseract.Output.DICT)
    tokens = []
    for text, conf in zip(data.get("text", []), data.get("conf", [])):
        cleaned = re.sub(r"\D", "", str(text or ""))
        if not cleaned:
            continue
        try:
            conf_val = float(conf)
        except Exception:
            continue
        if conf_val < 0:
            continue
        tokens.append((cleaned, conf_val))

    print(f"\n[{label}] tokens (text, conf):")
    for t, c in tokens:
        print(f"  {t}  ({c:.1f})")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_ocr_local.py /path/to/image")
        sys.exit(1)

    path = sys.argv[1]
    img = Image.open(path)

    # Preprocess
    pre = preprocess_for_ocr(img)

    # Save a preview for quick check
    out_path = "/tmp/ocr_preprocessed.png"
    pre.save(out_path)
    print("Saved preprocessed image to", out_path)

    w, h = pre.size
    left = pre.crop((0, 0, w // 2, h))
    right = pre.crop((w // 2, 0, w, h))

    ocr_tokens(pre, "full")
    ocr_tokens(left, "left")
    ocr_tokens(right, "right")
