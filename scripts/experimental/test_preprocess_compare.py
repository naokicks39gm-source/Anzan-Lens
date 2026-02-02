import os
import re
import sys
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract


def ocr_tokens(image, psm):
    config = f"--psm {psm} -c tessedit_char_whitelist=0123456789"
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
    return tokens


def preprocess(image, threshold=180, scale=1.0, mode="binarize"):
    img = image.convert("L")
    if scale != 1.0:
        w, h = img.size
        img = img.resize((int(w * scale), int(h * scale)))
    img = ImageEnhance.Contrast(img).enhance(1.6)
    img = img.filter(ImageFilter.SHARPEN)
    if mode == "binarize":
        img = img.point(lambda x: 255 if x > threshold else 0)
    elif mode == "gray":
        pass
    return img


def score(tokens):
    if not tokens:
        return 0.0, 0
    avg = sum(c for _, c in tokens) / len(tokens)
    return avg, len(tokens)


def run_variant(image, name, threshold, scale, psm, mode):
    img = preprocess(image, threshold=threshold, scale=scale, mode=mode)
    tokens = ocr_tokens(img, psm)
    avg_conf, count = score(tokens)
    return {
        "name": name,
        "threshold": threshold,
        "scale": scale,
        "psm": psm,
        "mode": mode,
        "avg_conf": avg_conf,
        "count": count,
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_preprocess_compare.py /path/to/image")
        sys.exit(1)

    path = sys.argv[1]
    image = Image.open(path)

    thresholds = [180, 210]
    scales = [1.0, 1.5]
    psms = [6, 7, 11]
    modes = ["binarize", "gray"]

    results = []
    for t in thresholds:
        for s in scales:
            for p in psms:
                for m in modes:
                    name = f"{m}_thr{t}_x{s}_psm{p}"
                    results.append(run_variant(image, name, t, s, p, m))

    results.sort(key=lambda r: (r["avg_conf"], r["count"]), reverse=True)

    print("Top 10 variants by avg_conf/count:")
    for r in results[:10]:
        print(f"{r['name']}  avg_conf={r['avg_conf']:.1f}  count={r['count']}")

    best = results[0]
    print("\nBest variant:")
    print(best)
