import io
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Callable, Dict, Any, Tuple

from PIL import Image, ImageEnhance, ImageFilter
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_ID = "models/gemini-2.5-flash"
IMAGE_PATH = os.path.expanduser("~/Downloads/IMG_0231.JPG")
OUT_DIR = "/tmp/anzan_variant_results"
os.makedirs(OUT_DIR, exist_ok=True)

client = genai.Client(api_key=API_KEY, http_options=types.HttpOptions(apiVersion="v1"))


def parse_json_text(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        text = m.group()
    return json.loads(text)


def call_gemini(prompt: str, img_bytes: bytes, mime_type: str = "image/jpeg") -> Dict[str, Any]:
    try:
        resp = client.models.generate_content(
            model=MODEL_ID,
            contents=[prompt, types.Part.from_bytes(data=img_bytes, mime_type=mime_type)],
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )
    except Exception:
        resp = client.models.generate_content(
            model=MODEL_ID,
            contents=[prompt, types.Part.from_bytes(data=img_bytes, mime_type=mime_type)],
        )
    return parse_json_text(resp.text)


def img_bytes(img: Image.Image) -> bytes:
    b = io.BytesIO()
    img.save(b, format="PNG")
    return b.getvalue()


def pre_light(img: Image.Image, scale=1.5, contrast=1.2) -> Image.Image:
    x = img.convert("RGB")
    if scale != 1.0:
        w, h = x.size
        x = x.resize((int(w * scale), int(h * scale)))
    x = ImageEnhance.Contrast(x).enhance(contrast)
    x = x.filter(ImageFilter.SHARPEN)
    return x


def pre_bin(img: Image.Image, thr=210, scale=1.0) -> Image.Image:
    x = img.convert("L")
    if scale != 1.0:
        w, h = x.size
        x = x.resize((int(w * scale), int(h * scale)))
    x = ImageEnhance.Contrast(x).enhance(1.6)
    x = x.filter(ImageFilter.SHARPEN)
    x = x.point(lambda p: 255 if p > thr else 0)
    return x.convert("RGB")


def prompt_general() -> str:
    return """
あなたはOCR抽出器です。画像から みとりざん/かけざん/わりざん を抽出しJSONのみで返してください。
- みとりざん: table/results [{column, numbers, total}]。列ごとに行数は可変。
- かけざん/わりざん: list/items [{number, expression, answer}]。
- かけざんは左右2列(1-3,4-6)を混在させない。
JSON: {"format_type":"multi","sections":[...]} のみ。
"""


def prompt_strict() -> str:
    return """
OCR抽出。JSONのみ。
- みとりざん: 列1-4は原則7個、列5-8は原則8個を読み取る。可変長で numbers を返す。
- かけざん: 1-6を必ず返す。式は3桁×3桁として読み取る。
- わりざん: 1-3を返す。
JSON: {"format_type":"multi","sections":[{"title":"みとりざん","type":"table","results":[{"column":1,"numbers":[],"total":0}]},{"title":"かけざん","type":"list","items":[{"number":"1","expression":"","answer":""}]},{"title":"わりざん","type":"list","items":[{"number":"1","expression":"","answer":""}]}]}
"""


def prompt_mitori_only() -> str:
    return """
みとりざんのみ抽出。JSONのみ。
- resultsに column, numbers, total を返す。
- 列1-4は7個、列5-8は8個を優先して読み取る。
JSON: {"format_type":"multi","sections":[{"title":"みとりざん","type":"table","results":[{"column":1,"numbers":[],"total":0}]}]}
"""


def prompt_kake_only() -> str:
    return """
かけざんのみ抽出。JSONのみ。
- 1〜6を必ず返す。
- 式は3桁×3桁として読み取る。
JSON: {"format_type":"multi","sections":[{"title":"かけざん","type":"list","items":[{"number":"1","expression":"","answer":""}]}]}
"""


def get_section(parsed: Dict[str, Any], key: str):
    for s in parsed.get("sections", []):
        if key in str(s.get("title", "")):
            return s
    return None


def metrics(parsed: Dict[str, Any]) -> Dict[str, Any]:
    mitori = get_section(parsed, "みとり")
    kake = get_section(parsed, "かけざん")

    expected = {1: 7, 2: 7, 3: 7, 4: 7, 5: 8, 6: 8, 7: 8, 8: 8}
    lens = {}
    if mitori and isinstance(mitori.get("results"), list):
        for r in mitori["results"]:
            try:
                c = int(r.get("column"))
            except Exception:
                continue
            nums = r.get("numbers", [])
            lens[c] = len(nums) if isinstance(nums, list) else 0

    mitori_exact = sum(1 for c, n in expected.items() if lens.get(c, 0) == n)

    kake_items = 0
    kake_3x3 = 0
    if kake and isinstance(kake.get("items"), list):
        items = kake.get("items", [])
        kake_items = len(items)
        for it in items:
            expr = str(it.get("expression", "")).replace("×", "*").replace("x", "*").replace("X", "*")
            parts = [p for p in expr.split("*") if p]
            if len(parts) >= 2 and len(parts[0].strip()) == 3 and len(parts[1].strip()) == 3:
                kake_3x3 += 1

    acc_score = mitori_exact * 10 + min(kake_items, 6) * 2 + kake_3x3 * 6
    return {
        "mitori_exact_cols": mitori_exact,
        "mitori_lens": [lens.get(i, 0) for i in range(1, 9)],
        "kake_items": kake_items,
        "kake_3x3": kake_3x3,
        "acc_score": acc_score,
    }


def merge_sections(base: Dict[str, Any], sec: Dict[str, Any], title_key: str) -> Dict[str, Any]:
    out = json.loads(json.dumps(base))
    sections = out.get("sections", [])
    replaced = False
    for i, s in enumerate(sections):
        if title_key in str(s.get("title", "")):
            sections[i] = sec
            replaced = True
            break
    if not replaced and sec:
        sections.append(sec)
    out["sections"] = sections
    return out


def run_variant(name: str, fn: Callable[[Image.Image, bytes], Dict[str, Any]], base_img: Image.Image, raw_bytes: bytes):
    t0 = time.time()
    parsed = fn(base_img, raw_bytes)
    elapsed = int((time.time() - t0) * 1000)
    m = metrics(parsed)
    m["name"] = name
    m["processing_time_ms"] = elapsed
    with open(os.path.join(OUT_DIR, f"{name}.json"), "w") as f:
        json.dump(parsed, f, ensure_ascii=False, indent=2)
    return m


def main():
    if not API_KEY:
        raise SystemExit("GEMINI_API_KEY is not set")
    if not os.path.exists(IMAGE_PATH):
        raise SystemExit(f"Image not found: {IMAGE_PATH}")

    img = Image.open(IMAGE_PATH)
    with open(IMAGE_PATH, "rb") as f:
        raw = f.read()

    w, h = img.size
    right = img.crop((int(w * 0.45), int(h * 0.05), int(w * 0.98), int(h * 0.95)) )

    variants = {
        "v1_single_orig_general": lambda im, rb: call_gemini(prompt_general(), rb, "image/jpeg"),
        "v2_single_orig_strict": lambda im, rb: call_gemini(prompt_strict(), rb, "image/jpeg"),
        "v3_single_light15_strict": lambda im, rb: call_gemini(prompt_strict(), img_bytes(pre_light(im, 1.5, 1.2)), "image/png"),
        "v4_single_light20_strict": lambda im, rb: call_gemini(prompt_strict(), img_bytes(pre_light(im, 2.0, 1.2)), "image/png"),
        "v5_single_bin210_strict": lambda im, rb: call_gemini(prompt_strict(), img_bytes(pre_bin(im, 210, 1.0)), "image/png"),
        "v6_dual_full_strict_mitori": lambda im, rb: merge_sections(
            call_gemini(prompt_general(), rb, "image/jpeg"),
            get_section(call_gemini(prompt_mitori_only(), rb, "image/jpeg"), "みとり"),
            "みとり",
        ),
        "v7_dual_full_strict_kake": lambda im, rb: merge_sections(
            call_gemini(prompt_general(), rb, "image/jpeg"),
            get_section(call_gemini(prompt_kake_only(), rb, "image/jpeg"), "かけざん"),
            "かけざん",
        ),
        "v8_dual_full_mitori_kake": lambda im, rb: merge_sections(
            merge_sections(
                call_gemini(prompt_general(), rb, "image/jpeg"),
                get_section(call_gemini(prompt_mitori_only(), rb, "image/jpeg"), "みとり"),
                "みとり",
            ),
            get_section(call_gemini(prompt_kake_only(), rb, "image/jpeg"), "かけざん"),
            "かけざん",
        ),
        "v9_dual_mitori_full_kake_right": lambda im, rb: merge_sections(
            merge_sections(
                call_gemini(prompt_general(), rb, "image/jpeg"),
                get_section(call_gemini(prompt_mitori_only(), rb, "image/jpeg"), "みとり"),
                "みとり",
            ),
            get_section(call_gemini(prompt_kake_only(), img_bytes(pre_light(right, 2.0, 1.1)), "image/png"), "かけざん"),
            "かけざん",
        ),
        "v10_dual_mitori_rot90_kake_full": lambda im, rb: merge_sections(
            merge_sections(
                call_gemini(prompt_general(), rb, "image/jpeg"),
                get_section(call_gemini(prompt_mitori_only(), img_bytes(im.rotate(90, expand=True)), "image/png"), "みとり"),
                "みとり",
            ),
            get_section(call_gemini(prompt_kake_only(), rb, "image/jpeg"), "かけざん"),
            "かけざん",
        ),
    }

    results = []
    for name, fn in variants.items():
        try:
            r = run_variant(name, fn, img, raw)
            results.append(r)
            print(name, r)
        except Exception as e:
            print(name, "ERROR", e)

    results.sort(key=lambda x: (-x["acc_score"], x["processing_time_ms"]))
    print("\nBEST:")
    if results:
        print(results[0])
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
