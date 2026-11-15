from pathlib import Path
import json
import re
from PIL import Image, ImageDraw

ROOT = Path(r"c:\Users\ysj\Desktop\upload_all\AI_edu_dataset")
PAGES_DIR = ROOT / "intermediate_results" / "pages"
OCR_JSON = ROOT / "intermediate_results" / "ocr_all_pages.json"
OUT_IMG = ROOT / "intermediate_results" / "textbook_refs_verified.png"

CANDIDATE_BOOK_PAGES = {10, 11, 25, 46, 50, 62, 68, 123, 143, 150, 156, 169, 176, 180, 185, 197}


def main() -> None:
    with open(OCR_JSON, "r", encoding="utf-8") as f:
        pages = json.load(f)

    for p in pages:
        if re.search(r"(교과서|미래|사회\s*p)", p["text"], re.IGNORECASE):
            CANDIDATE_BOOK_PAGES.add(p["page"] + 3)

    sorted_pages = sorted(CANDIDATE_BOOK_PAGES)
    print("Checking book pages:", sorted_pages)

    # Render full-height left margin (x: 0.12w .. 0.30w) cropped to non-white bands or top-to-bottom strip
    # Even simpler: check where "인공지능과" / "미래사회" or dark handwriting appears in x [0.12w, 0.28w]
    import numpy as np

    crops = []
    for bp in sorted_pages:
        pp = bp - 3
        if not (1 <= pp <= 198):
            continue
        img = Image.open(PAGES_DIR / f"page_{pp:03d}.png").convert("RGB")
        w, h = img.size
        strip = img.crop((int(0.12 * w), int(0.10 * h), int(0.28 * w), int(0.92 * h)))
        arr = np.asarray(strip)
        dark = (arr[:, :, 0] < 130) & (arr[:, :, 1] < 130) & (arr[:, :, 2] < 130)
        ys = np.where(dark.sum(axis=1) > 2)[0]
        if len(ys) == 0:
            continue
        # Cluster vertically
        chunks = []
        s = prev = ys[0]
        for y in ys[1:]:
            if y - prev > 40:
                chunks.append((s, prev))
                s = y
            prev = y
        chunks.append((s, prev))
        for cy0, cy1 in chunks:
            if 15 <= (cy1 - cy0) <= 180:
                sub = strip.crop((0, max(0, cy0 - 15), strip.width, min(strip.height, cy1 + 15)))
                crops.append((bp, sub))

    cols = 4
    cell_w, cell_h = 300, 140
    rows = (len(crops) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * cell_w, rows * cell_h), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    for idx, (bp, sub) in enumerate(crops):
        r, c = divmod(idx, cols)
        x, y = c * cell_w, r * cell_h
        s = sub.copy()
        s.thumbnail((cell_w - 10, cell_h - 24))
        canvas.paste(s, (x + 5, y + 20))
        draw.text((x + 6, y + 4), f"Book p.{bp:03d}", fill=(0, 0, 180))
    canvas.save(OUT_IMG)
    print(f"Saved {OUT_IMG} with {len(crops)} items")


if __name__ == "__main__":
    main()
