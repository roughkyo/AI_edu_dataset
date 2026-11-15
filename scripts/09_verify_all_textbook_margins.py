from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(r"c:\Users\ysj\Desktop\upload_all\AI_edu_dataset")
PAGES_DIR = ROOT / "intermediate_results" / "pages"
OUT_IMG = ROOT / "intermediate_results" / "textbook_all_verified.png"


def main() -> None:
    # Check book pages 8..201 (PDF 5..198)
    # Even book page: margin is x in [0.13w, 0.27w]; Odd book page: margin is x in [0.18w, 0.32w]
    # Wait: let's look at page_008 (book 011, odd): dotted line is at ~0.325w, text "인공지능과 미래사회 p.90" is x in [0.18w, 0.315w]
    # Look at page_007 (book 010, even): dotted line is at ~0.275w, text "인공지능과 미래사회 p.70" is x in [0.13w, 0.268w]
    crops = []
    for pp in range(5, 199):
        bp = pp + 3
        img = Image.open(PAGES_DIR / f"page_{pp:03d}.png").convert("RGB")
        w, h = img.size
        if bp % 2 == 0:
            x0, x1 = int(0.13 * w), int(0.268 * w)
        else:
            x0, x1 = int(0.18 * w), int(0.318 * w)
        strip = img.crop((x0, int(0.12 * h), x1, int(0.92 * h)))
        arr = np.asarray(strip)
        dark = (arr[:, :, 0] < 120) & (arr[:, :, 1] < 120) & (arr[:, :, 2] < 120)
        ys = np.where(dark.sum(axis=1) >= 2)[0]
        if len(ys) == 0:
            continue
        chunks = []
        s = prev = ys[0]
        for y in ys[1:]:
            if y - prev > 40:
                chunks.append((s, prev))
                s = y
            prev = y
        chunks.append((s, prev))
        for cy0, cy1 in chunks:
            ch = cy1 - cy0
            if 35 <= ch <= 115:
                sub = strip.crop((0, max(0, cy0 - 10), strip.width, min(strip.height, cy1 + 10)))
                sarr = np.asarray(sub)
                # Filter out blue/green pills or illustrations (check average saturation is low & dark pixel ratio is handwriting-like)
                sdark = ((sarr[:, :, 0] < 130) & (sarr[:, :, 1] < 130) & (sarr[:, :, 2] < 130)).sum()
                # Check color pixels (illustrations have many saturated pixels)
                max_c = sarr.max(axis=2)
                min_c = sarr.min(axis=2)
                sat_pixels = ((max_c - min_c) > 40).sum()
                if 120 <= sdark <= 1800 and sat_pixels < 150:
                    crops.append((bp, sub))

    print(f"Found {len(crops)} textbook margin candidates")
    cols = 4
    cell_w, cell_h = 300, 100
    rows = max(1, (len(crops) + cols - 1) // cols)
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
    print(f"Saved {OUT_IMG}")


if __name__ == "__main__":
    main()
