from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(r"c:\Users\ysj\Desktop\upload_all\AI_edu_dataset")
PAGES_DIR = ROOT / "intermediate_results" / "pages"
OUT_IMG = ROOT / "intermediate_results" / "textbook_margin_strips.png"


def main() -> None:
    # At 200 DPI, page size is ~1654 x 2205 (aspect 3:4).
    # Let's inspect one known page (page_007.png = book p.010) to verify coordinates,
    # and scan x in [0.16*W, 0.27*W], y in [0.12*H, 0.90*H] for dark pixels (handwriting + icon).
    crops = []
    for pno in range(5, 199):
        img = Image.open(PAGES_DIR / f"page_{pno:03d}.png").convert("RGB")
        w, h = img.size
        # Left margin strip where "인공지능과 미래사회 p.XX" appears
        x0, x1 = int(0.15 * w), int(0.268 * w)
        y0, y1 = int(0.12 * h), int(0.92 * h)
        strip = img.crop((x0, y0, x1, y1))
        arr = np.asarray(strip)
        # Non-white & non-light-gray/blue-header pixels: look for dark ink (R,G,B < 120)
        dark = (arr[:, :, 0] < 120) & (arr[:, :, 1] < 120) & (arr[:, :, 2] < 120)
        # Find vertical blocks of dark pixels
        row_counts = dark.sum(axis=1)
        if row_counts.max() >= 3 and dark.sum() >= 80:
            # Find bounding box of dark pixels vertically
            ys = np.where(row_counts > 0)[0]
            # Cluster into contiguous chunks with gap <= 40px
            chunks = []
            start = ys[0]
            prev = ys[0]
            for y in ys[1:]:
                if y - prev > 45:
                    chunks.append((start, prev))
                    start = y
                prev = y
            chunks.append((start, prev))
            for cy0, cy1 in chunks:
                ch = cy1 - cy0
                if 20 <= ch <= 160:
                    sub = strip.crop((0, max(0, cy0 - 15), strip.width, min(strip.height, cy1 + 15)))
                    # Check it's not mostly empty
                    sarr = np.asarray(sub)
                    sdark = ((sarr[:, :, 0] < 130) & (sarr[:, :, 1] < 130) & (sarr[:, :, 2] < 130)).sum()
                    if sdark >= 80:
                        crops.append((pno + 3, sub))

    print(f"Found {len(crops)} candidate margin annotations")
    if not crops:
        return

    # Arrange into a neat grid
    cols = 4
    cell_w, cell_h = 260, 140
    rows = (len(crops) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * cell_w, rows * cell_h), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    for idx, (book_p, sub) in enumerate(crops):
        r, c = divmod(idx, cols)
        x, y = c * cell_w, r * cell_h
        sub_resized = sub.copy()
        sub_resized.thumbnail((cell_w - 10, cell_h - 24))
        canvas.paste(sub_resized, (x + 5, y + 20))
        draw.text((x + 6, y + 4), f"Book p.{book_p:03d}", fill=(0, 0, 180))
    canvas.save(OUT_IMG)
    print(f"Saved {OUT_IMG}")


if __name__ == "__main__":
    main()
