from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(r"c:\Users\ysj\Desktop\upload_all\AI_edu_dataset")
PAGES_DIR = ROOT / "intermediate_results" / "pages"
OUT_IMG = ROOT / "intermediate_results" / "textbook_odd_check.png"

BOOK_PAGES = [11, 18, 19, 25, 32, 33, 39, 40, 57, 58, 123, 143, 169, 185, 197]


def main() -> None:
    # Crop full height of left margin (x: 0.14w .. 0.36w, y: 0.12h .. 0.90h) for quick visual scan
    cols = 5
    cell_w, cell_h = 280, 820
    rows = (len(BOOK_PAGES) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * cell_w, rows * cell_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for idx, bp in enumerate(BOOK_PAGES):
        pp = bp - 3
        img = Image.open(PAGES_DIR / f"page_{pp:03d}.png").convert("RGB")
        w, h = img.size
        strip = img.crop((int(0.14 * w), int(0.12 * h), int(0.36 * w), int(0.90 * h)))
        strip.thumbnail((cell_w - 10, cell_h - 30))
        r, c = divmod(idx, cols)
        x, y = c * cell_w, r * cell_h
        canvas.paste(strip, (x + 5, y + 24))
        draw.text((x + 8, y + 4), f"Book p.{bp:03d}", fill=(200, 0, 0))
    canvas.save(OUT_IMG)
    print(f"Saved {OUT_IMG}")


if __name__ == "__main__":
    main()
