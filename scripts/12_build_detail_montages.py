from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(r"c:\Users\ysj\Desktop\upload_all\AI_edu_dataset")
PAGES_DIR = ROOT / "intermediate_results" / "pages"
OUT_DIR = ROOT / "intermediate_results" / "detail_montages"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# (id, book_start, book_end)
DATASETS = [
    ("T01", 8, 14),
    ("T02", 15, 21),
    ("T03", 22, 28),
    ("T04", 29, 35),
    ("T05", 36, 42),
    ("T06", 43, 48),
    ("T07", 49, 54),
    ("T08", 55, 60),
    ("T09", 61, 66),
    ("T10", 67, 73),
    ("I01", 76, 80),
    ("I02", 81, 86),
    ("I03", 87, 91),
    ("I04", 92, 98),
    ("I05", 99, 105),
    ("I06", 106, 112),
    ("I07", 113, 120),
    ("I08", 121, 125),
    ("I09", 126, 132),
    ("I10", 133, 139),
    ("V01", 142, 148),
    ("V02", 149, 154),
    ("V03", 155, 160),
    ("V04", 161, 167),
    ("V05", 168, 173),
    ("V06", 174, 178),
    ("V07", 179, 183),
    ("V08", 184, 189),
    ("V09", 190, 195),
    ("V10", 196, 201),
]


def main() -> None:
    # Build detail montage for pages (book_start+2 .. book_end) for each dataset
    for ds_id, b_start, b_end in DATASETS:
        rem_pages = list(range(b_start + 2, b_end + 1))
        cols = 2 if len(rem_pages) <= 4 else 3
        rows = (len(rem_pages) + cols - 1) // cols
        cell_w, cell_h = 900, 1200
        canvas = Image.new("RGB", (cols * cell_w, rows * cell_h), "white")
        draw = ImageDraw.Draw(canvas)
        for idx, bp in enumerate(rem_pages):
            pp = bp - 3
            img = Image.open(PAGES_DIR / f"page_{pp:03d}.png").resize((cell_w, cell_h), Image.Resampling.LANCZOS)
            r, c = divmod(idx, cols)
            canvas.paste(img, (c * cell_w, r * cell_h))
            draw.rectangle([c * cell_w + 5, r * cell_h + 5, c * cell_w + 150, r * cell_h + 32], fill=(255, 255, 200))
            draw.text((c * cell_w + 12, r * cell_h + 10), f"{ds_id} | Book p.{bp:03d}", fill=(180, 0, 0))
        canvas.save(OUT_DIR / f"{ds_id}_detail.png", optimize=True)
    print(f"Created {len(DATASETS)} detail montages in {OUT_DIR}")


if __name__ == "__main__":
    main()
