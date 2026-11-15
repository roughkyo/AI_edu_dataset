from pathlib import Path
import time
import pymupdf

ROOT = Path(r"c:\Users\ysj\Desktop\upload_all\AI_edu_dataset")
PDF_PATH = ROOT / "인공지능 데이터셋 활용집(2021,keris).pdf"
PAGES_DIR = ROOT / "intermediate_results" / "pages"
PAGES_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    t0 = time.time()
    doc = pymupdf.open(PDF_PATH)
    for i, page in enumerate(doc):
        out_path = PAGES_DIR / f"page_{i + 1:03d}.png"
        if not out_path.exists():
            pix = page.get_pixmap(dpi=200)
            pix.save(out_path)
    elapsed = time.time() - t0
    print(f"Rendered {len(doc)} pages in {elapsed:.2f}s -> {PAGES_DIR}")


if __name__ == "__main__":
    main()
