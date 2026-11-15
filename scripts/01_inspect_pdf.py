from pathlib import Path
import json
import fitz  # PyMuPDF

ROOT = Path(r"c:\Users\ysj\Desktop\upload_all\AI_edu_dataset")
PDF_PATH = ROOT / "인공지능 데이터셋 활용집(2021,keris).pdf"
OUT_DIR = ROOT / "intermediate_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    doc = fitz.open(PDF_PATH)
    meta = doc.metadata
    toc = doc.get_toc(simple=False)
    num_pages = len(doc)

    page_stats = []
    empty_text_pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text").strip()
        images = page.get_images(full=True)
        char_count = len(text)
        if char_count == 0:
            empty_text_pages.append(i + 1)
        page_stats.append(
            {
                "page": i + 1,
                "char_count": char_count,
                "image_count": len(images),
                "width": round(page.rect.width, 1),
                "height": round(page.rect.height, 1),
            }
        )

    summary = {
        "file_name": PDF_PATH.name,
        "file_size_bytes": PDF_PATH.stat().st_size,
        "num_pages": num_pages,
        "metadata": meta,
        "toc_count": len(toc),
        "toc_sample": toc[:50],
        "empty_text_pages_count": len(empty_text_pages),
        "empty_text_pages": empty_text_pages[:30],
        "total_chars": sum(p["char_count"] for p in page_stats),
        "total_images": sum(p["image_count"] for p in page_stats),
    }

    with open(OUT_DIR / "pdf_inspection_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(OUT_DIR / "page_stats.json", "w", encoding="utf-8") as f:
        json.dump(page_stats, f, ensure_ascii=False, indent=2)

    # Save first 15 pages text for quick inspection
    front_text = []
    for i in range(min(15, num_pages)):
        front_text.append(f"=== PAGE {i + 1} ===\n{doc[i].get_text('text')}")
    with open(OUT_DIR / "front_pages_1_15.txt", "w", encoding="utf-8") as f:
        f.write("\n\n".join(front_text))

    print(json.dumps({
        "num_pages": num_pages,
        "toc_count": len(toc),
        "empty_text_pages_count": len(empty_text_pages),
        "total_chars": summary["total_chars"],
        "total_images": summary["total_images"],
        "metadata": meta,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
