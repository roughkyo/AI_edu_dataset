from pathlib import Path
import json
import importlib.util
import pymupdf

ROOT = Path(r"c:\Users\ysj\Desktop\upload_all\AI_edu_dataset")
PDF_PATH = ROOT / "인공지능 데이터셋 활용집(2021,keris).pdf"
RENDER_DIR = ROOT / "intermediate_results" / "sample_pages"
RENDER_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    doc = pymupdf.open(PDF_PATH)
    sample_info = []
    for pno in range(min(12, len(doc))):
        page = doc[pno]
        drawings = page.get_drawings()
        images = page.get_images(full=True)
        pix = page.get_pixmap(dpi=150)
        out_png = RENDER_DIR / f"page_{pno + 1:03d}.png"
        pix.save(out_png)
        sample_info.append({
            "page": pno + 1,
            "drawings_count": len(drawings),
            "images_count": len(images),
            "size": f"{pix.width}x{pix.height}",
        })

    libs = [
        "pytesseract",
        "easyocr",
        "rapidocr_onnxruntime",
        "onnxruntime",
        "cv2",
        "PIL",
        "openai",
        "dotenv",
        "winrt",
    ]
    available_libs = {m: bool(importlib.util.find_spec(m)) for m in libs}
    has_env = (ROOT / ".env").exists()

    print(json.dumps({
        "sample_pages": sample_info,
        "available_libs": available_libs,
        "has_env": has_env,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
