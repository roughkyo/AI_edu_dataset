from pathlib import Path
import json
import re

ROOT = Path(r"c:\Users\ysj\Desktop\upload_all\AI_edu_dataset")
OCR_JSON = ROOT / "intermediate_results" / "ocr_all_pages.json"

def main() -> None:
    with open(OCR_JSON, "r", encoding="utf-8") as f:
        pages = json.load(f)

    for p in pages:
        txt = p["text"]
        book_p = p["page"] + 3
        # Find lines mentioning textbook-like patterns or 활용
        hits = []
        for line in p["lines"]:
            if re.search(r"(미래|사회|교과서|활용|코드|colab|kagg|github|aihub|openslr|torch|tensorflow|keras|sklearn|librosa|konlpy|okt|mecab|cv2)", line, re.IGNORECASE):
                hits.append(line.strip())
        if hits:
            # Keep only concise relevant lines
            interesting = [h for h in hits if re.search(r"(미래|p\.\s*\d+|교과서|import |colab|데이터셋 활용|데이터셋 코드|1\s+|2\s+|3\s+|4\s+)", h, re.IGNORECASE)]
            if interesting:
                print(f"[Book p.{book_p:03d}] " + " | ".join(interesting[:6]))

if __name__ == "__main__":
    main()
