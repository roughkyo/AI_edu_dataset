from pathlib import Path
import json
import re

ROOT = Path(r"c:\Users\ysj\Desktop\upload_all\AI_edu_dataset")
OCR_JSON = ROOT / "intermediate_results" / "ocr_all_pages.json"
OUT_JSON = ROOT / "intermediate_results" / "dataset_page_analysis.json"

# (id, section, book_start_page)
DATASETS = [
    ("T01", "수치/텍스트", 8),
    ("T02", "수치/텍스트", 15),
    ("T03", "수치/텍스트", 22),
    ("T04", "수치/텍스트", 29),
    ("T05", "수치/텍스트", 36),
    ("T06", "수치/텍스트", 43),
    ("T07", "수치/텍스트", 49),
    ("T08", "수치/텍스트", 55),
    ("T09", "수치/텍스트", 61),
    ("T10", "수치/텍스트", 67),
    ("I01", "이미지", 76),
    ("I02", "이미지", 81),  # actual order in book: Fashion MNIST at 81, CIFAR10 at 87
    ("I03", "이미지", 87),
    ("I04", "이미지", 92),
    ("I05", "이미지", 99),
    ("I06", "이미지", 106),
    ("I07", "이미지", 113),
    ("I08", "이미지", 121),
    ("I09", "이미지", 126),
    ("I10", "이미지", 133),
    ("V01", "음성", 142),
    ("V02", "음성", 149),
    ("V03", "음성", 155),
    ("V04", "음성", 161),
    ("V05", "음성", 168),
    ("V06", "음성", 174),
    ("V07", "음성", 179),
    ("V08", "음성", 184),
    ("V09", "음성", 190),
    ("V10", "음성", 196),
]

# Section end book pages (last book page of each section before next section cover)
SECTION_ENDS = {
    "T10": 73,   # PDF page 70 -> book page 73 (PDF 71 is II. 이미지 표지)
    "I10": 139,  # PDF page 136 -> book page 139 (PDF 137 is III. 음성 표지)
    "V10": 201,  # PDF page 198 -> book page 201
}


def main() -> None:
    with open(OCR_JSON, "r", encoding="utf-8") as f:
        pages = json.load(f)
    page_map = {p["page"]: p["text"] for p in pages}

    results = []
    for idx, (ds_id, section, book_start) in enumerate(DATASETS):
        if ds_id in SECTION_ENDS:
            book_end = SECTION_ENDS[ds_id]
        else:
            book_end = DATASETS[idx + 1][2] - 1

        pdf_start = book_start - 3
        pdf_end = book_end - 3
        texts = [page_map.get(p, "") for p in range(pdf_start, pdf_end + 1)]
        full_text = "\n".join(texts)

        # Detect features in OCR text
        textbook_refs = re.findall(r"(?:인공지능과\s*미래사회|미래사회)\s*[pP]\.?\s*(\d+)", full_text)
        has_code = bool(re.search(r"(import\s+|from\s+\w+\s+import|torch|tensorflow|keras|sklearn|librosa|데이터셋\s*코드)", full_text, re.IGNORECASE))
        code_pages = sum(
            1
            for t in texts
            if re.search(r"(import\s+|def\s+|torch\.|plt\.|np\.|pd\.|librosa\.|데이터셋\s*코드)", t)
        )

        libs = []
        for lib, pat in [
            ("PyTorch", r"\b(torch|torchvision|torchaudio)\b"),
            ("TensorFlow/Keras", r"\b(tensorflow|keras|tf\.)\b"),
            ("scikit-learn", r"\b(sklearn|scikit-learn)\b"),
            ("pandas", r"\b(pandas|pd\.)\b"),
            ("numpy", r"\b(numpy|np\.)\b"),
            ("matplotlib/seaborn", r"\b(matplotlib|plt\.|seaborn|sns\.)\b"),
            ("librosa", r"\blibrosa\b"),
            ("soundfile", r"\b(soundfile|sf\.)\b"),
            ("OpenCV/PIL", r"\b(cv2|PIL|Image\.)\b"),
            ("KoNLPy/NLTK", r"\b(konlpy|okt|mecab|nltk)\b"),
            ("Colab", r"\bcolab\b"),
        ]:
            if re.search(pat, full_text, re.IGNORECASE):
                libs.append(lib)

        results.append({
            "id": ds_id,
            "section": section,
            "book_start": book_start,
            "book_end": book_end,
            "page_count": book_end - book_start + 1,
            "pdf_start": pdf_start,
            "pdf_end": pdf_end,
            "textbook_refs": sorted(set(textbook_refs), key=lambda x: int(x)),
            "has_code": has_code,
            "code_pages_est": code_pages,
            "libraries": libs,
            "char_count": len(full_text),
        })

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    for r in results:
        print(f"{r['id']} | p.{r['book_start']:03d}-{r['book_end']:03d} ({r['page_count']}p) | code_p={r['code_pages_est']} | tb={r['textbook_refs']} | libs={r['libraries']}")


if __name__ == "__main__":
    main()
