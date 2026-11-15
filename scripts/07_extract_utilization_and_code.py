from pathlib import Path
import json

ROOT = Path(r"c:\Users\ysj\Desktop\upload_all\AI_edu_dataset")
OCR_JSON = ROOT / "intermediate_results" / "ocr_all_pages.json"
OUT_TXT = ROOT / "intermediate_results" / "ocr_by_dataset.txt"

DATASETS = [
    ("T01", "보험료 예측 데이터셋", 8, 14),
    ("T02", "학생 수학 성적 예측 데이터셋", 15, 21),
    ("T03", "포도주 품질 예측 데이터셋", 22, 28),
    ("T04", "자전거 대여량 예측 데이터셋", 29, 35),
    ("T05", "타이타닉 생존자 예측 데이터셋", 36, 42),
    ("T06", "스팸 SMS 분류 데이터셋", 43, 48),
    ("T07", "한국어 혐오 분류 데이터셋", 49, 54),
    ("T08", "네이버 영화리뷰 데이터셋", 55, 60),
    ("T09", "한국어 의도 분류 데이터셋", 61, 66),
    ("T10", "한국어 챗봇 데이터셋", 67, 73),
    ("I01", "MNIST", 76, 80),
    ("I02", "Fashion MNIST (목차상 CIFAR10 위치)", 81, 86),
    ("I03", "CIFAR10 (목차상 Fashion MNIST 위치)", 87, 91),
    ("I04", "Dogs vs. Cats", 92, 98),
    ("I05", "SVHN", 99, 105),
    ("I06", "STL-10", 106, 112),
    ("I07", "PASCAL VOC", 113, 120),
    ("I08", "Horse to Zebra (목차상 Zebra2Horse)", 121, 125),
    ("I09", "한국 음식 이미지 데이터셋", 126, 132),
    ("I10", "한국어 글자체 이미지 데이터셋", 133, 139),
    ("V01", "ESC-10", 142, 148),
    ("V02", "Yes/No", 149, 154),
    ("V03", "FSD(Free Spoken Digit)", 155, 160),
    ("V04", "URBAN SOUND", 161, 167),
    ("V05", "Speech Command", 168, 173),
    ("V06", "GTZan", 174, 178),
    ("V07", "LJSpeech", 179, 183),
    ("V08", "Librispeech", 184, 189),
    ("V09", "Bird Audio Detection challenge", 190, 195),
    ("V10", "TIMIT", 196, 201),
]


def main() -> None:
    with open(OCR_JSON, "r", encoding="utf-8") as f:
        pages = json.load(f)
    page_map = {p["page"]: p["text"] for p in pages}

    chunks = []
    for ds_id, title, b_start, b_end in DATASETS:
        chunks.append(f"==================== {ds_id}: {title} (book p.{b_start}-{b_end}) ====================")
        for bp in range(b_start, b_end + 1):
            pp = bp - 3
            chunks.append(f"--- [Book p.{bp:03d} | PDF p.{pp:03d}] ---")
            chunks.append(page_map.get(pp, ""))

    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(chunks))
    print(f"Wrote {OUT_TXT} ({OUT_TXT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
