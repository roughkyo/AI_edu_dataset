from pathlib import Path
from PIL import Image

ROOT = Path(r"c:\Users\ysj\Desktop\upload_all\AI_edu_dataset")
PAGES_DIR = ROOT / "intermediate_results" / "pages"
MONTAGE_DIR = ROOT / "intermediate_results" / "montages"
MONTAGE_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = [
    # I. 수치/텍스트 (book_start_page)
    ("T01", "보험료 예측 데이터셋", 8),
    ("T02", "학생 수학 성적 예측 데이터셋", 15),
    ("T03", "포도주 품질 예측 데이터셋", 22),
    ("T04", "자전거 대여량 예측 데이터셋", 29),
    ("T05", "타이타닉 생존자 예측 데이터셋", 36),
    ("T06", "스팸 SMS 분류 데이터셋", 43),
    ("T07", "한국어 혐오 분류 데이터셋", 49),
    ("T08", "네이버 영화리뷰 데이터셋", 55),
    ("T09", "한국어 의도 분류 데이터셋", 61),
    ("T10", "한국어 챗봇 데이터셋", 67),
    # II. 이미지
    ("I01", "MNIST", 76),
    ("I02", "CIFAR10", 81),
    ("I03", "Fashion MNIST", 87),
    ("I04", "Dogs & Cats", 92),
    ("I05", "SVHN", 99),
    ("I06", "STL-10", 106),
    ("I07", "PASCAL VOC", 113),
    ("I08", "Zebra2Horse 데이터셋", 121),
    ("I09", "한국 음식 이미지 데이터셋", 126),
    ("I10", "한국어 글자체 이미지 데이터셋", 133),
    # III. 음성
    ("V01", "ESC-10 데이터셋", 142),
    ("V02", "Yes/No 데이터셋", 149),
    ("V03", "FSD(Free Spoken Digit) 데이터셋", 155),
    ("V04", "Urbansound 데이터셋", 161),
    ("V05", "Speech Command 데이터셋", 168),
    ("V06", "GTZan 데이터셋", 174),
    ("V07", "LJSpeech 데이터셋", 179),
    ("V08", "Librispeech 데이터셋", 184),
    ("V09", "Bird", 190),
    ("V10", "TIMIT", 196),
]


def make_2x2_grid(img_paths: list[Path], out_path: Path) -> None:
    imgs = [Image.open(p).resize((900, 1200), Image.Resampling.LANCZOS) for p in img_paths]
    canvas = Image.new("RGB", (1800, 2400), "white")
    positions = [(0, 0), (900, 0), (0, 1200), (900, 1200)]
    for img, pos in zip(imgs, positions):
        canvas.paste(img, pos)
    canvas.save(out_path, optimize=True)


def main() -> None:
    # Pair consecutive datasets (4 pages per montage: ds1_p1, ds1_p2, ds2_p1, ds2_p2)
    for idx in range(0, len(DATASETS), 2):
        ds1 = DATASETS[idx]
        ds2 = DATASETS[idx + 1]
        p1 = ds1[2] - 3
        p2 = ds2[2] - 3
        img_paths = [
            PAGES_DIR / f"page_{p1:03d}.png",
            PAGES_DIR / f"page_{p1 + 1:03d}.png",
            PAGES_DIR / f"page_{p2:03d}.png",
            PAGES_DIR / f"page_{p2 + 1:03d}.png",
        ]
        out_path = MONTAGE_DIR / f"montage_{ds1[0]}_{ds2[0]}.png"
        make_2x2_grid(img_paths, out_path)
    print(f"Created {len(DATASETS) // 2} montages in {MONTAGE_DIR}")


if __name__ == "__main__":
    main()
