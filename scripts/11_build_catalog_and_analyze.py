from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib as mpl

# Configure Korean font for Windows
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(r"c:\Users\ysj\Desktop\upload_all\AI_edu_dataset")
OCR_JSON = ROOT / "intermediate_results" / "ocr_all_pages.json"
SEC_DIR = ROOT / "secondary_data"
INT_DIR = ROOT / "intermediate_results"
VIS_DIR = ROOT / "visualizations"
SEC_DIR.mkdir(parents=True, exist_ok=True)
INT_DIR.mkdir(parents=True, exist_ok=True)
VIS_DIR.mkdir(parents=True, exist_ok=True)

# Ground-truth verified catalog from direct visual inspection of all 30 datasets + margin scans
CATALOG = [
    # id, section, order_in_book, title_kr, title_en, book_start, book_end, difficulty, interest, file_format, primary_task, learning_paradigm, source_platform, url, korean_specific, textbook_pages, edu_adapted, raw_scale_summary, editorial_note
    ("T01", "수치/텍스트", 1, "보험료 예측 데이터셋", "Medical Cost Personal Datasets", 8, 14, 1, 2, "CSV", "회귀(Regression)", "지도학습", "Kaggle", "https://www.kaggle.com/mirichoi0218/insurance", False, "p.70, p.90", True, "1,338행 x 6열(원본 region 제거, 달러->원화 변환)", ""),
    ("T02", "수치/텍스트", 2, "학생 수학 성적 예측 데이터셋", "Student Performance Dataset", 15, 21, 2, 3, "CSV", "회귀(Regression)", "지도학습", "UCI ML Repository", "https://archive.ics.uci.edu/ml/datasets/Student+Performance", False, "", True, "349행 x 9열(원본 33속성 중 9속성 추출)", ""),
    ("T03", "수치/텍스트", 3, "포도주 품질 예측 데이터셋", "Red Wine Quality", 22, 28, 1, 2, "CSV", "회귀(Regression)", "지도학습", "Kaggle", "https://www.kaggle.com/uciml/red-wine-quality-cortez-et-al-2009", False, "p.70", True, "1,599행 x 8열(원본 12속성 중 8속성 추출)", ""),
    ("T04", "수치/텍스트", 4, "자전거 대여량 예측 데이터셋", "Bike Sharing Demand", 29, 35, 1, 2, "CSV", "회귀(Regression)", "지도학습", "Kaggle", "https://www.kaggle.com/c/bike-sharing-demand", False, "", True, "10,886행 x 12열(원본 날짜 제외)", "p.029 도입문 '총 9가지 속성'과 '12가지 속성' 표기 혼재"),
    ("T05", "수치/텍스트", 5, "타이타닉 생존자 예측 데이터셋", "Titanic", 36, 42, 1, 3, "CSV", "이진 분류(Binary Classification)", "지도학습", "Kaggle", "https://www.kaggle.com/c/titanic", False, "", True, "434행 x 8열(원본 10속성 중 8속성 추출)", ""),
    ("T06", "수치/텍스트", 6, "스팸 SMS 분류 데이터셋", "SMS Spam Collection", 43, 48, 2, 3, "CSV", "텍스트 이진 분류(Text Classification)", "지도학습", "UCI ML Repository", "https://archive.ics.uci.edu/ml/datasets/sms+spam+collection", False, "p.236, p.240", False, "5,572건 x 2열(정상 87%, 스팸 13%)", ""),
    ("T07", "수치/텍스트", 7, "한국어 혐오 분류 데이터셋", "Korean Extremist Website Womad Hate Speech Data", 49, 54, 3, 3, "CSV", "텍스트 이진 분류(Text Classification)", "지도학습", "Kaggle", "https://www.kaggle.com/captainnemo9292/korean-extremist-website-womad-hate-speech-data", True, "p.236", False, "2,000문장 x 2열(혐오 50%, 일반 50%)", "p.049 '데이터셋 목적' 라벨 중복 표기"),
    ("T08", "수치/텍스트", 8, "네이버 영화 리뷰데이터(NSMC)", "Naver Sentiment Movie Corpus (NSMC)", 55, 60, 3, 4, "TXT", "텍스트 이진 분류(Sentiment Analysis)", "지도학습", "GitHub", "https://github.com/e9t/nsmc.git", True, "", False, "200,000건(train 150k, test 50k) x 3열", ""),
    ("T09", "수치/텍스트", 9, "한국어 의도 분류 데이터셋", "3i4k (Intention Identification in Korean)", 61, 66, 4, 3, "TXT", "텍스트 다중 분류(Intent Classification)", "지도학습", "GitHub", "https://github.com/warnikchow/3i4k", True, "p.236", False, "55,134문장 x 2열(7개 의도 클래스)", "p.061 도입문 '전체 2,000개 문장' 오타(T07 잔재)"),
    ("T10", "수치/텍스트", 10, "한국어 챗봇 데이터셋", "Chatbot_data", 67, 73, 4, 3, "CSV", "질의응답/텍스트 생성(QA & Text Generation)", "지도학습", "GitHub", "https://github.com/songys/Chatbot_data", True, "p.236", False, "11,876쌍 x 3열(일상 0, 이별 1, 사랑 2)", ""),

    ("I01", "이미지", 11, "MNIST 데이터셋", "MNIST Datasets", 76, 80, 2, 2, "JPG", "이미지 다중 분류(Image Classification)", "지도학습", "개인/학술 사이트(Yann LeCun)", "http://yann.lecun.com/exdb/mnist/", False, "", True, "원본 70,000장(60k/10k), 코드로 6,000장 축소 추출(28x28 흑백)", ""),
    ("I02", "이미지", 12, "Fashion MNIST 데이터셋", "Fashion MNIST Datasets", 81, 86, 2, 2, "JPG", "이미지 다중 분류(Image Classification)", "지도학습", "GitHub", "https://github.com/zalandoresearch/fashion-mnist", False, "", True, "원본 70,000장(60k/10k), 코드로 축소 추출(28x28 흑백)", "목차(p.2)에서는 3번(087)이나 본문은 2번(081)에 수록"),
    ("I03", "이미지", 13, "CIFAR10 데이터셋", "CIFAR10", 87, 91, 2, 2, "JPG", "이미지 다중 분류(Image Classification)", "지도학습", "학술 사이트(Univ. of Toronto)", "https://www.cs.toronto.edu/~kriz/cifar.html", False, "", True, "원본 60,000장(50k/10k), 코드로 축소 추출(32x32 컬러)", "목차(p.2)에서는 2번(081)이나 본문은 3번(087)에 수록"),
    ("I04", "이미지", 14, "Dogs vs. Cats 데이터셋", "Dogs vs. Cats Datasets", 92, 98, 3, 3, "JPG", "이미지 이진 분류(Image Classification)", "지도학습", "Kaggle", "https://www.kaggle.com/c/dogs-vs-cats/overview", False, "", False, "37,500장(train 25k, test 12.5k), 가변 해상도 컬러", ""),
    ("I05", "이미지", 15, "SVHN 데이터셋", "Street View House Numbers (SVHN)", 99, 105, 3, 4, "PNG+MAT", "이미지 다중 분류(Image Classification)", "지도학습", "학술 사이트(Stanford)", "http://ufldl.stanford.edu/housenumbers", False, "", True, "train 73,257 + test 26,032 + extra 531,131(32x32 컬러)", ""),
    ("I06", "이미지", 16, "STL-10 데이터셋", "STL-10", 106, 112, 4, 3, "MAT/BIN(+PNG)", "이미지 분류/비지도 표현학습", "지도+비지도학습", "학술 사이트(Stanford)", "https://cs.stanford.edu/~acoates/stl10/", False, "", False, "10클래스(train 5,000, test 8,000) + 비지도 100,000장(96x96)", ""),
    ("I07", "이미지", 17, "PASCAL VOC 데이터셋", "Pascal VOC Dataset", 113, 120, 5, 4, "JPG", "객체 검출(Object Detection)", "지도학습", "미러 사이트(pjreddie)", "https://pjreddie.com/projects/pascal-voc-dataset-mirror/", False, "", False, "VOC2007 5,011장(객체 12,608) / VOC2012 11,540장(객체 27,450), 20클래스", ""),
    ("I08", "이미지", 18, "Horse to Zebra 데이터셋", "Horse2zebra dataset", 121, 125, 4, 3, "JPG+CSV", "스타일 이전(Style Transfer)", "비지도학습", "학술 사이트(UC Berkeley)", "https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/", False, "p.241", False, "말 939장 + 얼룩말 1,177장(256x256 컬러)", "목차명 'Zebra2Horse', p.122 구성란에 'Pascal VOC 20클래스' 복붙 오타"),
    ("I09", "이미지", 19, "한국 음식 이미지 데이터셋", "Korean Food Image Dataset", 126, 132, 3, 4, "JPG", "이미지 다중 분류(Image Classification)", "지도학습", "AI Hub", "https://aihub.or.kr/aidata/13594", True, "", True, "원본 150종 150,000장 -> 축소 12종 x 600장 = 7,200장(256x256)", ""),
    ("I10", "이미지", 20, "한국어 글자체 이미지 데이터셋", "Korean font image dataset", 133, 139, 3, 4, "PNG+JSON", "이미지 분류/문자 인식(OCR)", "지도학습", "AI Hub", "https://aihub.or.kr/aidata/133", True, "", False, "실사 포함 이미지 10만장(손글씨 370만자, 인쇄체 280만자)", ""),

    ("V01", "음성", 21, "ESC-10 데이터셋", "Dataset for Environmental Sound Classification", 142, 148, 2, 2, "WAV", "음성/환경음 분류(Audio Classification)", "지도학습", "GitHub", "https://github.com/karolpiczak/ESC-50", False, "p.71", True, "ESC-50에서 10클래스 추출, 클래스당 40개 총 400개(5초)", ""),
    ("V02", "음성", 22, "Yes/No 데이터셋", "Yes/No Dataset", 149, 154, 2, 2, "WAV", "음성 이진 분류(Audio Classification)", "지도학습", "OpenSLR", "https://www.openslr.org/1/", False, "p.71", True, "원본 60개 파일(파일당 8발화 -> 개별 분할 가공)", "p.150 구성란에 'ESC-10 데이터셋은 총 10개 클래스' 복붙 오타"),
    ("V03", "음성", 23, "FSD 데이터셋", "Free Spoken Digit (FSD)", 155, 160, 2, 2, "WAV", "음성 다중 분류(Spoken Digit Classification)", "지도학습", "GitHub", "https://github.com/Jakobovski/free-spoken-digit-dataset.git", False, "p.71", False, "10클래스(0~9), 화자 6명 x 50개 x 10 = 3,000개(8kHz)", ""),
    ("V04", "음성", 24, "URBAN SOUND 데이터셋", "URBAN SOUND Dataset (UrbanSound8K)", 161, 167, 2, 2, "WAV", "음성/도시소음 분류(Audio Classification)", "지도학습", "프로젝트 사이트(NYU/Weebly)", "https://urbansounddataset.weebly.com/urbansound8k.html", False, "", False, "10클래스, 8,732개(4초 이내, 5.6GB)", ""),
    ("V05", "음성", 25, "Speech Command 데이터셋", "Speech Command", 168, 173, 3, 3, "WAV", "음성 명령어 분류(Keyword Spotting)", "지도학습", "arXiv", "https://arxiv.org/abs/1804.03209", False, "p.71", True, "35개 키워드(코드에서 10개 클래스 축소 추출 설정)", ""),
    ("V06", "음성", 26, "GTZan 데이터셋", "GTZan", 174, 178, 3, 4, "WAV", "음악 장르 분류(Music Genre Recognition)", "지도학습", "Kaggle", "https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification", False, "p.71", True, "원본 10장르 x 100곡 = 1,000곡(30초) -> 코드로 클래스당 5개 축소", "p.175 원 데이터 설명에 'Speech Command 데이터셋의 경우' 복붙 오타"),
    ("V07", "음성", 27, "LJSpeech 데이터셋", "LJSpeech", 179, 183, 4, 2, "WAV", "음성 인식/합성(Speech Recognition/ASR)", "지도학습", "개인 사이트(Keith Ito)", "https://keithito.com/LJ-Speech-Dataset/", False, "p.71", False, "단일 화자 약 13,100개 발화(1~10초)", ""),
    ("V08", "음성", 28, "Librispeech 데이터셋", "Librispeech", 184, 189, 4, 3, "WAV", "음성 인식(Speech Recognition/ASR)", "지도학습", "OpenSLR", "http://www.openslr.org/11/", False, "p.71", True, "오디오북 다중 화자(100h/360h; 입문용 test-clean 5.4h 권장)", ""),
    ("V09", "음성", 29, "Bird Audio Detection challenge", "Bird Audio Detection challenge", 190, 195, 3, 3, "WAV", "음성 이진 분류/도메인 일반화", "지도학습", "DCASE Community", "http://dcase.community/challenge2018/task-bird-audio-detection", False, "", False, "dev 35,690개(3종) + eval 12,620개(3종), hasbird(0/1)", "p.190 제목 '데이터셋 데이터셋' 중복 표기"),
    ("V10", "음성", 30, "TIMIT 데이터셋", "TIMIT", 196, 201, 3, 2, "WAV", "음소/음성 인식(Phoneme/Speech Recognition)", "지도학습", "LDC(Univ. of Pennsylvania)", "https://catalog.ldc.upenn.edu/LDC93S1", False, "p.71", True, "8방언 630화자 x 10문장 = 6,300개(16kHz; 코드에서 100파일 제한 추출)", "p.196 두 번째 '데이터셋 목적' 라벨에 V09(이진분류/도메인 일반화) 복붙 오타"),
]

LIB_PATTERNS = [
    ("PyTorch", r"\b(torch|torchvision|torchaudio)\b"),
    ("TensorFlow/Keras", r"\b(tensorflow|keras|tf\.)\b"),
    ("scikit-learn", r"\b(sklearn|scikit-learn)\b"),
    ("pandas", r"\b(pandas|pd\.)\b"),
    ("numpy", r"\b(numpy|np\.)\b"),
    ("matplotlib", r"\b(matplotlib|plt\.)\b"),
    ("OpenCV(cv2)", r"\bcv2\b"),
    ("librosa", r"\blibrosa\b"),
    ("soundfile", r"\b(soundfile|sf\.)\b"),
    ("KoNLPy(Okt)", r"\b(konlpy|Okt)\b"),
    ("albumentations", r"\balbumentations\b"),
]


def main() -> None:
    with open(OCR_JSON, "r", encoding="utf-8") as f:
        ocr_pages = {p["page"]: p["text"] for p in json.load(f)}

    rows = []
    for item in CATALOG:
        (
            ds_id, section, order, title_kr, title_en, b_start, b_end,
            diff, intr, fmt, task, paradigm, source, url, kr_spec,
            tb_pages, edu_adapted, scale_sum, ed_note
        ) = item
        p_start, p_end = b_start - 3, b_end - 3
        full_text = "\n".join(ocr_pages.get(p, "") for p in range(p_start, p_end + 1))
        detected_libs = [name for name, pat in LIB_PATTERNS if re.search(pat, full_text, re.IGNORECASE)]
        has_colab = bool(re.search(r"\bcolab\b", full_text, re.IGNORECASE))
        rows.append({
            "id": ds_id,
            "order_in_book": order,
            "section": section,
            "title_kr": title_kr,
            "title_en": title_en,
            "book_start_page": b_start,
            "book_end_page": b_end,
            "page_count": b_end - b_start + 1,
            "difficulty": diff,
            "interest": intr,
            "file_format": fmt,
            "primary_task": task,
            "learning_paradigm": paradigm,
            "source_platform": source,
            "url": url,
            "korean_specific": kr_spec,
            "textbook_linked": bool(tb_pages),
            "textbook_pages": tb_pages,
            "edu_adapted": edu_adapted,
            "detected_libraries": ", ".join(detected_libs),
            "mentions_colab": has_colab,
            "raw_scale_summary": scale_sum,
            "editorial_note": ed_note,
        })

    df = pd.DataFrame(rows)
    csv_path = SEC_DIR / "datasets_catalog.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # Statistical Analysis
    sec_order = ["수치/텍스트", "이미지", "음성"]
    sec_stats = {}
    for sec in sec_order:
        sub = df[df["section"] == sec]
        sec_stats[sec] = {
            "n": int(len(sub)),
            "page_count_sum": int(sub["page_count"].sum()),
            "page_count_mean": round(float(sub["page_count"].mean()), 2),
            "difficulty_mean": round(float(sub["difficulty"].mean()), 2),
            "difficulty_sd": round(float(sub["difficulty"].std(ddof=1)), 2),
            "difficulty_median": float(sub["difficulty"].median()),
            "difficulty_iqr": float(sub["difficulty"].quantile(0.75) - sub["difficulty"].quantile(0.25)),
            "interest_mean": round(float(sub["interest"].mean()), 2),
            "interest_sd": round(float(sub["interest"].std(ddof=1)), 2),
            "interest_median": float(sub["interest"].median()),
            "interest_iqr": float(sub["interest"].quantile(0.75) - sub["interest"].quantile(0.25)),
            "korean_specific_count": int(sub["korean_specific"].sum()),
            "textbook_linked_count": int(sub["textbook_linked"].sum()),
            "edu_adapted_count": int(sub["edu_adapted"].sum()),
        }

    # Non-parametric Kruskal-Wallis H test across the 3 modalities (ordinal 1-5 Likert scales, n=10 each)
    diff_groups = [df[df["section"] == s]["difficulty"].values for s in sec_order]
    intr_groups = [df[df["section"] == s]["interest"].values for s in sec_order]
    kw_diff = stats.kruskal(*diff_groups)
    kw_intr = stats.kruskal(*intr_groups)

    # Spearman correlation between difficulty and interest across all 30 datasets
    sp_corr = stats.spearmanr(df["difficulty"], df["interest"])

    # Library frequencies
    lib_counts = {}
    for libs_str in df["detected_libraries"]:
        for lib in [x.strip() for x in libs_str.split(",") if x.strip()]:
            lib_counts[lib] = lib_counts.get(lib, 0) + 1

    # By section library frequencies
    lib_by_sec = {}
    for sec in sec_order:
        sub = df[df["section"] == sec]
        c = {}
        for libs_str in sub["detected_libraries"]:
            for lib in [x.strip() for x in libs_str.split(",") if x.strip()]:
                c[lib] = c.get(lib, 0) + 1
        lib_by_sec[sec] = c

    summary = {
        "total_datasets": int(len(df)),
        "overall": {
            "difficulty_mean": round(float(df["difficulty"].mean()), 2),
            "difficulty_sd": round(float(df["difficulty"].std(ddof=1)), 2),
            "difficulty_median": float(df["difficulty"].median()),
            "interest_mean": round(float(df["interest"].mean()), 2),
            "interest_sd": round(float(df["interest"].std(ddof=1)), 2),
            "interest_median": float(df["interest"].median()),
            "korean_specific_total": int(df["korean_specific"].sum()),
            "textbook_linked_total": int(df["textbook_linked"].sum()),
            "edu_adapted_total": int(df["edu_adapted"].sum()),
            "editorial_issues_count": int((df["editorial_note"] != "").sum()),
        },
        "by_section": sec_stats,
        "hypothesis_tests": {
            "kruskal_wallis_difficulty": {"H": round(float(kw_diff.statistic), 4), "p": round(float(kw_diff.pvalue), 4)},
            "kruskal_wallis_interest": {"H": round(float(kw_intr.statistic), 4), "p": round(float(kw_intr.pvalue), 4)},
            "spearman_diff_vs_interest": {"rho": round(float(sp_corr.statistic), 4), "p": round(float(sp_corr.pvalue), 4)},
        },
        "source_platforms": df["source_platform"].value_counts().to_dict(),
        "file_formats": df["file_format"].value_counts().to_dict(),
        "learning_paradigms": df["learning_paradigm"].value_counts().to_dict(),
        "library_counts": dict(sorted(lib_counts.items(), key=lambda x: -x[1])),
        "library_by_section": lib_by_sec,
    }

    with open(INT_DIR / "analysis_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Saved CSV and summary JSON.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    # --- Visualizations ---
    palette = {"수치/텍스트": "#3B82F6", "이미지": "#10B981", "음성": "#06B6D4"}

    # Fig 1: Difficulty & Interest distribution + Scatter
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), dpi=180)
    # Panel A: Grouped bar of mean difficulty & interest by section
    x = np.arange(len(sec_order))
    w = 0.34
    diff_means = [sec_stats[s]["difficulty_mean"] for s in sec_order]
    diff_sds = [sec_stats[s]["difficulty_sd"] for s in sec_order]
    intr_means = [sec_stats[s]["interest_mean"] for s in sec_order]
    intr_sds = [sec_stats[s]["interest_sd"] for s in sec_order]

    b1 = axes[0].bar(x - w/2, diff_means, w, yerr=diff_sds, capsize=4, label="난이도 (1~5)", color="#6366F1", alpha=0.88)
    b2 = axes[0].bar(x + w/2, intr_means, w, yerr=intr_sds, capsize=4, label="흥미도 (1~5)", color="#F59E0B", alpha=0.88)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(sec_order)
    axes[0].set_ylim(0, 5.5)
    axes[0].set_ylabel("점수 (별 개수)")
    axes[0].set_title(f"분야별 난이도·흥미도 (평균 ± SD)\nKW 난이도 p={kw_diff.pvalue:.3f}, 흥미도 p={kw_intr.pvalue:.3f}")
    axes[0].legend(frameon=False)
    for bar, sd in zip(b1, diff_sds):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + sd + 0.1, f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8.5)
    for bar, sd in zip(b2, intr_sds):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + sd + 0.1, f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8.5)

    # Panel B: Difficulty vs Interest Bubble Matrix (counts at each (diff, intr))
    counts = df.groupby(["difficulty", "interest"]).size().reset_index(name="n")
    axes[1].scatter(counts["difficulty"], counts["interest"], s=counts["n"] * 180, color="#0EA5E9", alpha=0.55, edgecolors="#0369A1", linewidth=1.5)
    for _, row in counts.iterrows():
        axes[1].text(row["difficulty"], row["interest"], str(int(row["n"])), ha="center", va="center", fontsize=10, fontweight="bold", color="#0F172A")
    axes[1].set_xlim(0.4, 5.6)
    axes[1].set_ylim(1.4, 4.6)
    axes[1].set_xticks([1, 2, 3, 4, 5])
    axes[1].set_yticks([2, 3, 4])
    axes[1].set_xlabel("난이도 (★)")
    axes[1].set_ylabel("흥미도 (★)")
    axes[1].set_title(f"난이도–흥미도 분포 (N=30)\nSpearman ρ={sp_corr.statistic:.3f} (p={sp_corr.pvalue:.3f})")
    axes[1].grid(True, linestyle="--", alpha=0.35)

    # Panel C: Educational Features by Section (Korean, Textbook-linked, Edu-adapted)
    feats = {
        "한국어 특화": [sec_stats[s]["korean_specific_count"] for s in sec_order],
        "교과서 연계 표기": [sec_stats[s]["textbook_linked_count"] for s in sec_order],
        "교육용 축소·가공": [sec_stats[s]["edu_adapted_count"] for s in sec_order],
    }
    w3 = 0.24
    for idx_f, (fname, vals) in enumerate(feats.items()):
        bars = axes[2].bar(x + (idx_f - 1) * w3, vals, w3, label=fname)
        for bar in bars:
            axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15, str(int(bar.get_height())), ha="center", va="bottom", fontsize=8.5)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(sec_order)
    axes[2].set_ylim(0, 11)
    axes[2].set_ylabel("데이터셋 수 (최대 10)")
    axes[2].set_title("분야별 교육적 특성 비교")
    axes[2].legend(frameon=False, fontsize=8.5)

    plt.tight_layout()
    fig.savefig(VIS_DIR / "01_section_characteristics.png", bbox_inches="tight")
    plt.close(fig)

    # Fig 2: Source Platforms, File Formats, and Code Libraries
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=180)

    src_s = df["source_platform"].value_counts().sort_values()
    axes[0].barh(src_s.index, src_s.values, color="#3B82F6", alpha=0.85)
    for i, v in enumerate(src_s.values):
        axes[0].text(v + 0.15, i, str(v), va="center", fontsize=9)
    axes[0].set_xlim(0, src_s.max() + 1.5)
    axes[0].set_title("데이터셋 출처 플랫폼 분포")
    axes[0].set_xlabel("데이터셋 수")

    fmt_s = df["file_format"].value_counts().sort_values()
    axes[1].barh(fmt_s.index, fmt_s.values, color="#10B981", alpha=0.85)
    for i, v in enumerate(fmt_s.values):
        axes[1].text(v + 0.15, i, str(v), va="center", fontsize=9)
    axes[1].set_xlim(0, fmt_s.max() + 2)
    axes[1].set_title("제공 파일 형식 분포")
    axes[1].set_xlabel("데이터셋 수")

    lib_s = pd.Series(lib_counts).sort_values()
    axes[2].barh(lib_s.index, lib_s.values, color="#8B5CF6", alpha=0.85)
    for i, v in enumerate(lib_s.values):
        axes[2].text(v + 0.3, i, str(v), va="center", fontsize=9)
    axes[2].set_xlim(0, lib_s.max() + 3.5)
    axes[2].set_title("수록 코드 내 주요 Python 라이브러리 사용 빈도")
    axes[2].set_xlabel("언급된 데이터셋 수 (중복 포함)")

    plt.tight_layout()
    fig.savefig(VIS_DIR / "02_platforms_formats_libraries.png", bbox_inches="tight")
    plt.close(fig)

    # Fig 3: Full 30-dataset roadmap (Difficulty, Interest, Page Count, Framework)
    fig, ax = plt.subplots(figsize=(13, 10), dpi=180)
    y_pos = np.arange(len(df))[::-1]
    labels = [f"[{r['id']}] {r['title_kr']} (p.{r['book_start_page']:03d})" for _, r in df.iterrows()]
    colors = [palette[r["section"]] for _, r in df.iterrows()]

    ax.barh(y_pos + 0.18, df["difficulty"], height=0.34, color=colors, alpha=0.9, label="난이도 (★)")
    ax.barh(y_pos - 0.18, df["interest"], height=0.34, color="#F59E0B", alpha=0.75, label="흥미도 (★)")

    for yp, (_, r) in zip(y_pos, df.iterrows()):
        tags = []
        if r["korean_specific"]:
            tags.append("한국어")
        if r["textbook_pages"]:
            tags.append(f"교과서({r['textbook_pages']})")
        if r["edu_adapted"]:
            tags.append("교육용가공")
        if r["editorial_note"]:
            tags.append("편집주의")
        tag_str = " | ".join(tags)
        max_x = max(r["difficulty"], r["interest"])
        ax.text(max_x + 0.12, yp, f"{r['primary_task'].split('(')[0].strip()}  [{tag_str}]" if tag_str else r['primary_task'].split('(')[0].strip(), va="center", fontsize=7.8, color="#334155")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlim(0, 8.8)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xlabel("평가 척도 (별 개수 1~5)")
    ax.set_title("『인공지능 데이터셋 활용집』(KERIS, 2021) 수록 30개 데이터셋 전체 프로파일", fontsize=12, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    # Custom legend
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="#3B82F6", label="난이도 - I. 수치/텍스트"),
        Patch(facecolor="#10B981", label="난이도 - II. 이미지"),
        Patch(facecolor="#06B6D4", label="난이도 - III. 음성"),
        Patch(facecolor="#F59E0B", alpha=0.75, label="흥미도 (공통)"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=True, fontsize=8.5)

    plt.tight_layout()
    fig.savefig(VIS_DIR / "03_all_30_datasets_profile.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved 3 visualization figures.")


if __name__ == "__main__":
    main()
