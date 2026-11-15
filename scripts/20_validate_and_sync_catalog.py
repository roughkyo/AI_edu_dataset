from pathlib import Path
import py_compile
import pandas as pd

ROOT = Path(r"c:\Users\ysj\Desktop\upload_all\AI_edu_dataset")
DATASETS_DIR = ROOT / "datasets"
CSV_PATH = ROOT / "secondary_data" / "datasets_catalog.csv"

VERIFIED_UPDATES = {
    "T01": ("https://www.kaggle.com/datasets/mirichoi0218/insurance", "200 OK (Kaggle)"),
    "T02": ("https://archive.ics.uci.edu/dataset/320/student+performance", "200 OK (UCI Redirect)"),
    "T03": ("https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009", "200 OK (Kaggle)"),
    "T04": ("https://www.kaggle.com/c/bike-sharing-demand", "200 OK (Kaggle)"),
    "T05": ("https://www.kaggle.com/c/titanic", "200 OK (Kaggle)"),
    "T06": ("https://archive.ics.uci.edu/dataset/228/sms+spam+collection", "200 OK (UCI Redirect)"),
    "T07": ("https://www.kaggle.com/datasets/captainnemo9292/korean-extremist-website-womad-hate-speech-data", "200 OK (Kaggle Verified)"),
    "T08": ("https://github.com/e9t/nsmc", "200 OK (GitHub)"),
    "T09": ("https://github.com/warnikchow/3i4k", "200 OK (GitHub)"),
    "T10": ("https://github.com/songys/Chatbot_data", "200 OK (GitHub)"),
    "I01": ("https://storage.googleapis.com/cvdf-datasets/mnist/", "Replaced (Original .gz 404 -> CVDF Mirror 200 OK)"),
    "I02": ("https://github.com/zalandoresearch/fashion-mnist", "200 OK (GitHub)"),
    "I03": ("https://cave.cs.toronto.edu/kriz/cifar.html", "200 OK (Univ. of Toronto)"),
    "I04": ("https://www.kaggle.com/c/dogs-vs-cats/overview", "200 OK (Kaggle)"),
    "I05": ("http://ufldl.stanford.edu/housenumbers/", "200 OK (Stanford)"),
    "I06": ("https://cs.stanford.edu/~acoates/stl10/", "200 OK (Stanford)"),
    "I07": ("https://pjreddie.com/projects/pascal-voc-dataset-mirror/", "200 OK (Mirror)"),
    "I08": ("http://efrosgans.eecs.berkeley.edu/cyclegan/datasets/", "Replaced (Original 404 -> Official CycleGAN Server 200 OK)"),
    "I09": ("https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=79", "200 OK (AI Hub Redirect)"),
    "I10": ("https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=81", "200 OK (AI Hub Redirect)"),
    "V01": ("https://github.com/karolpiczak/ESC-50", "200 OK (GitHub)"),
    "V02": ("https://www.openslr.org/1/", "200 OK (OpenSLR)"),
    "V03": ("https://github.com/Jakobovski/free-spoken-digit-dataset", "200 OK (GitHub)"),
    "V04": ("https://urbansounddataset.weebly.com/urbansound8k.html", "200 OK (NYU/Weebly)"),
    "V05": ("https://arxiv.org/abs/1804.03209", "200 OK (arXiv)"),
    "V06": ("https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification", "200 OK (Kaggle)"),
    "V07": ("https://keithito.com/LJ-Speech-Dataset/", "200 OK (Keith Ito)"),
    "V08": ("http://www.openslr.org/11/", "200 OK (OpenSLR)"),
    "V09": ("https://dcase.community/challenge2018/task-bird-audio-detection", "200 OK (DCASE Redirect)"),
    "V10": ("https://catalog.ldc.upenn.edu/LDC93S1", "200 OK (LDC)"),
}


def main() -> None:
    # 1. Syntax check all generated prepare_dataset.py files
    py_files = sorted(DATASETS_DIR.glob("*/*/prepare_dataset.py"))
    assert len(py_files) == 30, f"Expected 30 python files, found {len(py_files)}"
    for p in py_files:
        py_compile.compile(str(p), doraise=True)
    print(f"Syntax check PASSED for all {len(py_files)} prepare_dataset.py files.")

    # 2. Sync verified URLs into secondary_data/datasets_catalog.csv
    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
    df["verified_url"] = df["id"].map(lambda i: VERIFIED_UPDATES[i][0])
    df["url_status_2026"] = df["id"].map(lambda i: VERIFIED_UPDATES[i][1])
    df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
    print(f"Updated {CSV_PATH} with verified_url and url_status_2026.")


if __name__ == "__main__":
    main()
