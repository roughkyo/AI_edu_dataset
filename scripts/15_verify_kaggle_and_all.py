import urllib.request
import urllib.error

KAGGLE_DATASETS = [
    ("T01", "datasets", "mirichoi0218/insurance"),
    ("T03", "datasets", "uciml/red-wine-quality-cortez-et-al-2009"),
    ("T04", "competitions", "bike-sharing-demand"),
    ("T05", "competitions", "titanic"),
    ("T07", "datasets", "captainnemo9292/korean-extremist-website-womad-hate-speech-data"),
    ("I04", "competitions", "dogs-vs-cats"),
    ("V06", "datasets", "andradaolteanu/gtzan-dataset-music-genre-classification"),
]

for tid, kind, slug in KAGGLE_DATASETS:
    if kind == "datasets":
        api_url = f"https://www.kaggle.com/api/v1/datasets/view/{slug}"
    else:
        api_url = f"https://www.kaggle.com/api/v1/competitions/list?search={slug}"
    try:
        req = urllib.request.Request(api_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as r:
            body = r.read(300).decode("utf-8", errors="ignore")
            print(f"{tid} | {r.status} | {slug} | {body[:100]}")
    except urllib.error.HTTPError as e:
        print(f"{tid} | HTTP {e.code} | {slug} | {e.reason}")
    except Exception as e:
        print(f"{tid} | ERR | {slug} | {e}")
