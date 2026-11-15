import re
import urllib.request

URLS = [
    ("T01", "https://www.kaggle.com/mirichoi0218/insurance"),
    ("T03", "https://www.kaggle.com/uciml/red-wine-quality-cortez-et-al-2009"),
    ("T04", "https://www.kaggle.com/c/bike-sharing-demand"),
    ("T05", "https://www.kaggle.com/c/titanic"),
    ("T07", "https://www.kaggle.com/captainnemo9292/korean-extremist-website-womad-hate-speech-data"),
    ("I04", "https://www.kaggle.com/c/dogs-vs-cats/overview"),
    ("V06", "https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification"),
]

for tid, u in URLS:
    try:
        req = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as r:
            html = r.read().decode("utf-8", errors="ignore")
            title = re.findall(r"<title>(.*?)</title>", html, re.S)
            og = re.findall(r'property="og:title"\s+content="([^"]+)"', html)
            has_404 = "couldn't find that page" in html.lower() or "page not found" in html.lower()
            print(f"{tid} | status={r.status} | final={r.geturl()} | title={title} | og={og} | not_found={has_404} | len={len(html)}")
    except Exception as e:
        print(f"{tid} | ERR={e}")
