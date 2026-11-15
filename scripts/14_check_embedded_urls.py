import urllib.request

URLS = [
    ("T01_code", "https://raw.githubusercontent.com/krish1407/Medical-Cost-Personal-Datasets/master/insurance.csv"),
    ("T02_code", "https://drive.google.com/u/3/uc?id=1ROipU8RCoiw9gXl-78jOTT4BfjGyhg9Y&export=download"),
    ("T03_code", "https://drive.google.com/u/3/uc?id=18w9sj9bwY7nU1z51nqCg3bfAcAU7C-qXA&export=download"),
    ("T04_code", "https://drive.google.com/u/3/uc?id=1scy6yMrhaOLXjhrVbpD70zxk5n6SafSD&export=download"),
    ("T05_code", "https://blog.kakaocdn.net/dn/3rvGg/btqB3XNjWym/F0nUXZeecI8eoZX-L2MMWn0/train.csv?attach=1&knm=tfile.csv"),
    ("T06_code", "https://raw.githubusercontent.com/mohitgupta-omg/Kaggle-SMS-Spam-Collection-Dataset-/master/spam.csv"),
    ("I04_code", "https://drive.google.com/u/3/uc?id=1scy6yMrhaOLXjhrVbpD70zxk5n6SafSD&export=download"),
    ("I04_code_real", "https://drive.google.com/u/3/uc?id=1sc6yMbhAOLXjhRVbpD70zxk5n6SafSD&export=download"),
    ("V01_code", "https://github.com/karolpiczak/ESC-50/archive/master.zip"),
    ("V10_code", "https://data.deepai.org/timit.zip"),
]

for tag, u in URLS:
    try:
        req = urllib.request.Request(u, method="HEAD", headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as r:
            print(f"{tag:14s} | {r.status} | {u}")
    except Exception as e:
        print(f"{tag:14s} | ERR | {e} | {u}")
