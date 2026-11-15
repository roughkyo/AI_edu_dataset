from pathlib import Path
import json
import urllib.request
import urllib.error
import ssl
import pandas as pd

ROOT = Path(r"c:\Users\ysj\Desktop\upload_all\AI_edu_dataset")
CSV_PATH = ROOT / "secondary_data" / "datasets_catalog.csv"
OUT_JSON = ROOT / "intermediate_results" / "url_check_results.json"


def check_url(url: str) -> dict:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=15, context=ctx) as resp:
            final_url = resp.geturl()
            status = resp.getcode()
            body = resp.read(8192).decode("utf-8", errors="ignore")
            title = ""
            import re
            m = re.search(r"<title[^>]*>(.*?)</title>", body, re.IGNORECASE | re.DOTALL)
            if m:
                title = " ".join(m.group(1).split())[:120]
            return {
                "status": status,
                "final_url": final_url,
                "title": title,
                "error": "",
            }
    except urllib.error.HTTPError as e:
        return {
            "status": e.code,
            "final_url": e.geturl() if hasattr(e, "geturl") else url,
            "title": "",
            "error": str(e),
        }
    except Exception as e:
        return {
            "status": -1,
            "final_url": url,
            "title": "",
            "error": str(e),
        }


def main() -> None:
    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
    results = []
    for _, row in df.iterrows():
        res = check_url(row["url"])
        entry = {
            "id": row["id"],
            "title_kr": row["title_kr"],
            "url": row["url"],
            **res,
        }
        results.append(entry)
        print(f"{entry['id']} | {entry['status']} | {entry['url']} -> {entry['final_url']} | {entry['title'][:50]} | {entry['error'][:60]}")

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
