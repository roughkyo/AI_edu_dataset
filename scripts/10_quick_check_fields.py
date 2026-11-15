from pathlib import Path
import re

ROOT = Path(r"c:\Users\ysj\Desktop\upload_all\AI_edu_dataset")
TXT = ROOT / "intermediate_results" / "ocr_by_dataset.txt"

def main() -> None:
    content = TXT.read_text(encoding="utf-8")
    sections = content.split("====================")
    for sec in sections:
        if not sec.strip():
            continue
        lines = sec.splitlines()
        header = lines[0].strip()
        if not re.match(r"^(T|I|V)\d{2}:", header):
            continue
        has_util = "데이터셋 활용" in sec or "활용" in sec
        has_diff = "원본 데이터와의 차이" in sec or "원본데이터와의 차이" in sec
        has_example = "데이터셋 예시" in sec or "원 데이터(Raw Data)" in sec or "원 데이터" in sec
        has_code = "데이터셋 코드" in sec or "import " in sec
        print(f"{header[:42]:42s} | 예시={has_example} | 활용={has_util} | 코드={has_code} | 원본차이={has_diff}")

if __name__ == "__main__":
    main()
