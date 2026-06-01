import pdfplumber
import os

for fname in sorted(os.listdir("청년안심주택/경쟁률")):
    if not fname.endswith(".pdf"):
        continue
    with pdfplumber.open(f"청년안심주택/경쟁률/{fname}") as pdf:
        text = pdf.pages[0].extract_text()
        print(f"\n=== {fname} ===")
        print(text[:300] if text else "텍스트 추출 불가")