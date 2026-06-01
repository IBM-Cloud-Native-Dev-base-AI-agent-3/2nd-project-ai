import pdfplumber

pdf_path = "청년안심주택/2023년 2차 청년안심주택 당첨자 커트라인.pdf"

with pdfplumber.open(pdf_path) as pdf:
    for i, page in enumerate(pdf.pages):
        text = page.extract_text()
        print(f"\n=== 페이지 {i+1} ===")
        print(text)