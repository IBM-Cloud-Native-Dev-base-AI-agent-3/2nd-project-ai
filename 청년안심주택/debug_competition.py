import pdfplumber
from collections import defaultdict

def get_lines_by_position(page, y_tol=4):
    words = page.extract_words(keep_blank_chars=False)
    if not words:
        return []
    row_map = defaultdict(list)
    for w in words:
        y_key = round(w["top"] / y_tol) * y_tol
        row_map[y_key].append(w)
    result = []
    for y_key in sorted(row_map.keys()):
        line_words = sorted(row_map[y_key], key=lambda w: w["x0"])
        line_text = " ".join(w["text"] for w in line_words).strip()
        if line_text:
            result.append(line_text)
    return result

pdf_path = "청년안심주택/경쟁률/2025년 2차 청년안심주택(공공임대) 청약경쟁률(게시용).pdf"

with pdfplumber.open(pdf_path) as pdf:
    for i, page in enumerate(pdf.pages):
        print(f"\n========== 페이지 {i+1} ==========")
        lines = get_lines_by_position(page)
        for line in lines:
            print(repr(line))