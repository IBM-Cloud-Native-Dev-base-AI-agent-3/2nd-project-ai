import fitz
import os

# 커트라인 PDF
CUTLINE_DIR = "청년안심주택"          
CUTLINE_OUT = "png_youth_cutline"
os.makedirs(CUTLINE_OUT, exist_ok=True)

cutline_keywords = ["서류심사", "당첨자", "최종당첨자", "커트라인"]

for pdf_file in sorted(os.listdir(CUTLINE_DIR)):
    if not pdf_file.endswith(".pdf"):
        continue
    if not any(kw in pdf_file for kw in cutline_keywords):
        continue

    pdf_path = os.path.join(CUTLINE_DIR, pdf_file)
    doc = fitz.open(pdf_path)
    prefix = pdf_file.replace(".pdf", "")

    for i, page in enumerate(doc):
        mat = fitz.Matrix(200/72, 200/72)
        pix = page.get_pixmap(matrix=mat)
        out_path = os.path.join(CUTLINE_OUT, f"{prefix}_page_{i+1:03d}.png")
        pix.save(out_path)
        print(f"저장: {out_path}")

print("커트라인 PNG 변환 완료")

# 경쟁률 PDF
COMP_DIR = os.path.join("청년안심주택", "경쟁률")   
COMP_OUT = "png_youth_competition"
os.makedirs(COMP_OUT, exist_ok=True)

for pdf_file in sorted(os.listdir(COMP_DIR)):
    if not pdf_file.endswith(".pdf"):
        continue

    pdf_path = os.path.join(COMP_DIR, pdf_file)
    doc = fitz.open(pdf_path)
    prefix = pdf_file.replace(".pdf", "")

    for i, page in enumerate(doc):
        mat = fitz.Matrix(200/72, 200/72)
        pix = page.get_pixmap(matrix=mat)
        out_path = os.path.join(COMP_OUT, f"{prefix}_page_{i+1:03d}.png")
        pix.save(out_path)
        print(f"저장: {out_path}")

print("경쟁률 PNG 변환 완료")