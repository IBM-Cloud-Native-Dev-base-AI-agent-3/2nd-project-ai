import fitz  # PyMuPDF
import os

def pdf_to_png(pdf_path, output_dir, dpi=200):
    os.makedirs(output_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    for i, page in enumerate(doc):
        mat = fitz.Matrix(dpi/72, dpi/72)  # DPI 변환
        pix = page.get_pixmap(matrix=mat)
        out_path = os.path.join(output_dir, f"page_{i+1:03d}.png")
        pix.save(out_path)
        print(f"저장: {out_path}")
        
    print(f"완료: {len(doc)}페이지")
    doc.close()
    

# 서류 심사(99페이지)
pdf_to_png("document_cutline.pdf", "png_document")

# 최종 당첨(44페이지)
pdf_to_png("final_cutline.pdf", "png_final")