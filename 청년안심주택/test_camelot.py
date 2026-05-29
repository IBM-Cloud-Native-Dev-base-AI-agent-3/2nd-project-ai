# test_camelot.py
import camelot

pdf_path = "청년안심주택/2024년 3차 청년안심주택 서류심사대상자 커트라인.pdf"
tables = camelot.read_pdf(pdf_path, flavor="lattice", pages="all")

print(f"표 개수: {len(tables)}")
for i, table in enumerate(tables):
    print(f"\n--- 표 {i+1} ---")
    print(table.df)