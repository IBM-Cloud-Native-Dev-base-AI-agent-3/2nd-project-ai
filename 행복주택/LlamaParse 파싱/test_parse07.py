import os
from llama_parse import LlamaParse

os.environ["LLAMA_CLOUD_API_KEY"] = "llx-UZyoip0e7FS8a4hvDMx0oH4saPtPOpiK2kiANw8dSktFpYea"

parser = LlamaParse(result_type="markdown", language="ko")

pdf_map = {
    "./document_cutline.pdf": "parsed_document_cutline.md",
    "./final_cutline.pdf":    "parsed_final_cutline.md",
}

for pdf_path, out_path in pdf_map.items():
    if not os.path.exists(pdf_path):
        print(f"파일 없음: {pdf_path}")
        continue
    print(f"파싱 중: {pdf_path} ...")
    documents = parser.load_data(pdf_path)
    with open(out_path, "w", encoding="utf-8") as f:
        for doc in documents:
            f.write(doc.text + "\n\n")
    print(f"저장 완료: {out_path}")