import os
from llama_parse import LlamaParse

os.environ["LLAMA_CLOUD_API_KEY"] = "llx-UZyoip0e7FS8a4hvDMx0oH4saPtPOpiK2kiANw8dSktFpYea"

parser = LlamaParse(result_type="markdown", language="ko")

print("파싱 중: ./final_cutline.pdf ...")
documents = parser.load_data("./final_cutline.pdf")

with open("parsed_final_cutline.md", "w", encoding="utf-8") as f:
    for doc in documents:
        f.write(doc.text + "\n\n")

print("저장 완료: parsed_final_cutline.md")