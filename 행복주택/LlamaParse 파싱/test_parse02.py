import os
from llama_parse import LlamaParse

os.environ["LLAMA_CLOUD_API_KEY"] = "llx-UZyoip0e7FS8a4hvDMx0oH4saPtPOpiK2kiANw8dSktFpYea"

# LlamaParse 설정(한국어, 마크다운형식)
parser = LlamaParse(
    result_type="markdown",
    language="ko"
)

pdf_file_path = "./2025년 3차 청년안심주택 서류심사대상자 커트라인.pdf"

print("LlamaParse로 PDF 분석 시작.")

# 파싱 실행
# 해당 pdf 파일을 LlamaCloud 서버로 보내고, 서버에서 AI가 문서를 분석하여 텍스트로 바꾼 결과를 다시 받아옴.
documents = parser.load_data(pdf_file_path)

print("\n 분석 완료. 터미널에 결과를 출력합니다:\n")
print(documents[0].text)

output_filename = "parsed_result.md"
# paresd_result.md 파일을 쓰기(write) 모드로 열고, 한글 깨짐 방지를 위해 utf-8 인코딩 설정
with open(output_filename, "w", encoding="utf-8") as f:
    for doc in documents:
        f.write(doc.text + "\n\n")

print(f"\n 성공! '{output_filename}' 파일이 생성 완료.")