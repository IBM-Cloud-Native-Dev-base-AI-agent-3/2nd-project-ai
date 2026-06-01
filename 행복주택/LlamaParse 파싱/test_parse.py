import os
from llama_parse import LlamaParse

os.environ["LLAMA_CLOUD_API_KEY"] = "llx-UZyoip0e7FS8a4hvDMx0oH4saPtPOpiK2kiANw8dSktFpYea"

# LlamaParse 설정(한국어, 마크다운형식)
parser = LlamaParse(
    result_type="markdown",
    language="ko"
)

pdf_file_path = "./서울오류 행복주택 예비입주자 모집공고문(2026.05.15).pdf"

print("LlamaParse로 PDF 분석을 시작합니다.(약 10~30초 소요)")

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

print(f"\n 성공! '{output_filename}' 파일이 생성되었습니다. VS CODE에서 열어보세요.")