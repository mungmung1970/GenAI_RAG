# ===============================================================
# 프로그램명: loader.py
# 개요: pdf파일을 읽어들어서 json형태로 저장
# 이력: 2025.12.08 최초 작성
# 기타: txt, html, json을 pyPDFLoade와 PDFPlumberLoader로 저장결과를 비교하여,
#      PDFPlumberLoader의 경우 표인식을 추가하여 저장
#      결과 문단, 표인식이 제대로 되는 것으로 보기 어려움
#      페이지와 다음줄(\n, <br>)정도만 인식 ==> 최종 2024년원천징수의무자를 위한 연말정산신고안내_pypdfloader.txt 파일 사용
# ===============================================================

import os
import json
from langchain_community.document_loaders import PyPDFLoader, PDFPlumberLoader

file_path = r"C:\Users\mungm\Documents\ai_engineer\genai_rag\data\2024년원천징수의무자를 위한 연말정산신고안내.pdf"

# 파일명 추출
base_name = os.path.splitext(os.path.basename(file_path))[0]
save_dir = os.path.dirname(file_path)

# ===============================================================
# PyPDFLoader
# ===============================================================
pypdf_loader = PyPDFLoader(file_path)
pypdf_docs = pypdf_loader.load()

pypdf_text = "\n\n".join([page.page_content for page in pypdf_docs])

# TXT 저장
pypdf_txt_path = os.path.join(save_dir, f"{base_name}_pypdfloader.txt")
with open(pypdf_txt_path, "w", encoding="utf-8") as f:
    f.write(pypdf_text)

# JSON 저장
pypdf_json_path = os.path.join(save_dir, f"{base_name}_pypdfloader.json")
with open(pypdf_json_path, "w", encoding="utf-8") as f:
    json.dump(
        [
            {"page": i + 1, "content": page.page_content}
            for i, page in enumerate(pypdf_docs)
        ],
        f,
        ensure_ascii=False,
        indent=2,
    )

# HTML 저장
pypdf_html_path = os.path.join(save_dir, f"{base_name}_pypdfloader.html")
with open(pypdf_html_path, "w", encoding="utf-8") as f:
    f.write("<html><body>")
    for i, page in enumerate(pypdf_docs):
        content = page.page_content.replace("\n", "<br>")
        f.write(f"<h2>Page {i+1}</h2><p>{content}</p>")
    f.write("</body></html>")


# ===============================================================
# PDFPlumberLoader
# ===============================================================
plumber_loader = PDFPlumberLoader(file_path)
plumber_docs = plumber_loader.load()

plumber_text = "\n\n".join([page.page_content for page in plumber_docs])

# TXT 저장
plumber_txt_path = os.path.join(save_dir, f"{base_name}_pdfplumber.txt")
with open(plumber_txt_path, "w", encoding="utf-8") as f:
    f.write(plumber_text)

# JSON 저장
plumber_json_path = os.path.join(save_dir, f"{base_name}_pdfplumber.json")
with open(plumber_json_path, "w", encoding="utf-8") as f:
    json.dump(
        [
            {"page": i + 1, "content": page.page_content}
            for i, page in enumerate(plumber_docs)
        ],
        f,
        ensure_ascii=False,
        indent=2,
    )

# HTML 저장
plumber_html_path = os.path.join(save_dir, f"{base_name}_pdfplumber.html")
with open(plumber_html_path, "w", encoding="utf-8") as f:
    f.write("<html><body>")
    for i, page in enumerate(plumber_docs):
        content = page.page_content.replace("\n", "<br>")
        f.write(f"<h2>Page {i+1}</h2><p>{content}</p>")
    f.write("</body></html>")

# 완료 메시지
print("PDF parsing 및 파일 저장 완료!")
print(f"- {pypdf_txt_path}")
print(f"- {plumber_txt_path}")
print("JSON / HTML 버전도 동일 경로에 저장됨.")


# ===============================================================
# PDFPlumberLoader_표 인식
# ===============================================================
# 파일명 추출
base_name = os.path.splitext(os.path.basename(file_path))[0]
save_dir = os.path.dirname(file_path)

# ===============================================================
# PyPDFLoader
# ===============================================================
pypdf_loader = PyPDFLoader(file_path)
pypdf_docs = pypdf_loader.load()

pypdf_text = "\n\n".join([page.page_content for page in pypdf_docs])

# TXT 저장
with open(
    os.path.join(save_dir, f"{base_name}_pypdfloader.txt"), "w", encoding="utf-8"
) as f:
    f.write(pypdf_text)

# JSON 저장
with open(
    os.path.join(save_dir, f"{base_name}_pypdfloader.json"), "w", encoding="utf-8"
) as f:
    json.dump(
        [
            {"page": i + 1, "content": page.page_content}
            for i, page in enumerate(pypdf_docs)
        ],
        f,
        ensure_ascii=False,
        indent=2,
    )

# HTML 저장
with open(
    os.path.join(save_dir, f"{base_name}_pypdfloader.html"), "w", encoding="utf-8"
) as f:
    f.write("<html><body>")
    for i, page in enumerate(pypdf_docs):
        content = page.page_content.replace("\n", "<br>")
        f.write(f"<h2>Page {i+1}</h2><p>{content}</p>")
    f.write("</body></html>")


# ===============================================================
# PDFPlumberLoader
# ===============================================================
plumber_loader = PDFPlumberLoader(file_path)
plumber_docs = plumber_loader.load()

plumber_text = "\n\n".join([page.page_content for page in plumber_docs])

with open(
    os.path.join(save_dir, f"{base_name}_pdfplumber.txt"), "w", encoding="utf-8"
) as f:
    f.write(plumber_text)

with open(
    os.path.join(save_dir, f"{base_name}_pdfplumber.json"), "w", encoding="utf-8"
) as f:
    json.dump(
        [
            {"page": i + 1, "content": page.page_content}
            for i, page in enumerate(plumber_docs)
        ],
        f,
        ensure_ascii=False,
        indent=2,
    )

with open(
    os.path.join(save_dir, f"{base_name}_pdfplumber.html"), "w", encoding="utf-8"
) as f:
    f.write("<html><body>")
    for i, page in enumerate(plumber_docs):
        content = page.page_content.replace("\n", "<br>")
        f.write(f"<h2>Page {i+1}</h2><p>{content}</p>")
    f.write("</body></html>")

print("PDF parsing 및 파일 저장 완료!\n")


# ===============================================================
# 본문 + 표 결합 JSON / HTML / MD
# ===============================================================
import pdfplumber

json_out = os.path.join(save_dir, f"{base_name}_content_with_tables.json")
html_out = os.path.join(save_dir, f"{base_name}_content_with_tables.html")
md_out = os.path.join(save_dir, f"{base_name}_content_with_tables.md")

# ① 본문 텍스트
page_texts = [page.page_content.strip() for page in plumber_docs]

# ② 표 추출 --- ★ pdfplumber 라이브러리 사용 ★
page_tables = []
with pdfplumber.open(file_path) as pdf:
    for page in pdf.pages:
        tables = page.extract_tables()
        page_tables.append(tables if tables else [])

# ③ 병합
merged = []
for idx, text in enumerate(page_texts):
    merged.append(
        {
            "page": idx + 1,
            "content": text,
            "tables": [{"rows": table} for table in page_tables[idx]],
        }
    )

# ④ JSON 저장
with open(json_out, "w", encoding="utf-8") as f:
    json.dump(merged, f, ensure_ascii=False, indent=2)

# ⑤ HTML 저장
html = "<html><head><meta charset='UTF-8'></head><body>"
for entry in merged:
    html += f"<h2>📄 Page {entry['page']}</h2>"
    content_html = entry["content"].replace("\n", "<br>")
    html += f"<p>{content_html}</p>"

    for table in entry["tables"]:
        html += "<table border='1' cellspacing='0' cellpadding='4'>"
        for row in table["rows"]:
            html += "<tr>" + "".join([f"<td>{cell}</td>" for cell in row]) + "</tr>"
        html += "</table><br>"

html += "</body></html>"

with open(html_out, "w", encoding="utf-8") as f:
    f.write(html)

# ⑥ Markdown 저장
md = ""
for entry in merged:
    md += f"## 📄 Page {entry['page']}\n\n"
    md += entry["content"] + "\n\n"

    for table in entry["tables"]:
        md += "| " + " | ".join(str(cell) for cell in table["rows"][0]) + " |\n"
        md += "|" + " | ".join("---" for _ in table["rows"][0]) + " |\n"
        for row in table["rows"][1:]:
            md += "| " + " | ".join(str(cell) for cell in row) + " |\n"
        md += "\n"

with open(md_out, "w", encoding="utf-8") as f:
    f.write(md)

print("본문 + 표 JSON / HTML / Markdown 저장 완료")
print(f"📁 JSON: {json_out}")
print(f"📁 HTML: {html_out}")
print(f"📁 MD: {md_out}")
