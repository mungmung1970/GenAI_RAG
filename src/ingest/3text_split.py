# ===============================================================
# 프로그램명: text_split.py
# 개요: 전처리된 JSON 파일을 읽어들여 청킹 후, JSON 파일로 저장
# 이력: 2025.12.09 JSON 기반 버전
# ===============================================================
import os
import json
import re
import uuid

# -----------------------------------------------
# ⚙️ 청킹 설정 (원하면 값만 바꿔서 재조정 가능)
# -----------------------------------------------
MAX_CHARS = 2000
MIN_CHARS = 600
OVERLAP_CHARS = 250
SECTION_BOOST = True  # [SECTION] 등장 시 chunk 경계 강화

# -----------------------------------------------
# 📌 입력 / 출력
# -----------------------------------------------
input_file = r"C:\Users\mungm\Documents\ai_engineer\genai_rag\data\2024년원천징수의무자를 위한 연말정산신고안내_pypdfloader_processing.json"
base_name = os.path.splitext(os.path.basename(input_file))[0]
save_dir = os.path.dirname(input_file)
output_file = os.path.join(save_dir, f"{base_name}_chunks.json")

# -----------------------------------------------
# 🔹 JSON 읽기
# -----------------------------------------------
with open(input_file, "r", encoding="utf-8") as f:
    pages = json.load(f)  # [{"page": int, "content": str, "length": int}, ...]

chunks = []
current = ""
current_page = None


def flush_chunk():
    """현재 chunk를 chunks 리스트에 저장"""
    global current, current_page
    if len(current.strip()) == 0:
        return
    if len(current) < MIN_CHARS and chunks:
        chunks[-1]["chunk"] += "\n" + current
    else:
        chunks.append(
            {
                "chunk_id": str(uuid.uuid4()),
                "chunk": current.strip(),
                "page": current_page,
            }
        )


# -----------------------------------------------
# 🔹 page 단위 content → paragraph 분리 후 chunking
# -----------------------------------------------
for page_obj in pages:
    page_num = page_obj.get("page")
    text = page_obj.get("content", "")

    # 단락 분리
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    for para in paragraphs:
        # 섹션 시작이면 chunk 강제 종료
        if SECTION_BOOST and para.startswith("[SECTION]") and len(current) > 0:
            flush_chunk()
            current = para
            current_page = page_num
            continue

        # 일반 추가
        if len(current) + len(para) + 1 <= MAX_CHARS:
            if not current:
                current_page = page_num
            current += ("\n" if current else "") + para
        else:
            flush_chunk()
            current = para
            current_page = page_num

# 마지막 잔여 chunk 저장
flush_chunk()

# -----------------------------------------------
# 🔹 2차 청킹: 너무 길면 오버랩 분할
# -----------------------------------------------
final_chunks = []
for ch in chunks:
    content = ch["chunk"]
    if len(content) <= MAX_CHARS:
        final_chunks.append(ch)
        continue

    start = 0
    while start < len(content):
        end = start + MAX_CHARS
        piece = content[start:end]
        final_chunks.append(
            {"chunk_id": str(uuid.uuid4()), "chunk": piece.strip(), "page": ch["page"]}
        )
        start = end - OVERLAP_CHARS

# -----------------------------------------------
# 🔹 SECTION 텍스트 추출 + 메타데이터 생성
# -----------------------------------------------
for ch in final_chunks:
    m_sec = re.search(r"\[SECTION\]\s*(.+)", ch["chunk"])
    ch["section"] = m_sec.group(1).strip() if m_sec else None
    ch["source_file"] = os.path.basename(input_file)
    ch["length"] = len(ch["chunk"])

# -----------------------------------------------
# 🔹 JSON 저장
# -----------------------------------------------
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(final_chunks, f, ensure_ascii=False, indent=2)

print("청킹 완료!")
print("총 chunk 수:", len(final_chunks))
print("저장 파일:", output_file)
