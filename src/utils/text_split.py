# ===============================================================
# 프로그램명: text_split.py
# 개요: 전처리된 텍스트 파일을 읽어들어서 청킹후, json파일로 저장
# 이력: 2025.12.08 최초 작성
# 기타:
# ===============================================================
import os
import json
import re
import uuid

# -----------------------------------------------
# ⚙️ 청킹 설정 — 원하면 값만 바꿔서 재조정 가능
# -----------------------------------------------
MAX_CHARS = 2000  # chunk 최대 길이 (문자 기준)
MIN_CHARS = 600  # 너무 짧은 chunk 방지
OVERLAP_CHARS = 250  # 겹침 범위
SECTION_BOOST = True  # [SECTION] 등장 시 chunk 경계 강화

# -----------------------------------------------
# 📌 입력 / 출력
# -----------------------------------------------
input_file = r"C:\Users\mungm\Documents\ai_engineer\genai_rag\data\2024년원천징수의무자를 위한 연말정산신고안내_pypdfloader_processing.txt"
base_name = os.path.splitext(os.path.basename(input_file))[0]
save_dir = os.path.dirname(input_file)
output_file = os.path.join(save_dir, f"{base_name}_chunks.json")

# -----------------------------------------------
# 🔹 TXT 읽기
# -----------------------------------------------
with open(input_file, "r", encoding="utf-8") as f:
    text = f.read()

# -----------------------------------------------
# 🔹 문단 기준 분리
# -----------------------------------------------
paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

chunks = []
current = ""


def flush_chunk():
    """현재 chunk를 chunks 리스트에 저장"""
    if len(current.strip()) == 0:
        return
    if len(current) < MIN_CHARS and chunks:
        # 직전 chunk에 병합
        chunks[-1]["chunk"] += "\n" + current
    else:
        chunks.append({"chunk_id": str(uuid.uuid4()), "chunk": current.strip()})


# -----------------------------------------------
# 🔹 1차 청킹: 문단 → chunk
# -----------------------------------------------
for para in paragraphs:
    # SECTION이면 우선적으로 chunk 종료
    if SECTION_BOOST and para.startswith("[SECTION]") and len(current) > 0:
        flush_chunk()
        current = para
        continue

    # 그냥 이어 쓰기
    if len(current) + len(para) + 1 <= MAX_CHARS:
        current += ("\n" if current else "") + para
    else:
        flush_chunk()
        current = para

# 마지막 잔여 chunk 저장
flush_chunk()

# -----------------------------------------------
# 🔹 2차 청킹: chunk가 너무 길 경우 → 토막 분할 + overlap
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

        final_chunks.append({"chunk_id": str(uuid.uuid4()), "chunk": piece.strip()})

        start = end - OVERLAP_CHARS  # 오버랩
        if start < 0:
            start = 0

# -----------------------------------------------
# 🔹 메타데이터 추출 (optional)
#  - [SECTION] 자동 추출
# -----------------------------------------------
for ch in final_chunks:
    m = re.search(r"\[SECTION\]\s*(.+)", ch["chunk"])
    ch["section"] = m.group(1).strip() if m else None
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
