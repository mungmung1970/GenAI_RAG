# ===============================================================
# 프로그램명: preprocessing.py
# 개요: txt파일을 읽어들어서 전처리
# 이력: 2025.12.08 최초 작성
# 기타:   기능	가능 여부
#        불필요한 개행/공백 정리	✔
#        페이지 번호/머리글/푸터 제거	✔
#        의미 있는 특수 기호 보존	✔
#        날짜 변환 (’25 → 2025)	✔
#        숫자 패턴 표준화	✔
#        기호 기반 조건 문장 유지 (O, ×, △)	✔
#        용어 표준화(동의어 매핑)	✔
#        문단 단위 재구성	✔
#        섹션/항목 “추정”	⚠ 가능하나 정확도 70~85%
#        표를 문장형으로 “추정 변환”	⚠ 가능하나 정확도 60~85%
#        카테고리 자동 태깅	⚠ LLM 기반이면 높아짐
# ===============================================================

import os
import re

# 입력 파일(txt) 경로
input_file = r"C:\Users\mungm\Documents\ai_engineer\genai_rag\data\2024년원천징수의무자를 위한 연말정산신고안내_pypdfloader.txt"

# 출력 경로 생성
base_name = os.path.splitext(os.path.basename(input_file))[0]
save_dir = os.path.dirname(input_file)
output_file = os.path.join(save_dir, f"{base_name}_processing.txt")

with open(input_file, "r", encoding="utf-8") as f:
    text = f.read()

# ==========================================================
# ① 페이지 헤더/푸터/번호 제거
# ==========================================================
text = re.sub(r"Page\s*\d+\s*/?\s*\d*", " ", text, flags=re.IGNORECASE)
text = re.sub(r"\b페이지\s*\d+\b", " ", text)
text = re.sub(r"\b\d+\s*쪽\b", " ", text)
text = re.sub(r"\n\s*\d+\s*\n", "\n", text)  # 단독 숫자 라인 제거

# ==========================================================
# ② 특수 문자 처리: 의미 있는 기호 유지, 노이즈 제거
# ==========================================================
meaningful_symbols = "O×△"  # 보존해야 하는 문자
noise_symbols = r"[■□●○▲◆◇★☆※▣▧▨¤]"  # 제거 또는 공백
cleaned = []
for c in text:
    if c in meaningful_symbols:
        cleaned.append(c)
    elif re.match(noise_symbols, c):
        cleaned.append(" ")
    else:
        cleaned.append(c)
text = "".join(cleaned)

# ==========================================================
# ③ 연도 약어 변환: ’25 → 2025
# ==========================================================
text = re.sub(r"’(\d{2})", r"20\1", text)

# ==========================================================
# ④ 숫자 패턴 정규화
# ==========================================================
# 1,000만 → 1000만 / 1,000 원 → 1000원 / " 20 만 원" → "20만원"
text = re.sub(r"(?<=\d),(?=\d{3})", "", text)  # 1,000 → 1000
text = re.sub(r"(\d+)\s*만\s*원", r"\1만원", text)
text = re.sub(r"(\d+)\s*원", r"\1원", text)
text = re.sub(r"(\d+)\s*만", r"\1만", text)

# ==========================================================
# ⑤ 리스트/항목 단위 줄바꿈 보정
# ==========================================================
bullet_patterns = [
    r"^\s*[•●◦·\-]\s+",
    r"^\s*\d+\)\s+",
    r"^\s*\(\d+\)\s+",
    r"^\s*[가-힣]\)\s+",
]
for bp in bullet_patterns:
    text = re.sub(bp, lambda m: "\n" + m.group(0), text, flags=re.MULTILINE)

# ==========================================================
# ⑥ 문장 중간의 불필요한 줄바꿈 제거 + 문단 유지
# ==========================================================
text = re.sub(r"(?<![.!?])\n(?=[가-힣A-Za-z0-9])", " ", text)
text = re.sub(r"\n{3,}", "\n\n", text)

# ==========================================================
# ⑦ 제목/섹션 단위 추정 라벨링
# (텍스트 기반 추정: 문장이 짧고 명사 위주면 제목으로 간주)
# ==========================================================
lines = text.split("\n")
processed_lines = []
for line in lines:
    stripped = line.strip()
    if (
        stripped
        and len(stripped) <= 26
        and stripped.count(" ") <= 3
        and not stripped.endswith("원")
    ):
        processed_lines.append(f"[SECTION] {stripped}")
    else:
        processed_lines.append(stripped)
text = "\n".join(processed_lines)

# ==========================================================
# ⑧ 중복 공백 정리 및 앞뒤 공백 제거
# ==========================================================
text = re.sub(r"[ \t]+", " ", text)
text = text.strip()

# 저장
with open(output_file, "w", encoding="utf-8") as f:
    f.write(text)

print("전처리 완료!")
print("저장 파일:", output_file)
