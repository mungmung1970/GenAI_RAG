# ===============================================================
# 프로그램명: preprocessing.py
# 개요:JSON 파일을 읽어들여 텍스트(content) 전처리 후 다시 JSON 저장
# 이력: 2025.12.08 최초 작성
# 기타:   기능	가능 여부
#        불필요한 개행/공백 정리	✔
#        머리글/푸터 제거	✔
#        의미 있는 특수 기호 보존(O, ×, △)	✔
#        날짜 변환 (’25 → 2025)	✔
#        숫자 패턴 표준화	✔
#        용어 표준화(동의어 매핑)	✔
#        문단 단위 재구성	✔
#        섹션/항목 “추정”	⚠ 가능하나 정확도 낮음
# ===============================================================

import os
import re
import json

# 입력 JSON 경로
input_json = r"C:\Users\mungm\Documents\ai_engineer\genai_rag\data\2024년원천징수의무자를 위한 연말정산신고안내_pypdfloader.json"

# 출력 JSON 경로
base_name = os.path.splitext(os.path.basename(input_json))[0]
save_dir = os.path.dirname(input_json)
output_json = os.path.join(save_dir, f"{base_name}_processing.json")

# JSON 로드
with open(input_json, "r", encoding="utf-8") as f:
    pages = json.load(f)

# ============================== 전처리 규칙 =============================
meaningful_symbols = "O×△"
noise_symbols = r"[■□●○▲◆◇★☆※▣▧▨¤]"

bullet_patterns = [
    r"^\s*[•●◦·\-]\s+",
    r"^\s*\d+\)\s+",
    r"^\s*\(\d+\)\s+",
    r"^\s*[가-힣]\)\s+",
]


def preprocess(text: str) -> str:
    """본문 텍스트 전처리"""

    # 특수 문자
    cleaned = []
    for c in text:
        if c in meaningful_symbols:
            cleaned.append(c)
        elif re.match(noise_symbols, c):
            cleaned.append(" ")
        else:
            cleaned.append(c)
    text = "".join(cleaned)

    # 연도 변환
    text = re.sub(r"’(\d{2})", r"20\1", text)

    # 숫자 패턴
    text = re.sub(r"(?<=\d),(?=\d{3})", "", text)
    text = re.sub(r"(\d+)\s*만\s*원", r"\1만원", text)
    text = re.sub(r"(\d+)\s*원", r"\1원", text)
    text = re.sub(r"(\d+)\s*만", r"\1만", text)

    # bullet/항목 줄바꿈
    for bp in bullet_patterns:
        text = re.sub(bp, lambda m: "\n" + m.group(0), text, flags=re.MULTILINE)

    # 줄바꿈 정리
    text = re.sub(r"(?<![.!?])\n(?=[가-힣A-Za-z0-9])", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # SECTION 추정
    lines = text.split("\n")
    processed_lines = []
    for line in lines:
        stripped = line.strip()
        if (
            stripped
            and len(stripped) <= 26
            and stripped.count(" ") <= 3
            and not stripped.endswith("원")
            and not stripped.startswith("[SECTION]")
        ):
            processed_lines.append(f"[SECTION] {stripped}")
        else:
            processed_lines.append(stripped)
    text = "\n".join(processed_lines)

    # 공백 정리
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


# ============================ 본 처리 =============================
result = []
for item in pages:
    page = item.get("page")
    content = item.get("content", "")

    cleaned = preprocess(content)

    result.append(
        {
            "page": page,
            "content": cleaned,  # 전처리 완료된 텍스트
            "length": len(cleaned),
        }
    )

# 저장
with open(output_json, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print("전처리 완료!")
print("저장 파일:", output_json)
