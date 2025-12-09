# ===============================================================
# 프로그램명: embedding.py
# 개요: json파일을 읽어들여 embedding 생성 후 Elasticsearch 벡터DB 저장
# 이력: 2025.12.08 최초 작성 / page 자동 추출 기능 추가
# ===============================================================

import json
import os
import re
import warnings
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from openai import OpenAI
from urllib3.exceptions import InsecureRequestWarning

# 🔹 TLS 경고 숨김
warnings.filterwarnings("ignore", category=InsecureRequestWarning)

# ① 환경 변수 로드
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY 없음")

# ② OpenAI 클라이언트 (embedding)
client = OpenAI(api_key=OPENAI_API_KEY)

# ③ Elasticsearch 클라이언트
es = Elasticsearch(
    "https://localhost:9200",
    basic_auth=("elastic", "elastic"),
    verify_certs=False,
)

print("🔗 Elasticsearch 연결 완료:", es.info()["version"]["number"])

# ④ JSON 파일 로드
json_path = r"C:\Users\mungm\Documents\ai_engineer\genai_rag\data\2024년원천징수의무자를 위한 연말정산신고안내_pypdfloader_processing_chunks.json"

with open(json_path, "r", encoding="utf-8") as f:
    chunks = json.load(f)

# ⑤ 반복하며 embedding 생성 & ES 저장
for idx, doc in enumerate(chunks, start=1):

    chunk_text = doc["chunk"]

    # 🔍 [PAGE] 5 같은 패턴 자동 추출
    page = None
    matches = re.findall(r"\[PAGE\]\s*(\d+)", chunk_text)
    if matches:
        page = int(matches[-1])  # 마지막 숫자 사용 (가장 최신 페이지)
        chunk_text = re.sub(r"\[PAGE\]\s*\d+", "", chunk_text).strip()  # 본문에서 제거

    # 실제 OpenAI 임베딩 호출
    resp = client.embeddings.create(model="text-embedding-3-small", input=chunk_text)
    embedding = resp.data[0].embedding

    # ES 저장 문서 구조
    body = {
        "text": chunk_text,
        "metadata": {
            "chunk_id": doc["chunk_id"],
            "page": page,  # 자동 추출된 page
            "source_file": doc.get("source_file", None),
            "section": doc.get("section", None),
            "length": len(chunk_text),
        },
        "embedding": embedding,
    }

    # id 기반 upsert
    es.index(index="rag_chunks", id=doc["chunk_id"], document=body)

    if idx % 50 == 0:
        print(f"📌 진행률: {idx}/{len(chunks)} chunks 업로드 완료")

# refresh
es.indices.refresh(index="rag_chunks")
print("🎉 모든 청크 임베딩 및 Elasticsearch 업로드 완료")
