# ===============================================================
# 프로그램명: embedding_bge.py  (bge-m3 버전)
# 개요: json파일을 읽어들여 bge-m3 임베딩 생성 후
#       Elasticsearch 인덱스(rag_chunks_bge)에 저장
# 이력: 2025.12.08 최초 작성 / bge-m3 적용
# ===============================================================

import json
import os
import warnings
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from urllib3.exceptions import InsecureRequestWarning
from sentence_transformers import SentenceTransformer

# 🔹 TLS 경고 숨김 (로컬 self-signed https 용)
warnings.filterwarnings("ignore", category=InsecureRequestWarning)

# ① 환경 변수 로드
load_dotenv()

# ES 접속 정보
ES_URL = os.getenv("ES_URL", "https://localhost:9200")
ES_USER = os.getenv("ES_USER", "elastic")
ES_PASSWORD = os.getenv("ES_PASSWORD", "elastic")
ES_INDEX = os.getenv("ES_INDEX", "rag_chunks_bge")  # ★ bge용 인덱스

# ② bge-m3 임베딩 모델 로드
#    - 기본 CPU 사용, GPU 있으면 device="cuda" 로 변경 가능
print("🔄 bge-m3 모델 로딩 중 ...")
embed_model = SentenceTransformer(
    "BAAI/bge-m3",
    device="cpu",  # 필요 시 "cuda"
)
print("✅ bge-m3 로딩 완료")

# ③ Elasticsearch 클라이언트
es = Elasticsearch(
    ES_URL,
    basic_auth=(ES_USER, ES_PASSWORD),
    verify_certs=False,
)

print("🔗 Elasticsearch 연결 완료:", es.info()["version"]["number"])
print(f"📌 대상 인덱스: {ES_INDEX}")

# ④ JSON 파일 로드
json_path = r"C:\Users\mungm\Documents\ai_engineer\genai_rag\data\2024년원천징수의무자를 위한 연말정산신고안내_pypdfloader_processing_chunks.json"

with open(json_path, "r", encoding="utf-8") as f:
    chunks = json.load(f)

print(f"📂 로드된 청크 수: {len(chunks)}")

# ⑤ 반복하며 embedding 생성 & ES 저장
for idx, doc in enumerate(chunks, start=1):
    # text / 메타데이터 추출
    chunk_text = doc["chunk"]
    chunk_id = doc["chunk_id"]
    page = doc.get("page", None)  # JSON에 page 필드가 있다고 가정
    section = doc.get("section", None)
    source_file = doc.get("source_file", None)
    length = doc.get("length", len(chunk_text))

    # 🔹 bge-m3 임베딩 생성 (1024차원)
    #    normalize_embeddings=True → 코사인 유사도에 적합하게 정규화
    vec = embed_model.encode(chunk_text, normalize_embeddings=True)
    embedding = vec.tolist()  # numpy 배열 → Python list

    # ES 저장 문서 구조
    body = {
        "text": chunk_text,
        "metadata": {
            "chunk_id": chunk_id,
            "page": page,
            "source_file": source_file,
            "section": section,
            "length": length,
        },
        "embedding": embedding,
    }

    # id 기반 upsert
    es.index(index=ES_INDEX, id=chunk_id, document=body)

    if idx % 50 == 0:
        print(f"📌 진행률: {idx}/{len(chunks)} chunks 업로드 완료")

# refresh
es.indices.refresh(index=ES_INDEX)
print("🎉 모든 청크 임베딩 및 Elasticsearch 업로드 완료 (bge-m3)")
