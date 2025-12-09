import os
import yaml
import logging
from pathlib import Path
from dotenv import load_dotenv

from elasticsearch import Elasticsearch
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_elasticsearch import ElasticsearchStore


# ─────────────────────────────────────────────
# 🔧 로깅 설정
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("rag")


# ─────────────────────────────────────────────
# 1) 환경 변수 로드
# ─────────────────────────────────────────────
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY 없음")

ES_URL = "https://localhost:9200"
ES_PASSWORD = "elastic"
ES_INDEX = "rag_chunks"


# ─────────────────────────────────────────────
# 2) prompts.yaml 로드
# ─────────────────────────────────────────────
def load_prompt(name: str, version: str = None):
    base_dir = Path(__file__).resolve().parent.parent
    path = base_dir / "prompt" / "prompts.yaml"

    if not path.exists():
        raise FileNotFoundError(f"prompts.yaml을 찾을 수 없습니다: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    for item in data.get("prompts", []):
        if item["name"] == name and (version is None or item["version"] == version):
            return item["template"]

    raise ValueError(f"Prompt not found: {name}, version: {version}")


# ─────────────────────────────────────────────
# 3) Embeddings & ES 연결
# ─────────────────────────────────────────────
log.info("🔌 임베딩 모델 생성 중...")
# ❗ match ES mapping: 1536차원
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)

log.info("🔌 Elasticsearch 연결 시도...")
es = Elasticsearch(
    ES_URL,
    basic_auth=("elastic", ES_PASSWORD),
    verify_certs=False,
)
log.info("✅ Elasticsearch 연결 성공: %s", es.info()["version"]["number"])

# 인덱스 존재 / 문서 수 점검
if not es.indices.exists(index=ES_INDEX):
    log.error(f"❌ Elasticsearch 인덱스 '{ES_INDEX}' 없음")
else:
    count = es.count(index=ES_INDEX)["count"]
    log.info(f"📌 인덱스 '{ES_INDEX}' 문서 수: {count}")

vectorstore = ElasticsearchStore(
    es_connection=es,
    index_name=ES_INDEX,
    embedding=embeddings,
    vector_query_field="embedding",
)


# ─────────────────────────────────────────────
# 4) LLM
# ─────────────────────────────────────────────
llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, temperature=0.2)


# ─────────────────────────────────────────────
# 5) 검색 및 RAG 수행
# ─────────────────────────────────────────────
def retrieve_context(question: str, k: int = 5) -> str:
    log.info("[RETRIEVER] 검색 시작: '%s'", question)

    # 1) 임베딩 벡터 직접 생성
    embedding_vector = embeddings.embed_query(question)
    log.info(f"🔹 embedding length = {len(embedding_vector)}")

    # 2) LangChain이 아닌 '직접' ES에 보낼 KNN 쿼리 생성
    raw_knn_query = {
        "knn": {
            "field": "embedding",  # 반드시 ES 매핑과 동일
            "query_vector": embedding_vector,
            "k": k,
            "num_candidates": 50,  # 성능/정확도 균형
        },
        "_source": ["page_content", "source_file", "page", "section", "chunk_id"],
    }

    # 3) DevTools 실행용으로 JSON 그대로 출력
    log.info(
        "📌 [DevTools용 KNN Query] =================================================="
    )
    log.info(f"POST {ES_INDEX}/_knn_search\n{raw_knn_query}")
    log.info(
        "================================================================================"
    )

    # 4) Elasticsearch에 직접 KNN 쿼리 수행 → similarity_search와 결과 비교
    try:
        es_resp = es.search(index=ES_INDEX, body=raw_knn_query)
    except Exception:
        log.error("❌ ES 직접 조회 실패", exc_info=True)
        return ""

    hits = es_resp.get("hits", {}).get("hits", [])

    if not hits:
        log.warning(
            "⚠ ES 직접 KNN 검색 결과 0개 → vector mismatch / 매핑 불일치 / analyzer 문제 가능"
        )
    else:
        log.info(f"🔍 ES 직접 검색 결과: {len(hits)}개 (top score={hits[0]['_score']})")

    # 5) 기존 방식 (LangChain)도 시도 → 교차 검증
    try:
        docs = vectorstore.similarity_search(question, k=k)
    except Exception:
        log.error("❌ similarity_search 실패", exc_info=True)
        return ""

    if not docs:
        log.warning("⚠ LangChain similarity_search 결과 0개")
        return ""

    log.info(f"🔍 최종 검색 결과: {len(docs)}개")

    result = []
    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata or {}
        log.info(
            f"[DOC {i}] chunk_id={meta.get('chunk_id')} "
            f"page={meta.get('page')} source={meta.get('source_file')}"
        )

        header = (
            f"[문서 {i}]\n"
            f"chunk_id: {meta.get('chunk_id')}\n"
            f"page: {meta.get('page')}\n"
            f"source_file: {meta.get('source_file')}\n"
            f"section: {meta.get('section')}"
        )
        result.append(f"{header}\n\n{doc.page_content}")

    return "\n\n---\n\n".join(result)


def answer_with_rag(question: str) -> str:
    context = retrieve_context(question, k=5)

    # ❗ context가 비었으면 LLM만 호출
    if not context:
        prompt_template = load_prompt("rag_qa", "1.0.0")
        prompt = prompt_template.format(context="(참고 문서 없음)", question=question)
        reply = llm.invoke(prompt)
        return reply.content

    # 🔥 context에서 문서 메타데이터 추출해 출처 목록 만들기
    sources = []
    for block in context.split("---"):
        lines = block.strip().split("\n")
        meta = {k.split(": ")[0]: k.split(": ")[1] for k in lines[:4] if ": " in k}
        if meta:
            sources.append(meta)

    prompt_template = load_prompt("rag_qa", "1.0.0")
    prompt = prompt_template.format(context=context, question=question)
    reply = llm.invoke(prompt)

    # 🧾 출처 목록 문자열 구성
    source_text = "\n".join(
        f"{i+1}) {src.get('source_file', '?')} — page {src.get('page', '?')} — chunk_id={src.get('chunk_id', '?')}"
        for i, src in enumerate(sources)
    )

    return f"""{reply.content}

────────────────────────────────────────
🔎 참고 문서 출처
{source_text}
"""


# ─────────────────────────────────────────────
# 6) 실행
# ─────────────────────────────────────────────
if __name__ == "__main__":
    question = "2024년 연말정산에서 중소기업 취업자 소득세 감면과 근로소득세액공제는 각각 어떤 기준과 한도에 따라 적용되며, 두 공제를 동시에 받을 수 있을 때 실제 세액에 어떤 방식으로 영향을 미치는지 예시와 함께 설명해 줘."
    print("\n💡 질문:", question)
    print("\n📌 RAG 기반 답변:\n")
    print(answer_with_rag(question))
