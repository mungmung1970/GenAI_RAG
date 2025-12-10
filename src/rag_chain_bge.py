# rag_chain_bge.py (LCEL + Hybrid + 개선된 Reranker, bge-m3 적용)

import os
import yaml
import json
from pathlib import Path
from dotenv import load_dotenv
import logging

from elasticsearch import Elasticsearch
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_elasticsearch import ElasticsearchStore

from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)

# ─────────────────────────────────────────────
# 로깅
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("rag")

# ─────────────────────────────────────────────
# ENV + LangSmith
# ─────────────────────────────────────────────
load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGSMITH_TRACING", "true")
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "rag_bge_lcel")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY 없음")

# ─────────────────────────────────────────────
# Elasticsearch index 정보
# ─────────────────────────────────────────────
ES_URL = "https://localhost:9200"
ES_PASSWORD = "elastic"
ES_INDEX = "rag_chunks_bge"  # ES에 bge-m3로 색인된 index

DENSE_WEIGHT = 0.7
LEX_WEIGHT = 0.3
TOP_N_AFTER_RERANK = 5


# ─────────────────────────────────────────────
# prompts.yaml
# ─────────────────────────────────────────────
def load_prompt(name: str, version=None):
    base_dir = Path(__file__).resolve().parent.parent
    path = base_dir / "prompt" / "prompts.yaml"
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    for item in data["prompts"]:
        if item["name"] == name and (version is None or item["version"] == version):
            return item["template"]
    raise ValueError("Prompt not found")


prompt_template = load_prompt("rag_qa", "1.0.0")


# ─────────────────────────────────────────────
# bge-m3 Embedding (1024-dim)
# ─────────────────────────────────────────────
emb = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},  # cosine similarity
)

# ─────────────────────────────────────────────
# Elasticsearch Store
# ─────────────────────────────────────────────
es = Elasticsearch(ES_URL, basic_auth=("elastic", ES_PASSWORD), verify_certs=False)

vectorstore = ElasticsearchStore(
    es_connection=es, index_name=ES_INDEX, embedding=emb, vector_query_field="embedding"
)


# ─────────────────────────────────────────────
# LLM (reranker + 최종 응답 모두)
# ─────────────────────────────────────────────
llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, temperature=0.2)


# ─────────────────────────────────────────────
# Hybrid Search + Reranker
# ─────────────────────────────────────────────
def _normalize(scores):
    mn, mx = min(scores), max(scores)
    if abs(mx - mn) < 1e-9:
        return [1] * len(scores)
    return [(s - mn) / (mx - mn) for s in scores]


def hybrid_retrieve_with_rerank(question: str, k: int = 5):
    log.info(f"[RETRIEVER] 하이브리드 검색: {question}")

    # dense (bge-m3)
    qvec = emb.embed_query(question)

    dense_hits = es.knn_search(
        index=ES_INDEX,
        knn={"field": "embedding", "query_vector": qvec, "k": 10, "num_candidates": 60},
        source=["text", "metadata"],
    )["hits"]["hits"]

    # lexical (BM25)
    lex_hits = es.search(
        index=ES_INDEX,
        body={
            "size": 10,
            "_source": ["text", "metadata"],
            "query": {
                "multi_match": {
                    "query": question,
                    "fields": ["text^2", "metadata.section"],
                }
            },
        },
    )["hits"]["hits"]

    log.info(f"🔹 dense={len(dense_hits)}, bm25={len(lex_hits)}")

    dense_norm = _normalize([h["_score"] for h in dense_hits]) if dense_hits else []
    lex_norm = _normalize([h["_score"] for h in lex_hits]) if lex_hits else []

    # merge
    cands = {}

    for hit, s in zip(dense_hits, dense_norm):
        cid = hit["_source"]["metadata"]["chunk_id"]
        cands[cid] = {
            "chunk_id": cid,
            "text": hit["_source"]["text"],
            "meta": hit["_source"]["metadata"],
            "dense": s,
            "lex": 0.0,
        }

    for hit, s in zip(lex_hits, lex_norm):
        cid = hit["_source"]["metadata"]["chunk_id"]
        if cid not in cands:
            cands[cid] = {
                "chunk_id": cid,
                "text": hit["_source"]["text"],
                "meta": hit["_source"]["metadata"],
                "dense": 0.0,
                "lex": s,
            }
        else:
            cands[cid]["lex"] = s

    # hybrid score
    for c in cands.values():
        c["hybrid"] = DENSE_WEIGHT * c["dense"] + LEX_WEIGHT * c["lex"]

    merged = sorted(cands.values(), key=lambda x: x["hybrid"], reverse=True)

    # reranker pool
    rerank_pool = merged[: 2 * k]

    # reranker prompt (OpenAI version과 완전히 동일)
    rerank_prompt = (
        "검색 결과 reranker입니다.\n"
        "각 문서를 0~5 점으로 평가하되, 점수가 넓게 분포하도록 하세요.\n"
        "JSON 리스트만 출력하세요.\n\n"
        f"질문: {question}\n\n후보 문서:\n"
    )

    for c in rerank_pool:
        m = c["meta"]
        rerank_prompt += (
            f"chunk_id={c['chunk_id']} | page={m.get('page')} | section={m.get('section')}\n"
            f"내용:\n{c['text']}\n\n"
        )

    rerank_scores = {}
    try:
        resp = llm.invoke(rerank_prompt)
        txt = resp.content.strip().replace("```json", "").replace("```", "")
        for item in json.loads(txt):
            rerank_scores[item["chunk_id"]] = float(item["score"])
    except Exception:
        log.warning("⚠ rerank 실패 → hybrid만 사용")

    # rerank 적용 정렬
    def rank_key(c):
        if c["chunk_id"] in rerank_scores:
            return 1, rerank_scores[c["chunk_id"]]
        return 0, c["hybrid"]

    final = sorted(rerank_pool, key=rank_key, reverse=True)[:TOP_N_AFTER_RERANK]

    log.info(f"✅ 최종 선택 문서 = {len(final)}개")

    # context 구성 (원본 rag_chain.py 포맷과 100% 동일)
    ctxs = []
    for c in final:
        m = c["meta"]
        header = (
            f"chunk_id={c['chunk_id']} | page={m.get('page')} | "
            f"source={m.get('source_file')} | section={m.get('section')}"
        )
        ctxs.append(f"{header}\n{c['text']}")

    return "\n\n---\n\n".join(ctxs)


# Runnable
retriever = RunnableLambda(hybrid_retrieve_with_rerank)


# ─────────────────────────────────────────────
# Prompt + Postprocess
# ─────────────────────────────────────────────
def build_prompt(inputs):
    return prompt_template.format(
        context=inputs["context"], question=inputs["question"]
    )


prompt_builder = RunnableLambda(build_prompt)


def extract_sources(answer: str, ctx: str):
    sources = []
    for block in ctx.split("---"):
        for line in block.split("\n"):
            line = line.strip()
            if line.startswith("chunk_id="):
                sources.append(line)
    sources = list(dict.fromkeys(sources))  # dedupe
    return (
        answer
        + "\n\n──────────────────────────────\n📎 참고 문서\n"
        + "\n".join(sources)
    )


postprocess = RunnableLambda(lambda x: extract_sources(x["answer"], x["context"]))


# ─────────────────────────────────────────────
# LCEL 전체 체인
# ─────────────────────────────────────────────
rag_chain = (
    RunnableParallel(question=RunnablePassthrough(), context=retriever)
    | RunnableParallel(prompt=prompt_builder, context=lambda x: x["context"])
    | (lambda x: {"answer": llm.invoke(x["prompt"]).content, "context": x["context"]})
    | postprocess
)


def answer_with_rag(question: str):
    return rag_chain.invoke(question)


if __name__ == "__main__":
    q = "2024년 연말정산 중 자녀 교육비 공제 기준과 한도는?"
    print(answer_with_rag(q))
