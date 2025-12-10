# rag_chain_ollama_optionB.py
# (OpenAI Embedding + OpenAI Reranker + Ollama Final Answer + Context Length Control)

import os
import yaml
import json
import requests
import logging
from pathlib import Path
from dotenv import load_dotenv

from elasticsearch import Elasticsearch
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

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
# 환경 변수 및 LangSmith 설정
# ─────────────────────────────────────────────
load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGSMITH_TRACING", "true")
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv(
    "LANGSMITH_PROJECT", "rag_lcel_ollama_optionB"
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY 없음")

# ─────────────────────────────────────────────
# Elasticsearch 설정
# ─────────────────────────────────────────────
ES_URL = "https://localhost:9200"
ES_PASSWORD = "elastic"
ES_INDEX = "rag_chunks"

DENSE_WEIGHT = 0.7
LEX_WEIGHT = 0.3
TOP_N_AFTER_RERANK = 5

# Context 길이 제한
MAX_CONTEXT_CHARS = 8000  # ★ 원하는 길이로 조절 가능 (chars 기준)

# ─────────────────────────────────────────────
# Embedding
# ─────────────────────────────────────────────
emb = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)

# ─────────────────────────────────────────────
# OpenAI Reranker LLM
# ─────────────────────────────────────────────
reranker_llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, temperature=0)


# ─────────────────────────────────────────────
# Ollama LLM 호출 함수
# ─────────────────────────────────────────────
def ollama_chat(prompt: str, model="llama3:8b-instruct-q4_0"):
    """Ollama 모델 호출"""
    r = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=600,  # ★ 긴 context에서도 timeout 방지
    )
    r.raise_for_status()
    return r.json()["response"]


# ─────────────────────────────────────────────
# Prompt Loader
# ─────────────────────────────────────────────
def load_prompt(name: str, version=None, model=None):
    base_dir = Path(__file__).resolve().parent.parent
    path = base_dir / "prompt" / "prompts.yaml"

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    candidates = []
    for item in data.get("prompts", []):
        if item.get("name") != name:
            continue
        if version is not None and item.get("version") != version:
            continue
        if model is not None and item.get("model") != model:
            continue
        candidates.append(item)

    if not candidates:
        raise ValueError(f"Prompt not found: {name}, {version}, {model}")

    return candidates[0]["template"]


prompt_template = load_prompt("rag_qa", "1.0.0")

# ─────────────────────────────────────────────
# Elasticsearch Connect
# ─────────────────────────────────────────────
es = Elasticsearch(ES_URL, basic_auth=("elastic", ES_PASSWORD), verify_certs=False)


# ─────────────────────────────────────────────
# Hybrid Retrieval + OpenAI Reranker
# ─────────────────────────────────────────────
def _normalize(scores):
    mn, mx = min(scores), max(scores)
    if abs(mx - mn) < 1e-9:
        return [1] * len(scores)
    return [(s - mn) / (mx - mn) for s in scores]


def hybrid_retrieve_with_rerank(question: str, k: int = 5):
    log.info(f"[RETRIEVER] 하이브리드 검색: {question}")

    qvec = emb.embed_query(question)

    # Dense Search
    dense_hits = es.knn_search(
        index=ES_INDEX,
        knn={"field": "embedding", "query_vector": qvec, "k": 10, "num_candidates": 60},
        source=["text", "metadata"],
    )["hits"]["hits"]

    # BM25 Search
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

    # Normalize scores
    dense_norm = _normalize([h["_score"] for h in dense_hits]) if dense_hits else []
    lex_norm = _normalize([h["_score"] for h in lex_hits]) if lex_hits else []

    cands = {}

    for hit, s in zip(dense_hits, dense_norm):
        cid = hit["_source"]["metadata"]["chunk_id"]
        cands[cid] = {
            "chunk_id": cid,
            "text": hit["_source"]["text"],
            "meta": hit["_source"]["metadata"],
            "dense": s,
            "lex": 0,
        }

    for hit, s in zip(lex_hits, lex_norm):
        cid = hit["_source"]["metadata"]["chunk_id"]
        if cid not in cands:
            cands[cid] = {
                "chunk_id": cid,
                "text": hit["_source"]["text"],
                "meta": hit["_source"]["metadata"],
                "dense": 0,
                "lex": s,
            }
        else:
            cands[cid]["lex"] = s

    # Hybrid Score
    for c in cands.values():
        c["hybrid"] = DENSE_WEIGHT * c["dense"] + LEX_WEIGHT * c["lex"]

    pool = sorted(cands.values(), key=lambda x: x["hybrid"], reverse=True)[: 2 * k]

    # 🔥 OpenAI Reranker Prompt
    rerank_prompt = (
        "당신은 RAG 검색 결과 Reranker입니다.\n"
        "질문과 가장 관련 있는 문서를 0~5점으로 평가하세요.\n"
        "무조건 JSON 리스트만 출력하세요.\n\n"
        f"질문: {question}\n\n후보 문서:\n"
    )

    for c in pool:
        m = c["meta"]
        rerank_prompt += (
            f"chunk_id={c['chunk_id']} | page={m.get('page')}\n{c['text']}\n\n"
        )

    rerank_scores = {}
    try:
        resp = reranker_llm.invoke(rerank_prompt)
        txt = resp.content.strip().replace("```json", "").replace("```", "")
        for item in json.loads(txt):
            rerank_scores[item["chunk_id"]] = float(item["score"])
    except Exception as e:
        log.warning(f"⚠ Reranker 실패 → Hybrid-only 사용 ({e})")

    def rank_key(c):
        return (
            (1, rerank_scores[c["chunk_id"]])
            if c["chunk_id"] in rerank_scores
            else (0, c["hybrid"])
        )

    final = sorted(pool, key=rank_key, reverse=True)[:TOP_N_AFTER_RERANK]

    # Context building
    ctxs = []
    for c in final:
        m = c["meta"]
        ctxs.append(
            f"chunk_id={c['chunk_id']} | page={m.get('page')} | source={m.get('source_file')}\n{c['text']}"
        )

    context_raw = "\n\n---\n\n".join(ctxs)

    # ★ Context Length Control
    if len(context_raw) > MAX_CONTEXT_CHARS:
        context_raw = context_raw[:MAX_CONTEXT_CHARS] + "\n\n...(이하 생략)..."

    return context_raw


retriever = RunnableLambda(hybrid_retrieve_with_rerank)


# ─────────────────────────────────────────────
# Prompt Builder
# ─────────────────────────────────────────────
def build_prompt(inputs):
    return prompt_template.format(
        context=inputs["context"], question=inputs["question"]
    )


prompt_builder = RunnableLambda(build_prompt)


# ─────────────────────────────────────────────
# Postprocess
# ─────────────────────────────────────────────
def extract_sources(answer: str, ctx: str):
    sources = []
    for block in ctx.split("---"):
        for line in block.split("\n"):
            if line.strip().startswith("chunk_id="):
                sources.append(line.strip())
    sources = list(dict.fromkeys(sources))
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
    RunnableParallel(
        question=RunnablePassthrough(),
        context=retriever,
    )
    | RunnableParallel(
        prompt=prompt_builder,
        context=lambda x: x["context"],
    )
    | (lambda x: {"answer": ollama_chat(x["prompt"]), "context": x["context"]})
    | postprocess
)


def answer_with_rag(question: str):
    return rag_chain.invoke(question)


# ─────────────────────────────────────────────
# Test
# ─────────────────────────────────────────────
if __name__ == "__main__":
    q = "2024년 연말정산 중 자녀 교육비 공제 기준과 한도는?"
    print(answer_with_rag(q))
