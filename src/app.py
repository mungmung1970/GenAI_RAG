import streamlit as st
from rag_chain_old import answer_with_rag  # 기존 RAG 엔진 그대로 사용

st.set_page_config(page_title="RAG Q&A", layout="wide")

st.title("📌 연말정산 RAG 상담 서비스")
st.write("문서를 기반으로 정확한 답변을 제공합니다. 질문을 입력하세요.")

if "history" not in st.session_state:
    st.session_state.history = []  # 대화 기록 저장

question = st.text_input(
    "질문 입력",
    placeholder="예: 중소기업 취업자 소득세 감면 요건과 월세 세액공제 병행 여부 알려줘",
)

if st.button("질문하기") and question.strip():
    with st.spinner("검색 중… 문서를 분석하고 있습니다."):
        answer = answer_with_rag(question)

    st.session_state.history.append((question, answer))

# ───────── 대화 기록 표시 ─────────
for q, a in reversed(st.session_state.history):
    st.markdown(f"### ❓ {q}")
    st.markdown(f"🧠 **답변:**\n{a}")
    st.markdown("---")
