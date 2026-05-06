"""PortOne Sales Chatbot — Streamlit UI."""
from __future__ import annotations

import os
import time
from typing import Any
from urllib.parse import urlparse

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

from config import CONTACT_EMAIL, SOURCE_URLS, SUGGESTED_QUESTIONS
from rag_engine import IndexStats, build_chain, build_or_load_vector_store, extract_sources
from styles import CUSTOM_CSS

load_dotenv()


# ---------- Cached resources ---------------------------------------------------

@st.cache_resource(show_spinner=False)
def _bootstrap(force_rebuild_token: int = 0) -> tuple[Any, IndexStats]:
    """Build (or load) the vector store and conversational chain once per process.

    `force_rebuild_token` lets the sidebar invalidate this cache.
    """
    del force_rebuild_token  # used only for cache invalidation
    store, stats = build_or_load_vector_store()
    chain = build_chain(store)
    return chain, stats


# ---------- UI helpers ---------------------------------------------------------

def _inject_styles() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def _render_hero() -> None:
    st.markdown(
        """
        <div class="po-hero">
          <div class="po-logo">P1</div>
          <div>
            <h1>PortOne Sales Assistant</h1>
            <p>포트원 서비스·제품 자료 기반 AI 영업 어시스턴트 · Gemini 2.5 Pro</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    items_html = []
    for s in sources:
        if s["type"] == "web":
            host = urlparse(s["source"]).netloc or s["source"]
            items_html.append(
                f'<span class="po-source-item"><span class="po-source-icon">🌐</span>'
                f'<a href="{s["source"]}" target="_blank" rel="noopener">{host}</a></span>'
            )
        else:
            items_html.append(
                f'<span class="po-source-item"><span class="po-source-icon">📄</span>{s["label"]}</span>'
            )
    st.markdown(
        f'<div class="po-sources">'
        f'<div class="po-source-title">참고 자료</div>'
        f'<div>{" ".join(items_html)}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )


def _render_suggested_questions() -> str | None:
    """Render quick-start chips. Returns the clicked question, or None."""
    st.markdown(
        '<div class="po-empty"><h3>무엇이든 물어보세요</h3>'
        "<p>아래 추천 질문으로 시작하거나, 직접 질문을 입력해 주세요.</p></div>",
        unsafe_allow_html=True,
    )
    cols = st.columns(2)
    for idx, q in enumerate(SUGGESTED_QUESTIONS):
        if cols[idx % 2].button(q, key=f"sugg_{idx}", use_container_width=True):
            return q
    return None


def _render_sidebar(stats: IndexStats | None, error: str | None) -> None:
    with st.sidebar:
        st.markdown("### 시스템 상태")
        if error:
            st.markdown(
                f'<span class="po-badge po-badge--err"><span class="dot"></span>오류</span>',
                unsafe_allow_html=True,
            )
            st.caption(error)
        elif stats:
            st.markdown(
                '<span class="po-badge po-badge--ok"><span class="dot"></span>준비 완료</span>',
                unsafe_allow_html=True,
            )
            st.caption(
                f"PDF {stats.pdf_count}건 · 웹 {stats.web_count}건 · "
                f"청크 {stats.chunk_count:,}개"
            )
            st.caption(f"인덱스 ID: `{stats.fingerprint}`")
        else:
            st.markdown(
                '<span class="po-badge po-badge--warn"><span class="dot"></span>로딩 중</span>',
                unsafe_allow_html=True,
            )

        st.divider()
        st.markdown("### 작업")
        if st.button("🧹 대화 초기화", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
        if st.button("🔄 자료 다시 인덱싱", use_container_width=True):
            st.session_state.rebuild_token = st.session_state.get("rebuild_token", 0) + 1
            _bootstrap.clear()
            st.rerun()

        st.divider()
        st.markdown("### 데이터 소스")
        st.caption("PDF: 프로젝트 루트의 모든 *.pdf")
        with st.expander("연결된 웹사이트", expanded=False):
            for u in SOURCE_URLS:
                st.markdown(f"- [{urlparse(u).netloc}]({u})")

        st.divider()
        st.markdown("### 문의")
        st.caption(f"도입/기술 논의: **{CONTACT_EMAIL}**")


def _replay_history() -> None:
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(message.content)
                _render_sources(message.additional_kwargs.get("sources", []))


def _check_api_key() -> str | None:
    if not os.environ.get("GOOGLE_API_KEY"):
        return (
            "GOOGLE_API_KEY 환경 변수가 설정되지 않았습니다. "
            "프로젝트 루트의 `.env` 파일에 `GOOGLE_API_KEY=...` 를 추가하고 페이지를 새로고침해 주세요."
        )
    return None


def _answer(chain: Any, prompt: str) -> None:
    """Run the chain, render assistant message, append to history."""
    st.session_state.chat_history.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        placeholder = st.empty()
        sources_slot = st.empty()
        with st.spinner("관련 자료를 찾아 답변을 작성 중입니다…"):
            t0 = time.time()
            response = chain.invoke(
                {
                    "input": prompt,
                    "chat_history": st.session_state.chat_history[:-1],
                }
            )
            elapsed = time.time() - t0

        answer = response.get("answer", "").strip()
        sources = extract_sources(response.get("context", []))
        placeholder.markdown(answer)
        with sources_slot.container():
            _render_sources(sources)
            st.caption(f"⏱ 응답 시간 {elapsed:.1f}s")

    st.session_state.chat_history.append(
        AIMessage(content=answer, additional_kwargs={"sources": sources})
    )


# ---------- Main ---------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="PortOne Sales Assistant",
        page_icon="💬",
        layout="centered",
        initial_sidebar_state="expanded",
    )
    _inject_styles()
    _render_hero()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "rebuild_token" not in st.session_state:
        st.session_state.rebuild_token = 0

    api_error = _check_api_key()
    chain: Any | None = None
    stats: IndexStats | None = None
    runtime_error: str | None = api_error

    if not api_error:
        try:
            with st.spinner("자료를 불러오고 인덱싱 중입니다… (최초 1회)"):
                chain, stats = _bootstrap(st.session_state.rebuild_token)
        except Exception as exc:
            runtime_error = f"초기화 실패: {exc}"

    _render_sidebar(stats, runtime_error)

    if runtime_error:
        st.error(runtime_error)
        return

    if not st.session_state.chat_history:
        clicked = _render_suggested_questions()
        if clicked:
            _answer(chain, clicked)
            st.rerun()
    else:
        _replay_history()

    if prompt := st.chat_input("포트원 서비스에 대해 질문해주세요…"):
        _answer(chain, prompt)
        st.rerun()


if __name__ == "__main__":
    main()
