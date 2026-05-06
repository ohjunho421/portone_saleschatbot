"""RAG pipeline: data loading, vector store persistence, conversational chain."""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import requests
from bs4 import BeautifulSoup
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CONTACT_EMAIL,
    CONTEXTUALIZE_Q_PROMPT,
    EMBEDDING_MODEL,
    LLM_MODEL,
    LLM_TEMPERATURE,
    PDF_DIR,
    REQUEST_TIMEOUT_SEC,
    RETRIEVAL_K,
    SOURCE_URLS,
    SYSTEM_PROMPT,
    VECTOR_STORE_DIR,
)

logger = logging.getLogger(__name__)


@dataclass
class IndexStats:
    pdf_count: int
    web_count: int
    chunk_count: int
    fingerprint: str


def _fingerprint(pdf_paths: list[Path], urls: list[str]) -> str:
    """Stable hash of input data so we can invalidate cache when sources change."""
    h = hashlib.sha256()
    for p in sorted(pdf_paths):
        try:
            stat = p.stat()
            h.update(p.name.encode("utf-8"))
            h.update(str(stat.st_size).encode("utf-8"))
            h.update(str(int(stat.st_mtime)).encode("utf-8"))
        except OSError:
            continue
    for u in sorted(urls):
        h.update(u.encode("utf-8"))
    return h.hexdigest()[:16]


def discover_pdfs(directory: Path = PDF_DIR) -> list[Path]:
    return sorted(p for p in directory.glob("*.pdf") if p.is_file())


def load_pdfs(pdf_paths: Iterable[Path]) -> list[Document]:
    docs: list[Document] = []
    for path in pdf_paths:
        try:
            loaded = PyMuPDFLoader(file_path=str(path)).load()
            for d in loaded:
                d.metadata.setdefault("source", path.name)
                d.metadata.setdefault("source_type", "pdf")
            docs.extend(loaded)
        except Exception as exc:
            logger.warning("Failed to load PDF %s: %s", path, exc)
    return docs


def load_websites(urls: Iterable[str]) -> list[Document]:
    docs: list[Document] = []
    for url in urls:
        try:
            resp = requests.get(
                url,
                timeout=REQUEST_TIMEOUT_SEC,
                headers={"User-Agent": "PortOneSalesBot/1.0"},
            )
            resp.raise_for_status()
            soup = BeautifulSoup(resp.content, "html.parser")
            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()
            text = soup.get_text(separator="\n")
            cleaned = "\n".join(line.strip() for line in text.splitlines() if line.strip())
            if cleaned:
                docs.append(
                    Document(
                        page_content=cleaned,
                        metadata={"source": url, "source_type": "web"},
                    )
                )
        except Exception as exc:
            logger.warning("Failed to fetch %s: %s", url, exc)
    return docs


def split_documents(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    return splitter.split_documents(docs)


def _index_path(fingerprint: str) -> Path:
    return VECTOR_STORE_DIR / fingerprint


def build_or_load_vector_store(force_rebuild: bool = False) -> tuple[FAISS, IndexStats]:
    """Build a FAISS index (or load it from disk if the fingerprint matches)."""
    pdfs = discover_pdfs()
    fingerprint = _fingerprint(pdfs, SOURCE_URLS)
    target = _index_path(fingerprint)

    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

    if not force_rebuild and (target / "index.faiss").exists():
        try:
            store = FAISS.load_local(
                str(target),
                embeddings,
                allow_dangerous_deserialization=True,
            )
            stats = IndexStats(
                pdf_count=len(pdfs),
                web_count=len(SOURCE_URLS),
                chunk_count=store.index.ntotal,
                fingerprint=fingerprint,
            )
            logger.info("Loaded cached vector store (%s)", fingerprint)
            return store, stats
        except Exception as exc:
            logger.warning("Cached store load failed, rebuilding: %s", exc)

    pdf_docs = load_pdfs(pdfs)
    web_docs = load_websites(SOURCE_URLS)
    all_docs = pdf_docs + web_docs
    if not all_docs:
        raise RuntimeError(
            "로드 가능한 문서가 없습니다. PDF가 프로젝트 루트에 있는지, 네트워크가 연결되어 있는지 확인하세요."
        )

    splits = split_documents(all_docs)
    store = FAISS.from_documents(documents=splits, embedding=embeddings)
    target.mkdir(parents=True, exist_ok=True)
    store.save_local(str(target))

    stats = IndexStats(
        pdf_count=len(pdfs),
        web_count=len(SOURCE_URLS),
        chunk_count=len(splits),
        fingerprint=fingerprint,
    )
    return store, stats


def build_chain(vector_store: FAISS):
    """Conversational RAG chain with history-aware retrieval."""
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE)

    contextualize_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", CONTEXTUALIZE_Q_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm,
        vector_store.as_retriever(search_kwargs={"k": RETRIEVAL_K}),
        contextualize_prompt,
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT.replace("{contact_email}", CONTACT_EMAIL)),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(history_aware_retriever, qa_chain)


def extract_sources(context_docs: list[Document]) -> list[dict]:
    """Deduplicated list of {source, type, label} dicts for UI rendering."""
    seen: set[str] = set()
    out: list[dict] = []
    for d in context_docs:
        src = d.metadata.get("source") or ""
        if not src or src in seen:
            continue
        seen.add(src)
        kind = d.metadata.get("source_type", "web" if src.startswith("http") else "pdf")
        label = src if kind == "pdf" else src
        out.append({"source": src, "type": kind, "label": label})
    return out
