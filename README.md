# 포트원(PortOne) Sales Assistant

PortOne 제품/서비스 자료(PDF · 공식 웹사이트)를 기반으로 한 RAG 기반 영업·문의 응대 챗봇입니다.
Streamlit · LangChain · Google Gemini 2.5 Pro · FAISS 로 구성됩니다.

## 주요 특징

- **자료 기반 RAG 답변** — 프로젝트 루트의 PortOne PDF + 공식 사이트(소개·도움말·개발자 문서·블로그)를 통합 인덱싱하여 답변합니다.
- **출처 표시** — 답변 하단에 참고한 PDF 파일과 웹페이지 도메인을 카드 형태로 함께 표기합니다.
- **대화 맥락 인식** — `create_history_aware_retriever` 로 이전 대화를 참고해 후속 질문도 자연스럽게 처리합니다.
- **빠른 응답** — FAISS 인덱스를 디스크에 영속화하고 `@st.cache_resource` 로 캐시하여 첫 1회 후에는 즉시 응답합니다.
- **데이터 변경 자동 감지** — PDF 추가/수정 시 파일 핑거프린트 기반으로 인덱스를 자동 재빌드합니다.
- **PortOne 브랜드 UI** — 커스텀 테마, 추천 질문 칩, 사이드바 컨트롤, 모바일 친화 레이아웃.

## 프로젝트 구조

```
portone_saleschatbot/
├── app.py              # Streamlit UI (히어로, 채팅, 사이드바)
├── rag_engine.py       # 데이터 로딩 · 벡터스토어 영속화 · 체인 구성
├── config.py           # 모델 · 청크 · 프롬프트 · URL 등 설정
├── styles.py           # PortOne 브랜드 CSS
├── .streamlit/
│   └── config.toml     # Streamlit 테마
├── data/
│   └── vector_store/   # FAISS 인덱스 캐시 (자동 생성)
├── *.pdf               # 인덱싱 대상 자료
├── requirements.txt
└── README.md
```

## 설치 및 실행

1. 리포지토리 클론
   ```bash
   git clone https://github.com/ohjunho421/portone_saleschatbot.git
   cd portone_saleschatbot
   ```

2. 가상환경 생성 및 활성화
   ```bash
   python -m venv venv
   # macOS / Linux
   source venv/bin/activate
   # Windows
   .\venv\Scripts\activate
   ```

3. 패키지 설치
   ```bash
   pip install -r requirements.txt
   ```

4. `.env` 파일 생성
   ```env
   GOOGLE_API_KEY=YOUR_API_KEY_HERE
   ```

5. 실행
   ```bash
   streamlit run app.py
   ```

## 운영 팁

- **자료 추가**: 프로젝트 루트에 PDF 를 추가하면 다음 실행 시 자동 재인덱싱됩니다. 수동 강제 재인덱싱은 사이드바의 **"🔄 자료 다시 인덱싱"** 버튼을 사용하세요.
- **URL 추가/변경**: `config.py` 의 `SOURCE_URLS` 를 수정하세요.
- **프롬프트 튜닝**: `config.py` 의 `SYSTEM_PROMPT` 에서 답변 톤·정책을 변경할 수 있습니다.
- **모델 변경**: `config.py` 의 `LLM_MODEL` / `EMBEDDING_MODEL` 을 교체하세요.

## 보안

- `.env` 는 절대 커밋하지 마세요(`.gitignore` 에 등록되어 있습니다).
- FAISS 인덱스는 로컬에서만 사용한다는 전제로 `allow_dangerous_deserialization=True` 가 활성화되어 있습니다.
  외부에서 받은 인덱스 파일은 사용하지 마세요.

## 문의

도입 및 기술 논의: **ocean@portone.io**
