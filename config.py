"""Application configuration and constants."""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parent
PDF_DIR: Path = PROJECT_ROOT
VECTOR_STORE_DIR: Path = PROJECT_ROOT / "data" / "vector_store"
VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

LLM_MODEL: str = "models/gemini-2.5-pro"
EMBEDDING_MODEL: str = "models/embedding-001"
LLM_TEMPERATURE: float = 0.3

CHUNK_SIZE: int = 1000
CHUNK_OVERLAP: int = 200
RETRIEVAL_K: int = 5

REQUEST_TIMEOUT_SEC: int = 15

SOURCE_URLS: list[str] = [
    "https://portone.io/korea/ko",
    "https://help.portone.io/",
    "https://developers.portone.io/opi/ko/readme?v=v2",
    "https://blog.portone.io/",
]

CONTACT_EMAIL: str = "ocean@portone.io"

SYSTEM_PROMPT: str = """당신은 포트원(PortOne)의 전문 영업 어시스턴트 'PortOne Sales Bot'입니다.
주어진 컨텍스트(Context)를 근거로 사용자의 질문에 친절하고 명확하게, 한국어로 답변해주세요.

답변 규칙:
1. 컨텍스트에서 근거를 찾을 수 있는 경우, 핵심 결론을 먼저 제시한 뒤 구조화(불릿/번호)된 설명을 덧붙입니다.
2. 가능한 경우 구체적인 기능/숫자/사례를 인용하고, 기술 용어는 짧게 풀어 설명합니다.
3. 컨텍스트에서 답을 찾을 수 없는 경우에는 다음과 같이 안내합니다:
   "죄송합니다. 문의하신 내용은 제공된 자료만으로는 정확히 답변드리기 어렵습니다.
   더 자세한 내용이 궁금하시거나 도입 논의가 필요하시면 {contact_email} 로 메일 주시면
   미팅을 통해 상세히 안내해 드리겠습니다."
4. 절대 컨텍스트 외의 사실을 지어내지 마세요. 추측 시에는 '추정' 임을 명시합니다.

Context:
{context}
"""

CONTEXTUALIZE_Q_PROMPT: str = (
    "이전 대화 내용과 마지막 사용자 질문을 참고하여, 대화 맥락 없이도 이해 가능한 "
    "독립된 질문으로 재작성하세요. 답변하지는 말고, 필요할 때만 재작성하고 그렇지 않으면 그대로 반환하세요."
)

SUGGESTED_QUESTIONS: list[str] = [
    "포트원의 핵심 서비스를 한 줄로 요약해줘",
    "원 페이먼트 인프라(One Payment Infra)는 어떤 문제를 해결하나요?",
    "재무 자동화 솔루션은 어떤 기능을 제공하나요?",
    "K-Brands(글로벌 결제)는 어떤 PG/결제수단을 지원하나요?",
    "도입 절차와 예상 일정이 어떻게 되나요?",
    "경쟁 솔루션 대비 포트원의 차별점은?",
]
