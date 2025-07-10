# 포트원(PortOne) 세일즈 챗봇

이 프로젝트는 포트원 서비스에 대한 고객의 질문에 답변하는 AI 챗봇입니다.

## 주요 기능

- **다중 소스 기반 답변:** PDF 문서와 공식 웹사이트(소개, 기술문서, 블로그 등)의 정보를 종합하여 답변을 생성합니다.
- **출처 표시:** 답변의 근거가 된 웹사이트 URL을 함께 제공하여 신뢰도를 높입니다.
- **Gemini 모델 활용:** Google의 최신 언어 모델인 Gemini를 기반으로 자연스러운 대화를 제공합니다.
- **Streamlit 기반 UI:** 사용자가 쉽게 상호작용할 수 있는 웹 기반 인터페이스를 제공합니다.

## 설치 및 실행 방법

1.  **리포지토리 클론:**
    ```bash
    git clone https://github.com/ohjunho421/portone_saleschatbot.git
    cd portone_saleschatbot
    ```

2.  **가상 환경 생성 및 활성화:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # macOS/Linux
    .\venv\Scripts\activate  # Windows
    ```

3.  **필요 패키지 설치:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **.env 파일 생성:**
    -   `.env` 파일을 생성하고 Google API 키를 다음과 같이 추가합니다.
        ```
        GOOGLE_API_KEY="YOUR_API_KEY_HERE"
        ```

5.  **애플리케이션 실행:**
    ```bash
    streamlit run app.py
    ```
