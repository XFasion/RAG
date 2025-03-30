# RAG 기반 캐릭터 대화 응답 시스템

한국어 캐릭터 간 대화 데이터를 벡터로 임베딩하고, 유사 문장을 검색해 OpenAI GPT 모델로 질문에 응답하는 시스템

---

## ✅ 주요 기능

- HuggingFace 임베딩 모델 `nlpai-lab/KURE-v1` 사용
- 캐릭터 대화 데이터 벡터화 → FAISS Vector DB 저장
- 유사 대화 검색 → GPT-4o 기반 응답 생성
- 실시간 문장 추가 및 벡터 업데이트 지원

---

## ⚙️ 실행 순서

1. **예시 대화 CSV 생성 (선택)**  
   → `dialogue_make.py` 실행 → `conversation_dataset_expanded.csv` 생성

2. **FAISS 벡터 DB 생성**  
   → `append_to_faiss.py` 실행 → 대화 데이터를 벡터화하여 `faiss_conversation_db_expanded/` 디렉토리에 저장

3. **질문 입력 후 응답 생성**  
   → `rag_openai.py` 실행 → 유사 대화 검색 후 GPT 모델을 통해 답변 생성

4. **실시간 대화 추가 반영 (선택)**  
   → `append_to_faiss.py` 내 `add_message_to_index(speaker, dialogue)` 함수 호출

5. **디버깅/확인 도구**  
   - `embedding_model_verify.py`: 임베딩 모델 차원 수 확인  
   - `vectordb_count.py`: 벡터 DB 내 문장 수 확인

---

## 📂 주요 파일 설명

| 파일명                      | 설명 |
|----------------------------|------|
| `dialogue_make.py`         | 예시 대화 CSV 생성 스크립트 |
| `append_to_faiss.py`       | 대화 데이터를 벡터화 및 FAISS DB 저장 |
| `rag_openai.py`            | 질문을 받아 RAG 방식으로 응답 생성 |
| `embedding_model_verify.py`| 임베딩 차원 수 확인용 |
| `vectordb_count.py`        | 벡터 DB 내 문장 수 확인 |

---

## 🧪 환경 설정

1. `.env` 파일 생성
OPENAI_API_KEY=your_openai_api_key
HF_TOKEN=your_huggingface_token
2. 패키지 설치

```bash
pip install -r requirements.txt
