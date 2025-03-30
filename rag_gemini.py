import os
from dotenv import load_dotenv
from llama_index.core import (
    StorageContext,
    load_index_from_storage,
    Settings
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import google.generativeai as genai

# ✅ 환경 변수 로드
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

# ✅ Gemini 설정
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("models/gemini-1.5-flash-001")

# ✅ HuggingFace 임베딩 모델 로드
embed_model = HuggingFaceEmbedding(
    model_name="nlpai-lab/KURE-v1",
    device="cpu",
    token=HF_TOKEN
)
Settings.embed_model = embed_model

# ✅ FAISS 벡터 DB 로드
storage_context = StorageContext.from_defaults(persist_dir="./faiss_conversation_db_expanded")
index = load_index_from_storage(storage_context)
retriever = index.as_retriever(similarity_top_k=3)
retriever.embed_model = embed_model

# ✅ 질문 → 검색 → 응답

def query_rag(question):
    nodes = retriever.retrieve(question)
    knowledge = "\n".join(node.text for node in nodes)

    print("\n✅ 검색된 문서:\n", knowledge)

    prompt = f"""다음 대화를 참고하여 질문에 간단히 답해주세요. 모르면 '정보 없음'이라고 답하세요.

대화: {knowledge}

질문: {question}
답변:"""

    response = model.generate_content(prompt)
    return response.text.strip()

# ✅ 테스트
if __name__ == "__main__":
    questions = [
        "사용자가 무서워하는 동물은 무엇인가요?",
        "사용자가 좋아하는 영화 장르는 무엇인가요?",
        "사용자가 싫어하는 음식은 무엇인가요?",
        "사용자가 두려워하는 것은 무엇인가요?",
        "사용자가 매운 음식을 잘 먹나요?"
    ]

    for question in questions:
        answer = query_rag(question)
        print(f"\n❓질문: {question}\n👉답변: {answer}")
