import os 
from llama_index.core import (
    StorageContext,
    load_index_from_storage,
    Settings
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from openai import OpenAI
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 임베딩 모델 설정
embed_model = HuggingFaceEmbedding(
    model_name="nlpai-lab/KURE-v1",
    device="cpu",
    token=HF_TOKEN
)

Settings.embed_model = embed_model

# FAISS 인덱스 로드
storage_context = StorageContext.from_defaults(persist_dir="./faiss_conversation_db_expanded")
index = load_index_from_storage(storage_context)

# Retriever 설정
retriever = index.as_retriever(similarity_top_k=3)
retriever.embed_model = embed_model

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=OPENAI_API_KEY)

# 질문에 답하는 함수 정의
def query_rag(question):
    nodes = retriever.retrieve(question)
    knowledge = "\n".join(node.text for node in nodes)

    print("\n✅ 검색된 문서:\n", knowledge)

    prompt = f"""
    다음 대화를 참고하여 질문에 간단히 답해주세요. 모르면 '정보 없음'이라고 답하세요.

    대화: {knowledge}

    질문: {question}
    답변:
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0
    )

    return response.choices[0].message.content.strip()

# 질문 테스트
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
