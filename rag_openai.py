import os
import time
from llama_index.core import (
    StorageContext,
    load_index_from_storage,
    Settings
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from openai import OpenAI
from dotenv import load_dotenv

from rag_prompt import RAG_PROMPT
from yusiyeon import CHARACTER_PROMPT 
os.environ["TOKENIZERS_PARALLELISM"] = "false" # 병렬처리 비활성화

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
    start_retrieve = time.time()
    nodes = retriever.retrieve(question)
    retrieve_time = time.time() - start_retrieve
 
    knowledge = "\n".join(node.text for node in nodes)

    print("\n✅ 검색된 문서:\n", knowledge)
    print(f"\n⏱️ 검색(retrieve) 소요시간: {retrieve_time:.4f} 초")

    prompt = RAG_PROMPT.format(
        character_prompt=CHARACTER_PROMPT,
        related_conversations=knowledge,
        user_question=question
    )
    start_llm = time.time()

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=800,
        temperature=0.8
    )
    llm_time = time.time() - start_llm
    print(f"⏱️ LLM 응답 생성 소요시간: {llm_time:.4f} 초")

    return response.choices[0].message.content.strip()

# 질문 테스트
questions = [
    "너 동물 무섭다고 난리 쳤었잖아 그때 기억나?"
]

for question in questions:
    answer = query_rag(question)
    print(f"\n❓질문: {question}\n👉답변: {answer}")
