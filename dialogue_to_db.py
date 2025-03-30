import pandas as pd
from llama_index.core import Document, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import os
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

df = pd.read_csv("./conversation_dataset_expanded.csv")
documents = [
    Document(text=f"{row['speaker']}: {row['dialogue']}") for _, row in df.iterrows()
]

embed_model = HuggingFaceEmbedding(
    model_name="nlpai-lab/KURE-v1",
    device="cpu",
    token=HF_TOKEN
)

index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
index.storage_context.persist(persist_dir="./faiss_conversation_db_expanded")

print("✅ conversation_id 없는 FAISS DB 생성 완료.")

# FAISS DB 생성 완료 후, 임베딩 차원 확인용 디버깅 코드 추가
vector_store = index.vector_store

# 첫 번째 문서의 임베딩 벡터 확인
first_embedding = next(iter(vector_store._data.embedding_dict.values()))

print("✅ KURE-v1 FAISS DB 생성 완료.")
print(f"✅ 임베딩 차원 확인: {len(first_embedding)}차원")

# 차원이 1024차원이라면 KURE-v1 모델이 맞습니다.
assert len(first_embedding) == 1024, "⚠️ 임베딩 차원이 KURE-v1(1024)이 아닙니다!"
