import os
from dotenv import load_dotenv
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 환경 변수 로드
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# 벡터 DB 로드
storage_context = StorageContext.from_defaults(persist_dir="./faiss_conversation_db_expanded")
index = load_index_from_storage(storage_context)

# 임베딩 모델 로딩 (현재 사용 중인 모델과 일치시켜야 함)
embed_model = HuggingFaceEmbedding(
    model_name="nlpai-lab/KURE-v1",
    device="cpu",
    token=HF_TOKEN
)

# Settings.embed_model = embed_model  # 필요 시 설정 반영

# 벡터 스토어 접근
vector_store = index.vector_store

# 첫 번째 임베딩 벡터 추출
first_embedding = next(iter(vector_store._data.embedding_dict.values()))

# 디버깅 정보 출력
model_name = getattr(embed_model, "model_name", "알 수 없음")
model_type = embed_model.__class__.__name__
embedding_dim = len(first_embedding)

print("\n✅ 벡터 DB 디버깅 정보")
print(f"• 임베딩 모델 종류: {model_type}")
print(f"• 사용한 모델 이름: {model_name}")
print(f"• 임베딩 벡터 차원 수: {embedding_dim}차원")
print(f"• 임베딩 값 예시 (앞 5개): {first_embedding[:5]}")