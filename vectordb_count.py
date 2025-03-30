import os
from dotenv import load_dotenv
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import StorageContext, load_index_from_storage, Settings

# 환경 변수 로드
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# ✅ HuggingFace 임베딩 모델 명시 설정
embed_model = HuggingFaceEmbedding(
    model_name="nlpai-lab/KURE-v1",
    device="cpu",
    token=HF_TOKEN
)
Settings.embed_model = embed_model  # 🔑 이게 핵심!

# 저장된 벡터 DB 로드
storage_context = StorageContext.from_defaults(persist_dir="./faiss_conversation_db_expanded")
index = load_index_from_storage(storage_context)

# 벡터 스토어 접근
vector_store = index.vector_store
embedding_count = len(vector_store._data.embedding_dict)
print(f"🔍 현재 벡터 DB에 저장된 문장 수: {embedding_count}")