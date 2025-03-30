import os
from dotenv import load_dotenv
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Document

# 환경변수 로드
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# 기존 DB 경로
persist_dir = "./faiss_conversation_db_expanded"

# 임베딩 모델 로딩 (KURE-v1)
embed_model = HuggingFaceEmbedding(
    model_name="nlpai-lab/KURE-v1",
    device="cpu",
    token=HF_TOKEN
)

# 기존 인덱스 로드
storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
index = load_index_from_storage(storage_context)
vector_store = index.vector_store

# 문장 추가 함수
def add_message_to_index(speaker: str, dialogue: str):
    text = f"{speaker}: {dialogue}"
    document = Document(text=text)

    # 문서 임베딩 및 추가
    index.insert(document)
    index.storage_context.persist(persist_dir=persist_dir)

    print("✅ 문장 추가 완료 및 DB 저장됨")
    print(f"➡️ 추가된 문장: {text}")

# 예시 호출
if __name__ == "__main__":
    speaker = "사용자"
    dialogue = "닌텐도 말하는거야? 당연하지!"
    add_message_to_index(speaker, dialogue)