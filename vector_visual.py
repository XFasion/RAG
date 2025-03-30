from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import os
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# 데이터 로드
df = pd.read_csv("./conversation_dataset_expanded.csv")
sentences = df["dialogue"].tolist()

# KURE-v1 로딩
model = SentenceTransformer("nlpai-lab/KURE-v1", use_auth_token=HF_TOKEN)
embeddings = model.encode(sentences)

# 차원 축소 후 시각화
pca = PCA(n_components=2)
reduced = pca.fit_transform(embeddings)

plt.figure(figsize=(10, 7))
plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.7)

for i in range(0, len(sentences), 5):  # 문장 일부만 라벨링
    plt.text(reduced[i, 0], reduced[i, 1], sentences[i][:10] + "...", fontsize=8)

plt.title("KURE-v1 임베딩 시각화 (PCA)")
plt.grid(True)
plt.tight_layout()
plt.show()