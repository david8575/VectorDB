from gensim.models import Word2Vec
import gensim.downloader as api
import warnings
warnings.filterwarnings('ignore')

# 데이터셋 로드
dataset = api.load('text8')

# 데이터셋 확인
"""
for i, doc in enumerate(dataset):
    print(f"Document {i+1}: {doc}...")
    if (i == 2):
        break
"""

# 모델 학습
model_text8 = Word2Vec(dataset, vector_size = 100, window=5, min_count=150, workers=4)

# 학습 모델 활용 예시
