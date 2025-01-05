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

# 모델 저장
model_file_path = "word2vec_text8.model"
model_text8.save(model_file_path)
print(f"모델이 '{model_file_path}'로 저장되었습니다.")

