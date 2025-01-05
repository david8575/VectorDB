import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import gensim.downloader as api
from transformers import BertTokenizer, BertModel
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity

# 비교 문장
sentences = ["Bill ran from the giraffe toward the dolphin",
             "Bill ran from the dolphin toward the giraffe"]

# sparse vector
# 단어의 등장 빈도를 카운트
# 예시의 경우 7차원
vectorizer = CountVectorizer()
sparse_vectors = vectorizer.fit_transform(sentences)
print("sparse vector")
print(sparse_vectors.toarray())

# dense vector
# bert 모델 이용
# from_pretrained으로 사전 학습 모델을 불러올 수 있음
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
# 변환함수
def create_dense_vector(sentence):
    # tensor형태로 문장 변환
    inputs = tokenizer(sentence, return_tensors='pt') 
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

dense_vector = [create_dense_vector(sentence) for sentence in sentences]

print("dense vector")
for v in dense_vector:
    print(v)

# 코사인 유사도 계산
# 1. sparse vector 기준
sparse_similarity = cosine_similarity(sparse_vectors[0], sparse_vectors[1])
print("cosine similarity : ", sparse_similarity[0][0])
# 2. dense vector 기준
def cal_cosine_sim(v1,v2):
    return F.cosine_similarity(v1,v2).item()
dense_similarity = cal_cosine_sim(dense_vector[0], dense_vector[1])
print("cosine similarity : ", dense_similarity)