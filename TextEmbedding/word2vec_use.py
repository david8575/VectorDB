from gensim.models import Word2Vec

# 저장된 모델 불러오기
model_file_path = "word2vec_text8.model"
model_text8 = Word2Vec.load(model_file_path)
print(f"모델 '{model_file_path}'이 성공적으로 불러와졌습니다.")


# 학습 모델 활용 예시
example_word1 = "king"
print(f"{example_word1}의 단어 임베딩:")
print(model_text8.wv[example_word1])

example_word2 = "queen"
print(f"{example_word2}의 단어 임베딩:")
print(model_text8.wv[example_word2])

# 단어 유사도 계산 확인
word_examples = ["king", "orange", "apple"]
for word in word_examples:
    print(f"{word}와 비슷한 단어:")
    similar_words = model_text8.wv.most_similar(word, topn=5)
    for similar_word, similarity in similar_words:
        print(f"    {similar_word} (유사도: {similarity:.4f})")