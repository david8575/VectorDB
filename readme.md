# 벡터DB

# 1. Text Embedding

기존 비정형 데이터 형태의 자연어 데이터를 n차원 수치형 벡터로 표상화시킨 데이터

머신러닝시 분석에 용이함

해당 언어가 가지는 컨텍스팅을 나타냄(단순한 텍스트 → 숫자 매핑이 아님)

## 1.2. Sparse Vector & Dense Vector

### 1.2.1. Sparse Vector

구문 정보 중심의 형상

단어와 단어간의 등장 빈도

저장 공간 비효율적

### 1.2.2. Dense Vector

의미 정보 중심의 형상

임베딩 벡터가 해당됨

저장 공간 효율적

## 1.2. Word2Vec

Dense Embedding Model

단어를 벡터로 변환하는 자연어 처리 기법

신경망 기반 언어 모델

문맥상 비슷한 위치의 단어들은 비슷한 의미를 가짐

특수한 상황에 맞게 파인튜닝을 해야 함

### 1.2.1 학습 메커니즘

1. CBOW(Continuous Bag of Words): 주변 단어들을 이용해 현재 단어를 예측
input 레이어에 앞 뒤로 4개의 단어를 입력하여 하나의 output단어를 출력
2. Skip-Gram: 현재 단어를 이용해 주변 단어들을 예측, 성능이 더 좋음
input 레이어에 1개의 단어를 입력하여 앞뒤 4개의 output단어들을 출력

## 1.3.Glove, FastText

Word2Vec의 고도화 버전

Shallow 신경망

## 1.4. Bi-LSTM

ELMo(Context-Sensitive 형상)

Deep한 LSTM 구성

## 1.5. Transformer

BERT, GPT 등 Attention 메커니즘 기반

Deep Neural Network