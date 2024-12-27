from transformers import pipeline

# 1. 감정 분석 사전학습모델 로딩
sentiment_analyzer = pipeline("text-classification", model="matthewburke/korean_sentiment")

text1 = "배달이 한시간 걸리고, 음식은 다 식어서 왔다"
print(sentiment_analyzer(text1))

text2 = "음식이 너무 맛있다"
print(sentiment_analyzer(text2))

# 2. 번역 모델
translator = pipeline('translation_en_to_fr', model='Helsinki-NLP/opus-mt-en-fr')
english = "i want go home"
translated_text = translator(english)
print("english : ", english)
print("translated_text : ", translated_text[0]["translation_text"])