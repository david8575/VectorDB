"""
import pandas as pd
from collections import Counter
df = pd.read_csv("datasets/simpsons_dataset.csv")
print(df.shape)
counts = Counter(df["raw_character_text"])
print(counts.most_common(5))
"""

from typing import Optional
import nltk
import pandas as pd
import spacy
import re
from gensim.models import Word2Vec
import numpy as np
from numpy.linalg import norm

nltk.download('punkt')
nltk.download('stopwords')

df = pd.read_csv("datasets/simpsons_dataset.csv")

# 데이터 확인
print(df.shape)
print(df.head())
print("========================================================================================")
print(df.isnull().sum())
print("========================================================================================")
print(df.loc[0, 'spoken_words'])

# 데이터 전처리

nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])

def cleaning(doc):
    """
    Cleans a spaCy Doc object by lemmatizing its tokens and removing stop words,
    then joins the remaining tokens into a single string if there are more than two tokens left.
    
    Parameters:
    ----------
    doc : spacy.tokens.Doc
        A spaCy Doc object containing the processed text.
    
    Returns:
    ----------
    Optional : str
        A string composed of the lemmatized, non-stop tokens separated by spaces,
        if the resulting list of tokens has more than two elements. Otherwise, returns None.
    """

    txt = [token.lemma_ for token in doc if not token.is_stop]
    if len(txt) > 2:
        return ' '.join(txt)
    
cleaner = (re.sub(r'[^A-Za-z\s]', '', df.loc[0, 'spoken_words']).lower() for row in df['spoken_words'])
txt = [cleaning(doc) for doc in nlp.pipe(cleaner, batch_size=5000)]

print(txt[0])