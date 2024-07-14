import re
import pandas as pd
import unicodedata
from stop_words import get_stop_words
from nltk.stem import WordNetLemmatizer
import nltk

def preprocess_text(text):

    # Convertimos a minúsculas y eliminamos caracteres que no sean letras o espacios.
    text = re.sub(r'[^a-zA-Z\s]', '', text).lower()

    # Tokenizamos por espacios.
    tokens = text.split()

    # Filtramos stopwords.
    sw = get_stop_words(language='en')
    tokens = [word for word in tokens if word not in sw]

    # Lemmatizamos.
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return ' '.join(tokens)








