import re
import pandas as pd
import unicodedata
from stop_words import get_stop_words
from nltk.stem import WordNetLemmatizer
import nltk

def preprocess_text(text):

    # Convertir a minúsculas y eliminar caracteres que no sean letras o espacios
    text = re.sub(r'[^a-zA-Z\s]', '', text).lower()

    # Tokenizar por espacios
    tokens = text.split()

    # Filtrar stopwords
    sw = get_stop_words(language='en')
    tokens = [word for word in tokens if word not in sw]

    # Lemmatizar
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return ' '.join(tokens)








