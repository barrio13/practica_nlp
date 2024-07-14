import pandas as pd
import re
from stop_words import get_stop_words
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split

# Carga de datos
df = pd.read_csv('C:\\Users\\guill\\OneDrive\\Desktop\\nlp_pr\\Video_Games_5_sample.csv', sep=';', header=0)
print(df.head())

# Preprocesamiento de texto
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

# Aplicar preprocesamiento a las reseñas
df['processed_reviewText'] = df['reviewText'].apply(preprocess_text)

# Crear etiquetas de sentimiento
def label_sentiment(row):
    if int(row['overall']) < 3:
        return 0
    else:
        return 1

df['sentiment_label'] = df.apply(lambda row: label_sentiment(row), axis=1)

# Dividir en train y test
X_train, X_test, y_train, y_test = train_test_split(
    df['processed_reviewText'],
    df['sentiment_label'],
    train_size=0.75,
    test_size=0.25,
    random_state=42,
    shuffle=True
)

# Mostrar las primeras 10 reseñas del conjunto de entrenamiento
print(X_train.iloc[:10])