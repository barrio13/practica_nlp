import random
import numpy as np
import pandas as pd
from stop_words import get_stop_words
from nltk.stem import WordNetLemmatizer
import re
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt

df = pd.read_csv('C:\\Users\\guill\\OneDrive\\Desktop\\nlp_pr\\Video_Games_5_sample.csv', sep=';', header=0)

print(df.head())

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
df['reviewText'] = df['reviewText'].apply(preprocess_text)


# etiquetas positivo,negativo.
def label_sentiment(row):
    if int(row['overall']) < 3:
        return 0
    else:
        return 1

df['sentiment_label'] = df.apply(lambda row: label_sentiment(row), axis=1)

# Separamos en train y test.
X_train, X_test, y_train, y_test = train_test_split(
    df['reviewText'],
    df['sentiment_label'],
    train_size=0.75,
    test_size=0.25,
    random_state=42,
    shuffle=True
)

print(X_train.iloc[:10])

# Extracción features.

cv = TfidfVectorizer(
    max_df=0.95,
    min_df=5,
    max_features=2500,
    strip_accents='ascii',
    ngram_range=(1, 1)
)
cv.fit(X_train)


X_train_ = cv.transform(X_train)
X_test_ = cv.transform(X_test)

c_params = [0.01, 0.05, 0.25, 0.5, 1, 10, 100, 1000, 10000]

train_acc = list()
test_acc = list()
for c in c_params:
    lr = LogisticRegression(C=c, solver='lbfgs', max_iter=500)
    lr.fit(X_train_, y_train)

    train_predict = lr.predict(X_train_)
    test_predict = lr.predict(X_test_)

    print ("Accuracy for C={}: {}".format(c, accuracy_score(y_test, test_predict)))

    train_acc.append(accuracy_score(y_train, train_predict))
    test_acc.append(accuracy_score(y_test, test_predict))


print('Confussion matrix:\n{}'.format(confusion_matrix(y_test, test_predict)))
print('\nClassification report:\n{}'.format(classification_report(y_test, test_predict)))
print('Accuracy score:{}'.format(accuracy_score(y_test, test_predict)))


plt.figure(figsize=(12, 8))
plt.plot(train_acc, label='train')
plt.plot(test_acc, label='test')
plt.axvline(np.argmax(test_acc), c='g', ls='--', alpha=0.8)
plt.title('Accuracy evolution for different C values')
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.xticks(list(range(len(c_params))), c_params)
plt.tight_layout()
plt.show()

# Según estos resultados deberíamos escoger C = 10, si cogemos un C más grande el modelo no generaliza bien.
# Los resultados de test son cada vez peores.

p, r, thresholds = precision_recall_curve(y_test, test_predict)
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.figure(figsize=(8, 8))
    plt.title("Precision and Recall Scores as a function of the decision threshold")
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.ylabel("Score")
    plt.xlabel("Decision Threshold")
    plt.legend(loc='best')


plot_precision_recall_vs_threshold(p, r, thresholds)
plt.show()