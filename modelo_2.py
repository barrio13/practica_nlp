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
from sklearn.ensemble import RandomForestClassifier  # Cambiado a Classifier
from sklearn.model_selection import GridSearchCV

# Cargar datos
df = pd.read_csv('C:\\Users\\guill\\OneDrive\\Desktop\\nlp_pr\\Video_Games_5_sample.csv', sep=';', header=0)

print(df.head())

# Preprocesamiento de texto
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text).lower()
    tokens = text.split()
    sw = get_stop_words(language='en')
    tokens = [word for word in tokens if word not in sw]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

df['reviewText'] = df['reviewText'].apply(preprocess_text)

# Etiquetas positivo, negativo
def label_sentiment(row):
    if int(row['overall']) < 3:
        return 0
    else:
        return 1

df['sentiment_label'] = df.apply(lambda row: label_sentiment(row), axis=1)

# Separar en train y test
X_train, X_test, y_train, y_test = train_test_split(
    df['reviewText'],
    df['sentiment_label'],
    train_size=0.75,
    test_size=0.25,
    random_state=42,
    shuffle=True
)

print(X_train.iloc[:10])

# Extracción de características
cv = CountVectorizer(ngram_range=(1, 2), max_features=2500, max_df=0.95, min_df=5)
cv.fit(X_train)
X_train_ = cv.transform(X_train)
X_test_ = cv.transform(X_test)

# Random Forest
maxDepth = range(1, 15)
tuned_parameters = {'max_depth': maxDepth}

grid = GridSearchCV(RandomForestClassifier(random_state=0, n_estimators=200, max_features='sqrt'), param_grid=tuned_parameters, cv=3, verbose=2)
grid.fit(X_train_, y_train)  # Ajustar a X_train_ en lugar de X_train

print("best mean cross-validation score: {:.3f}".format(grid.best_score_))
print("best parameters: {}".format(grid.best_params_))

scores = np.array(grid.cv_results_['mean_test_score'])
plt.plot(maxDepth, scores, '-o')
plt.xlabel('max_depth')
plt.ylabel('10-fold ACC')

plt.show()

maxDepthOptimo = grid.best_params_['max_depth']
randomForest = RandomForestClassifier(max_depth=maxDepthOptimo, n_estimators=200, max_features='sqrt')
randomForest.fit(X_train_, y_train)  # Ajustar a X_train_ en lugar de X_train

print("Train: ", randomForest.score(X_train_, y_train))  # Evaluar en X_train_
print("Test: ", randomForest.score(X_test_, y_test))  # Evaluar en X_test_


# best mean cross-validation score: 0.887
# best parameters: {'max_depth': 14}
# Train:  0.8973059482528675
# Test:  0.8876