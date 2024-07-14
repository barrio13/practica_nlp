import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
from nltk import ngrams
from nltk.probability import FreqDist
from stop_words import get_stop_words
from wordcloud import WordCloud
from gensim.models import Word2Vec
import multiprocessing
import re
from nltk.stem import WordNetLemmatizer
import nltk
from wordcloud import WordCloud, STOPWORDS

nltk.download('wordnet')
nltk.download('omw-1.4') 

df = pd.read_csv('C:\\Users\\guill\\OneDrive\\Desktop\\nlp_pr1\\Video_Games_5_sample.csv', sep=';', header=0)

print(df.head())

# convertimos a minúsculas y tokenizamos por espacios.
splitted_reviews = df['reviewText'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x).lower().split())
words = splitted_reviews.apply(pd.Series).stack().reset_index(drop = True)

# Cardinalidad del vocabulario.
vocabulary = Counter(words)
print(len(vocabulary))  # 66009


# Distribución de reviews por número de estrellas.
df_p1 = pd.DataFrame(df['overall'].value_counts(sort=False).sort_index())
df_p1 = df_p1.plot(kind='bar', title='Distribución reviews por número de estrellas', figsize=(8, 4))
plt.show()

# Vemos que hay un problema de grupos desbalanceados.

# etiquetas positivo,negativo.
def label_sentiment(row):
    if int(row['overall']) < 3:
        return 0
    else:
        return 1

df['sentiment_label'] = df.apply(lambda row: label_sentiment(row), axis=1)

print(np.shape(df))

# Número de reseñas negativas y positivas.
numero_de_reviews_positivas = df['sentiment_label'].sum()
print('reviews positivas: ', numero_de_reviews_positivas)                     # 8860
numero_de_reviews_negativas =  np.shape(df)[0] - numero_de_reviews_positivas
print('reviews negativas: ', numero_de_reviews_negativas)                     # 1138

# N-grams más frecuentes.
bigrams = ngrams(words, 2)
trigrams = ngrams(words, 3)
bigrams_freq = FreqDist(bigrams)
trigrams_freq = FreqDist(trigrams)

print(bigrams_freq.most_common(10))
print(trigrams_freq.most_common(10))

# Vemos que tenemos que hacer eliminación de stopwords en inglés.

# Eliminar stopwords en inglés
sw = get_stop_words(language='en')
words_new = [word for word in words if word not in sw]

# Actualizar el contador de vocabulario con las palabras filtradas
vocabulary_new = Counter(words_new)
print("Tamaño del vocabulario después de eliminar stopwords:", len(vocabulary_new))


# N-grams más frecuentes despúes de  aplicar stopword.
bigrams = ngrams(words_new, 2)
trigrams = ngrams(words_new, 3)
bigrams_freq = FreqDist(bigrams)
trigrams_freq = FreqDist(trigrams)

print(bigrams_freq.most_common(10))
print(trigrams_freq.most_common(10))

bg_freq_most_common = bigrams_freq.most_common(10)
bgs_ = [str(bg[0]) for bg in bg_freq_most_common]
bgs_f_ = [bg[1] for bg in bg_freq_most_common]

tg_freq_most_common = trigrams_freq.most_common(10)
tgs_ = [str(tg[0]) for tg in tg_freq_most_common]
tgs_f_ = [tg[1] for tg in tg_freq_most_common]

bgs_f_, bgs_ = zip(*sorted(zip(bgs_f_, bgs_)))
tgs_f_, tgs_ = zip(*sorted(zip(tgs_f_, tgs_)))

plt.barh(bgs_, bgs_f_)
plt.title('Bigram frequencies')
plt.show()

plt.barh(tgs_, tgs_f_)
plt.title('Trigram frequencies')
plt.show()

# Podríamos plantearnos quitar games, game, plays, player. 
# Palabras comunes a ambos sentimientos que no nos aportan más información.

# Nubes de palabras

# Combinamos las reviews para el sentimiento negativo.
combined_text = " ".join([review for review, label in zip(df['reviewText'], df['sentiment_label']) if label == 0])

# Inicializamos el wordcloud
wc = WordCloud(background_color='white', max_words=50,
               stopwords=STOPWORDS.update(['games','game','players','play','player']))

# Generamos el wordcloud
wc.generate(combined_text)

# Mostramos el wordcloud
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()

# Combinamos las reviews para el sentimiento positivo.
combined_text = " ".join([review for review, label in zip(df['reviewText'], df['sentiment_label']) if label == 1])

# Inicializamos el wordcloud
wc = WordCloud(background_color='white', max_words=50,
               stopwords=STOPWORDS.update(['games','game','players','play','player']))

# Generamos el wordcloud
wc.generate(combined_text)

# Mostramos el wordcloud
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()

# Word embeddings
sentences = df['reviewText'].str.lower().apply(lambda x: [word for word in x.split() if word not in sw]).tolist()

cores = multiprocessing.cpu_count()



w2v_model = Word2Vec(min_count=20,
                     window=2,
                     vector_size=300,
                     sample=6e-5,
                     alpha=0.03,
                     min_alpha=0.0007,
                     negative=20,
                     workers=cores-1)

w2v_model.build_vocab(sentences, progress_per=10000)


w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)


# Miramos palabras similares de las que puedan ser representativas de un sentimiento.

similar_words1 = w2v_model.wv.most_similar(positive=["fun"])
print(similar_words1)
similar_words2 = w2v_model.wv.most_similar(positive=["problem"])
print(similar_words2)
similar_words3 = w2v_model.wv.most_similar(positive=["great"])
print(similar_words3)
similar_words4 = w2v_model.wv.most_similar(positive=["feel"])
print(similar_words4)