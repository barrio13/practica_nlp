import pandas as pd
import gzip
import json
import numpy as np
import pandas as pd


# Descargamos el dasaset con la documentación de la página.

def parse(path):
  g = gzip.open(path, 'r')
  for l in g:
    yield json.loads(l)

def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

df = getDF('C:\\Users\\guill\\OneDrive\\Desktop\\nlp_p1r\\Video_Games_5.json.gz')

print(np.shape(df))


# Obtenemos una muestra aleatoria de 10000 registros.
df_sample = df.sample(n=10000, random_state=42)

print(df_sample.info())
print(np.shape(df_sample))


# Nos quedamos con las columnas que nos interesan.
df_sample = df_sample[['overall','reviewText']]

df_sample.dropna(subset=['overall','reviewText'], inplace=True)
df_sample.drop_duplicates()

print(np.shape(df_sample))

# Guardamos el Dataframe.
df_sample.to_csv('C:\\Users\\guill\\OneDrive\\Desktop\\nlp_pr1\\Video_Games_5_sample.csv', index=False, sep = ';')