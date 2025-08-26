import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()


df = pd.read_csv("./docs/arvore/MBA.csv")

#Excluir as conlunas não desejadas
df = df.drop(columns= ["application_id", "international"])

#Preencher os valores nulos da coluna "race"
df["race"] = df["race"].fillna("international")

#Preencher os valores nulos da coluna "admission"
df["admission"] = df["admission"].fillna("Refused")

#Label encoding das colunas em texto
df["race"] = label_encoder.fit_transform(df["race"])
df["gender"] = label_encoder.fit_transform(df["gender"])
df["major"] = label_encoder.fit_transform(df["major"])
df["work_industry"] = label_encoder.fit_transform(df["work_industry"])

print(df.sample(frac=.0015).to_markdown(index=False))