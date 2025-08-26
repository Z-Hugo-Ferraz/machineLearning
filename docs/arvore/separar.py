import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

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

#Separar em vairaveis indenpendetes e dependente
x = df[["gender", "gpa", "major", "race", "gmat", "work_exp", "work_industry"]]
y = df["admission"]

#Separar em teste e validação
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)