import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import seaborn as sns

plt.figure(figsize=(12, 10))

label_encoder = LabelEncoder()
scaler = StandardScaler()

df = pd.read_csv("./docs/base/MBA.csv")

#Excluir as conlunas não desejadas
df = df.drop(columns= ["application_id", "international"])

#Preencher os valores nulos da coluna "race"
df["race"] = df["race"].fillna("international")

#Preencher os valores nulos da coluna "admission"
df["admission"] = df["admission"].fillna("Refused")

#Label encoding da coluna em texto binária
df["gender"] = label_encoder.fit_transform(df["gender"])

#Escolonando as váriaveis continuas
df["gpa"] = scaler.fit_transform(df[["gpa"]])
df["gmat"] = scaler.fit_transform(df[["gmat"]])

#Gerando dummies das colunas em texto não binárias
df = pd.get_dummies(df,columns= ["race", "major", "work_industry"], drop_first=True)

print(df.sample(frac=.0015).to_markdown(index=False))