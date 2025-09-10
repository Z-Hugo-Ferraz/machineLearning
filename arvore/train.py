import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree

label_encoder = LabelEncoder()


df = pd.read_csv("./docs/base/MBA.csv")

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
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=27, stratify=y)

# Criar e treinar o modelo de árvore de decisão
classifier = tree.DecisionTreeClassifier()
classifier.fit(x_train, y_train)

# Avaliar o modelo
y_pred = classifier.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisão da Validação: {accuracy:.4f}")

feature_importance = pd.DataFrame({
    'Feature': classifier.feature_names_in_,
    'Importância': classifier.feature_importances_
})
print("<br>Importância das Features:")
print(feature_importance.sort_values(by='Importância', ascending=False).to_html())

plt.figure(figsize=(20, 10))
tree.plot_tree(classifier, max_depth=5, fontsize=10)

# Para imprimir na página HTML
buffer = StringIO()
plt.savefig(buffer, format="svg")
print(buffer.getvalue())