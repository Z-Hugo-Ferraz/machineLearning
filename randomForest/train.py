import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

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
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42, stratify=y)

# Initialize and train the model
rf = RandomForestClassifier(n_estimators=100,  # Number of trees
                            max_depth=5,       # Max depth of trees
                            max_features='sqrt',  # Features per split
                            random_state=42)
rf.fit(X_train, y_train)

# Predict and evaluate
predictions = rf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")

feature_importance = pd.DataFrame({
    'Feature': rf.feature_names_in_,
    'Importância': rf.feature_importances_
})
print("<br>Importância das Features:")
print(feature_importance.sort_values(by='Importância', ascending=False).to_html())