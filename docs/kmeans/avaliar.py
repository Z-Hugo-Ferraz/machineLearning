import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
from scipy.stats import mode
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA

# Carregar os dados
df = pd.read_csv("./docs/base/MBA.csv")

# Excluir as colunas não desejadas
df = df.drop(columns=["application_id", "international"])

# Preencher valores nulos
df["race"] = df["race"].fillna("international")
df["admission"] = df["admission"].fillna("Refused")

# Label encoding da coluna em texto binária
label_encoder = LabelEncoder()
df["gender"] = label_encoder.fit_transform(df["gender"])

# Escalonar variáveis contínuas
scaler = StandardScaler()
df[["gpa", "gmat", "work_exp"]] = scaler.fit_transform(df[["gpa", "gmat", "work_exp"]])

# Gerar dummies
df = pd.get_dummies(df, columns=["race", "major", "work_industry"], drop_first=True)

X = df.drop("admission", axis=1)
y = label_encoder.fit_transform(df["admission"])

#Separar em teste e validação
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reduzir para 2 dimensões com PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train)

# Treinar KMeans
kmeans = KMeans(n_clusters=3, init="k-means++", max_iter=100, random_state=42)
labels = kmeans.fit_predict(X_pca)

# Mapear clusters para classes reais por voto majoritário
cluster_map = {}
for c in np.unique(labels):
    mask = labels == c
    majority_class = mode(y_train[mask], keepdims=False)[0]
    cluster_map[c] = majority_class

# Reatribuir clusters como classes previstas
y_pred = np.array([cluster_map[c] for c in labels])

# Calcular acurácia e matriz de confusão
acc = accuracy_score(y_train, y_pred)
cm = confusion_matrix(y_train, y_pred)

cm_df = pd.DataFrame(
    cm,
    index=[f"Classe Real {cls}" for cls in np.unique(y_train)],
    columns=[f"Classe Pred {cls}" for cls in np.unique(y_train)]
)

print(f"Acurácia: {acc*100:.2f}%")
print("<br>Matriz de Confusão:")
print(cm_df.to_html())

