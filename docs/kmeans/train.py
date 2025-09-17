import base64
from io import BytesIO
import matplotlib.pyplot as plt
import pandas as pd
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

# Separar variáveis independentes e dependente
X = df.drop("admission", axis=1)

# Reduzir para 2 dimensões com PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Treinar KMeans
kmeans = KMeans(n_clusters=2, init="k-means++", max_iter=100, random_state=42)
labels = kmeans.fit_predict(X_pca)

# Plot
plt.figure(figsize=(12, 10))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="viridis", s=50)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            c="red", marker="*", s=200, label="Centroids")
plt.title("K-Means Clustering Results (PCA 2D)")
plt.xlabel("PCA Feature 1")
plt.ylabel("PCA Feature 2")
plt.legend()

# Salvar em buffer png
buffer = BytesIO()
plt.savefig(buffer, format="png", transparent=True, bbox_inches="tight")
buffer.seek(0)

# Converter em base64
img_base64 = base64.b64encode(buffer.read()).decode("utf-8")

# Criar tag HTML para embutir no MkDocs
html_img = f'<img src="data:image/png;base64,{img_base64}" alt="KMeans clustering" />'

print(html_img)
