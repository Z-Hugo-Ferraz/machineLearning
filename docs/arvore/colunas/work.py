import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO

df = pd.read_csv("./docs/arvore/MBA.csv")

fig, ax = plt.subplots(figsize=(8, 5))

count = pd.cut(df["work_exp"], bins=7).value_counts().sort_index()

ax.bar(count.index.astype(str), count.values, color="darkgreen")

ax.set_title("Composição da coluna")
ax.set_ylabel("Anos de experiência")

plt.xticks(rotation=15)

buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())