import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO

df = pd.read_csv("./docs/arvore/MBA.csv")

fig, ax = plt.subplots(figsize=(8, 4))

ax.hist(df["application_id"], bins=6, color="darkgreen")

ax.set_xlabel("ID da aplcação")
ax.set_ylabel("Frequência")
ax.set_title("Composição da coluna")

buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())