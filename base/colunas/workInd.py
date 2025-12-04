import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO

df = pd.read_csv("./docs/base/MBA.csv")

fig, ax = plt.subplots(figsize=(10, 6))

count = df["work_industry"].value_counts()

ax.bar(count.index.astype(str), count.values, color="darkgreen")

ax.set_title("Composição da coluna")
ax.set_ylabel("Frequência")

plt.xticks(rotation=35)

buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())