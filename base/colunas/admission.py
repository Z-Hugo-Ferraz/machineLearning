import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO

df = pd.read_csv("./docs/base/MBA.csv")

df["admission"] = df["admission"].fillna("null")

count = pd.Series(df["admission"]).value_counts()

fig, ax = plt.subplots(figsize=(8, 4))

ax.pie(count, labels=count.index, colors=["darkgreen", "steelblue", "grey"], autopct="%1.1f%%")

ax.set_title("Composição da coluna")

buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())