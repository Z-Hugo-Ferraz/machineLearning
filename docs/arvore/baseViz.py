import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("./docs/arvore/MBA.csv")

print(df.sample(frac=.0015).to_markdown(index=False))