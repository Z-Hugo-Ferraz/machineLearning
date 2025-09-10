import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("./docs/base/MBA.csv")

print(df.sample(frac=.0015).to_markdown(index=False))