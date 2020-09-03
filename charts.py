import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="ticks", color_codes=True)
import pandas as pd
ivs=pd.read_csv("/home/pooja/PycharmProjects/datanalysis/iv2.csv")

sns.catplot(x='ivValue', y='variable',  kind="bar", data=ivs);