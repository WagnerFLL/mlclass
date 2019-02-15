import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def replace_sex(data):
    data[['sex']] = data[['sex']].replace(
        {'I': 0, 'F': 1, 'M': 2}).astype(int)
    return data


train = pd.read_csv("abalone_dataset.csv")
test = pd.read_csv("abalone_app.csv")

data = train.drop('sex', axis=1, inplace=False)
data2 = test.drop('sex', axis=1, inplace=False)

sns.countplot(x='type', data=train, palette='Set3')
sns.pairplot(data.drop('type', axis=1, inplace=False))
plt.figure(figsize=(20, 7))
sns.heatmap(data.corr(), annot=True)

sns.pairplot(data2)
plt.figure(figsize=(20, 7))
sns.heatmap(data2.corr(), annot=True)
plt.show()
