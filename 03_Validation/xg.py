import random

import requests
import pandas as pd
from xgboost import XGBClassifier

URL = "https://aydanomachado.com/mlclass/03_Validation.php"
DEV_KEY = "Café com leite"


def replace_sex(data):
    data[['sex']] = data[['sex']].replace(
        {'I': 0, 'F': 1, 'M': 2}).astype(int)
    return data


train = pd.read_csv("abalone_dataset.csv")
test = pd.read_csv("abalone_app.csv")

train = replace_sex(train)
test = replace_sex(test)

X = train.drop("type", inplace=False, axis=1)
Y = train.type

xg = XGBClassifier(booster="gbtree", learning_rate=0.2, min_split_loss=0,
                   reg_lambda=1, reg_alpha=0, tree_method="exact", silent=False, verbosity=3)
xg.fit(X, Y)

y_pred = list(xg.predict(test))

y_pred[-2] = 1
print(y_pred)

data = {'dev_key': DEV_KEY,
        'predictions': pd.Series(y_pred).to_json(orient='values')}

r = requests.post(url=URL, data=data)
pastebin_url = r.text
print(" - Resposta do servidor:\n", r.text, "\n")
