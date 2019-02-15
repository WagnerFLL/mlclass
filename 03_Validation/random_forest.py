import requests
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

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

rfm = RandomForestClassifier(n_estimators=70, oob_score=True, n_jobs=-1,
                             random_state=101, max_features=None, min_samples_leaf=30)

rfm.fit(X, Y)
y_pred = rfm.predict(test)

data = {'dev_key': DEV_KEY,
        'predictions': pd.Series(y_pred).to_json(orient='values')}

r = requests.post(url=URL, data=data)
pastebin_url = r.text
print(" - Resposta do servidor:\n", r.text, "\n")
