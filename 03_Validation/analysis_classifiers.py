from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeClassifier

from sklearn.neural_network import MLPRegressor
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def replace_sex(data):
    data[['sex']] = data[['sex']].replace(
        {'I': 0, 'F': 1, 'M': 2}).astype(int)
    return data



models = []
names = []



train = pd.read_csv("abalone_dataset.csv")

categorical_columns = [col for col in train.columns if train[col].dtype == 'object']
print("Categóricas: ", end="")
print(categorical_columns)
print()

train = replace_sex(train)

# X = train.drop(['type', 'whole_weight', 'shucked_weight'], inplace=False, axis=1)
# X = train.drop(['type', 'shucked_weight'], inplace=False, axis=1)
X = train.drop(['type'], inplace=False, axis=1)
Y = train.type

# XGBClassifier
xg = XGBClassifier(booster="gbtree", learning_rate=0.2, min_split_loss=0,
                   reg_lambda=1, reg_alpha=0, tree_method="exact")
xg_scores = cross_val_score(xg, X, Y, cv=8)

# RandomForestClassifier
rfm = RandomForestClassifier(n_estimators=70, oob_score=True, n_jobs=-1,
                             random_state=101, max_features=None, min_samples_leaf=30)
rfm_scores = cross_val_score(rfm, X, Y, cv=8)

# KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=4)
neigh_scores = cross_val_score(neigh, X, Y, cv=8)

# DecisionTreeClassifier
tree = DecisionTreeClassifier(max_depth=10, random_state=101,
                              max_features=None, min_samples_leaf=15)
tree_scores = cross_val_score(tree, X, Y, cv=8)


print("XBoost: " + str(xg_scores))
print("RandomForest: " + str(np.mean(rfm_scores) * 100.0))
print("Knn: " + str(np.mean(neigh_scores) * 100.0))
print("Tree: " + str(np.mean(tree_scores) * 100.0))
# print("MLPRegressor: " + str(mlp_score))
