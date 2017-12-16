from sklearn.datasets import load_boston
from sklearn.preprocessing import scale
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import numpy as np

A = load_boston()
data = A['data']
data = scale(data)
target = A['target']
t = np.linspace(1, 10, num=200)
min_mean = -100.0
best_p = 0.0
kf = KFold(shuffle=True, n_splits=5, random_state=42)
for cur_p in t:
    clf = KNeighborsRegressor(p=cur_p, n_neighbors=5, weights='distance')
    qual = cross_val_score(clf, data, target, cv=kf, scoring='neg_mean_squared_error')
    current_qual = qual.max()
    if (current_qual > min_mean):
        min_mean = current_qual
        best_p = cur_p
print best_p
