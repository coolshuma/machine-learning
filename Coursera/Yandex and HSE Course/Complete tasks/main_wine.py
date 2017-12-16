import pandas
from sklearn.model_selection import KFold, cross_val_score
#from sklearn.cross_validation import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import sklearn.preprocessing
import numpy as np

A = pandas.read_csv('wine.data', header=None)
print A
wine_class = A[0]
print wine_class
del A[0]
wine_attributes = A
print wine_attributes

max_qual = 0.0
best_k = 0
wine_attributes = sklearn.preprocessing.scale(wine_attributes)
kf = KFold(shuffle=True, n_splits=5, random_state=42)
for k in range(1, 51):
    clf = KNeighborsClassifier(n_neighbors=k)
    #clf = clf.fit(wine_attributes, wine_class)
    qual = cross_val_score(clf, wine_attributes,
                                                   wine_class, cv=kf, scoring='accuracy')
    print qual
    current_qual = qual.mean()
    print current_qual
    if (max_qual <= current_qual):
        max_qual = current_qual
        best_k = k
print best_k
print max_qual

wine_attributes = sklearn.preprocessing.scale(wine_attributes)
