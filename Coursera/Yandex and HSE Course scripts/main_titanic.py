import pandas
import numpy as np

A = pandas.read_csv('titanic.csv')
A = A[['Survived', 'Pclass', 'Fare', 'Age', 'Sex']]
A = A.dropna()
data = A[['Pclass', 'Fare', 'Age', 'Sex']]
print data
data['Sex'].replace('female', 0, inplace=True)
data['Sex'].replace('male', 1, inplace=True)
print data
var = A["Survived"]
print var

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=241)
clf.fit(data, var)
importances = clf.feature_importances_
print importances