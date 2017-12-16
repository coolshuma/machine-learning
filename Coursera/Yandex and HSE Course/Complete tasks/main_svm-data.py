import pandas
import numpy as np
from sklearn.svm import SVC

features = pandas.read_csv('svm-data.csv', header=None)
classes = features[0]
del features[0]

clf = SVC(C=100000, random_state=241, kernel='linear')
clf.fit(features, classes)
sup = clf.support_
print sup
