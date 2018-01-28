import pandas
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler

Train_feature = pandas.read_csv('perceptron-train.csv', header=None)
Train_class = Train_feature[0]
del Train_feature[0]

Test_feature = pandas.read_csv('perceptron-test.csv', header=None)
Test_class = Test_feature[0]
del Test_feature[0]

clf_train = Perceptron(random_state=241)
clf_train.fit(Train_feature, Train_class)
predictions = clf_train.predict(Test_feature)
accuracy = accuracy_score(Test_class, predictions)
print accuracy


scaler = StandardScaler()
Train_feature = scaler.fit_transform(Train_feature)
Test_feature = scaler.transform(Test_feature)
print Train_feature
print Test_feature
clf_train.fit(Train_feature, Train_class)
predictions = clf_train.predict(Test_feature)
scaling_accuracy = accuracy_score(Test_class, predictions)
print scaling_accuracy
print (scaling_accuracy - accuracy)
