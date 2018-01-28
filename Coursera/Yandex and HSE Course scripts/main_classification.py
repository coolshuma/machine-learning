import pandas
import numpy as np
import sklearn.metrics
features = pandas.read_csv('classification.csv', header=None)

TP = 0
FP = 0
FN = 0
TN = 0
print features.shape[0]
shape = features.shape[0]
for i in range(1, shape):
    if features[0][i] == '1' and features[1][i] == '1':
        TP += 1
    if features[0][i] == '0' and features[1][i] == '1':
        FP += 1
    if features[0][i] == '1' and features[1][i] == '0':
        FN += 1
    if features[0][i] == '0' and features[1][i] == '0':
        TN += 1

print TP, FP, FN, TN

true = features[0]
pred = features[1]
del true[0]
del pred[0]
print true

print sklearn.metrics.accuracy_score(true, pred)
print sklearn.metrics.precision_score(true, pred, average='binary', pos_label='1')
print sklearn.metrics.recall_score(true, pred, average='binary', pos_label='1')
print sklearn.metrics.f1_score(true, pred, average='binary', pos_label='1')

accuracy = float(TP) / float(TP + FP)
print accuracy
print ''

scores = pandas.read_csv('scores.csv', header=None, skiprows=[0])
true = scores[0]
logreg = scores[1]
svm = scores[2]
knn = scores[3]
tree = scores[4]
print sklearn.metrics.roc_auc_score(true, logreg)
print sklearn.metrics.roc_auc_score(true, svm)
print sklearn.metrics.roc_auc_score(true, knn)
print sklearn.metrics.roc_auc_score(true, tree)
print ''

prec, recall, steps = \
    sklearn.metrics.precision_recall_curve(true, logreg)
max_prec_log = 0.0
for i in range(0, steps.shape[0]):
    if recall[i] >= 0.7:
        max_prec_log = max(max_prec_log, prec[i])
print max_prec_log

prec, recall, steps = \
    sklearn.metrics.precision_recall_curve(true, svm)
max_prec_svm = 0.0
for i in range(0, steps.shape[0]):
    if recall[i] >= 0.7:
        max_prec_svm = max(max_prec_svm, prec[i])
print max_prec_svm

prec, recall, steps = \
    sklearn.metrics.precision_recall_curve(true, knn)
max_prec_knn = 0.0
for i in range(0, steps.shape[0]):
    if recall[i] >= 0.7:
        max_prec_knn = max(max_prec_knn, prec[i])
print max_prec_knn

prec, recall, steps = \
    sklearn.metrics.precision_recall_curve(true, tree)
max_prec_tree = 0.0
for i in range(0, steps.shape[0]):
    if recall[i] >= 0.7:
        max_prec_tree = max(max_prec_tree, prec[i])
print max_prec_tree
