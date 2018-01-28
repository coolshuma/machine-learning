import pandas
from sklearn.svm import SVC
import numpy as np
from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

newsgroups = datasets.fetch_20newsgroups(
    subset='all',
    categories=['alt.atheism', 'sci.space']
)

vectorizer = TfidfVectorizer()
vectorizer.fit_transform(newsgroups.data)

grid = {'C': np.power(10.0, np.arange(-5, 6))}
cv = KFold(n_splits=5, shuffle=True, random_state=241)
clf = SVC(kernel='linear', random_state=241)
gs = GridSearchCV(clf, grid, scoring='accuracy', cv=cv)
gs.fit(vectorizer.transform(newsgroups.data), newsgroups.target)

print gs.best_params_

clf = SVC(kernel='linear', random_state=241, C=1.0)
clf.fit(vectorizer.transform(newsgroups.data), newsgroups.target)

print clf.coef_[0]

words = vectorizer.get_feature_names()
coeff = pandas.DataFrame(clf.coef_.data, clf.coef_.indices)
top_words = coeff[0].map(lambda w: abs(w)).sort_values(ascending=False).head(10).index.map(lambda i: words[i])
top_words.sort()
print top_words
