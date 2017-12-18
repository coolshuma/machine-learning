# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

from sklearn import pipeline
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score

def write_to_csv(filename, columns, pred):
    pred = np.array(pred)
    ids = np.array(test_ids)
    st = np.column_stack((ids, pred))
    from csv import writer
    with open(filename, 'w') as file:
        fieldnames = columns
        wr = writer(file, fieldnames)
        wr.writerow(fieldnames)
        for i in st:
            wr.writerow(i)

train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
y = train_data['Survived']
train_data = train_data.drop(labels='Survived', axis=1)
test_ids = test_data['PassengerId']

def transform_features(data):
    data.drop(labels=['PassengerId', 'Ticket', 'Name', 'Cabin'], 
                             axis=1, inplace=True)
    data.fillna(value=0, inplace=True)
    data['Embarked'] = list(map(lambda x: ord(x) if x in {'S', 'Q', 'C'} else 0, data['Embarked']))
    data['Sex'] = list(map(lambda x: 1 if x == 'male' else 0, data['Sex']))
    
transform_features(train_data)
transform_features(test_data)

print(train_data['Sex'])

cat_columns = ['Sex', 'Embarked']
real_features = ['Age', 'SibSp', 'Parch', 'Fare', 'Pclass'] 
real_data_indices = np.array([(column in real_features) for column in train_data.columns], dtype = bool)
cat_data_indices = np.array([(column in cat_columns) for column in train_data.columns], dtype = bool)

clf = XGBClassifier(max_depth=5, n_estimators=100)

estimator = pipeline.Pipeline(steps = [       
    ('feature_preprocessing', pipeline.FeatureUnion(transformer_list = [        
            #real
            ('numeric_variables_processing', pipeline.Pipeline(steps = [
                ('selecting', preprocessing.FunctionTransformer(lambda data: data[:, real_data_indices])),
                ('scaling', preprocessing.StandardScaler(with_mean = 0))            
                        ])),
        
            #categorical
            ('categorical_variables_processing', pipeline.Pipeline(steps = [
                ('selecting', preprocessing.FunctionTransformer(lambda data: data[:, cat_data_indices])),
                ('hot_encoding', preprocessing.OneHotEncoder(handle_unknown = 'ignore'))            
                        ])),
        ])),
    ('model_fitting', clf)
    ]
)

estimator.fit(train_data, y)
pred = estimator.predict(test_data)
write_to_csv('check.csv', ['PassengerId', 'Survived'], pred)