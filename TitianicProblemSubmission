# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train_data= pd.read_csv("../input/titanic/train.csv")

test_data=pd.read_csv("../input/titanic/test.csv")
men=train_data.loc[train_data.Sex=="male"]["Survived"]
men.sum()/len(men)
train_data.dropna((axis=1))
from sklearn.ensemble import RandomForestClassifier
y = train_data["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch","Fare"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

features2=["Pclass","Sex","SibSp","Parch","Fare"]

X1 = pd.get_dummies(train_data[features2])
X1_test = pd.get_dummies(test_data[features2])
model2 = RandomForestClassifier(n_estimators=10000, max_depth=4, random_state=2)
model.fit(X1, y)
predictions2 = model.predict(X1_test)
output2 = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions2})
output2.to_csv('my_submission.csv', index=False)
