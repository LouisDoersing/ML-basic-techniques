import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


train_data = pd.read_csv('LogisticRegression/files/train.csv', header=0, delimiter=',')
test_data = pd.read_csv('LogisticRegression/files/test.csv')
submission = pd.read_csv('LogisticRegression/files/gender_submission.csv')
print(train_data.head())
print(test_data.head())
#process the data
#target -> survided
target  = train_data['Survived']
print(target.head())

# input: Pclass, sex, age, SibSp
input = train_data[['Pclass', 'Sex', 'Age', 'SibSp']]



#transform male with 0 and female with 1
input.loc[:, 'Sex'] = input['Sex'].replace({'male': 0.0, 'female': 1.0})
#to numeric:
input = input.apply(pd.to_numeric, errors='coerce')




