import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer

speed_dating_data = pd.read_csv('speed_dating_data_mod.csv')
print(speed_dating_data.head())


#create X (attributes) and Y (target)
#target is dec, which means if the person wants to date the other one again
#1 = yes, 0 = no
X = speed_dating_data.drop('dec', axis=1)
Y = speed_dating_data['dec']

# check if there are nan-Values in X
print(X.isnull().sum())
#fill nan - Values with mean from columns.
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X = imputer.fit_transform(X)

#Split in test and training (80% Training, 20% Test)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

#Create classifier
classifier = DecisionTreeClassifier()
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)
print(y_pred)
#the Models accuracy is around 0.73. This is better than a random estimate. One reason for the lower accuracy could be the many nan values
print(accuracy_score(y_test, y_pred))


#Create people with fictitious ratings (traits) and have the model check whether they would be
#dated again or not
personOne = [1,0,27,5,4,3,2,6.8,9.0,7.0,7.0,7.0,5.0,8,2]
personTwo = [1,0,20,2,6,3,2,7.8,8.0,7.5,8.0,5.0,6.0,6,2]
personThree = [1,0,30,2,4,2,2,5.0,5.0,5.0,8.0,3.0,7.0,6,2]
#create predictions
predictions = np.array([
    personOne,
    personTwo,
    personThree
])
print(classifier.predict(predictions))





