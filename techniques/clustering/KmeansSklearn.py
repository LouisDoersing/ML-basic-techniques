import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

csv = pd.read_csv('wine.csv', skiprows=1, header=None)
print(csv.head())

#make columns numeric
X = csv.apply(pd.to_numeric, errors='coerce')

#Normalize data
scaler = StandardScaler().fit(X)
scaledX = scaler.transform(X)


#Fit K-Means with scaled x
kmeans = KMeans(n_clusters=3, n_init='auto').fit(scaledX)

#make predictions
predictionOne = [1, 12, 2, 11, 100, 3, 3, .28, 1, 5, 1, 2, 1, 1]
predictionTwo = [4, 14, 5, 10, 12000, 3, 3, .28, 1, 5, 1, 2, 10000, 1]

#reshape prediction in 2D Array
predictionOneReshape = np.array(predictionOne).reshape(1, -1)
predictionTwoReshape = np.array(predictionTwo).reshape(1, -1)

#normalize prediction
predictionOne_scaled = scaler.transform(predictionOneReshape)
predictionTwo_scaled = scaler.transform(predictionTwoReshape)

#predict with KMeans
predicted_cluster_one = kmeans.predict(predictionOne_scaled)
predicted_cluster_two = kmeans.predict(predictionTwo_scaled)


#Print Results
print("Prediction One cluster:", predicted_cluster_one)
print("Prediction Two cluster:", predicted_cluster_two)



