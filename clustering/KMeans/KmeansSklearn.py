import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
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
prediction_One = [1, 12, 2, 11, 100, 3, 3, .28, 1, 5, 1, 2, 1, 1]
prediction_Two = [4, 14, 5, 10, 12000, 3, 3, .28, 1, 5, 1, 2, 10000, 1]

#reshape prediction in 2D Array
prediction_One_Reshape = np.array(prediction_One).reshape(1, -1)
prediction_Two_Reshape = np.array(prediction_Two).reshape(1, -1)

#denormalize prediction
predictionOne_scaled = scaler.transform(prediction_One_Reshape)
predictionTwo_scaled = scaler.transform(prediction_Two_Reshape)

#predict with KMeans
predicted_cluster_one = kmeans.predict(predictionOne_scaled)
predicted_cluster_two = kmeans.predict(predictionTwo_scaled)


#Print Results
print("Prediction One cluster:", predicted_cluster_one)
print("Prediction Two cluster:", predicted_cluster_two)



#Visualization
# reduction from 14D to 2D with PCS for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(scaledX)

print(X_pca)

# create scatter Plot
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans.labels_, cmap='viridis', alpha=0.6)
plt.xlabel('X')
plt.ylabel('Y')
plt.show()


