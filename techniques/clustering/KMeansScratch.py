import numpy as np
import matplotlib.pyplot as plt


class KMansClustering:

    def __init__(self,  k=3):
        self.k = k
        self.centroids = None



#Calculates euclidean Distance between data Point and every Centroid
    @staticmethod
    def EuclideanDistance(dataPoint, centroids):
        return np.sqrt(np.sum((dataPoint - centroids) **2, axis = 1))

    # stellt sicher, dass centroids nicht außerhalb der Achsen generiert werden
    def fit(self, x, maxIterations=200):
        self.centroids = np.random.uniform(np.amin(x, axis=0), np.amax(x, axis=0),
                                           size=(self.k, x.shape[1]))
        for _ in range(maxIterations):
            y = []  # array with distances

            for dataPoint in x:
                distance = KMansClustering.EuclideanDistance(dataPoint,
                                            self.centroids)  # returns list of diatances betweeen data point and everyy centroiud
                clusterNumber = np.argmin(distance)
                y.append(clusterNumber)

            y = np.array(y)
            # Safe cluster idices, has lists of lists
            clusterIndices = []
            for i in range(self.k):
                clusterIndices.append(np.argwhere(y == i))

            clusterCenters = []
            for i, indices in enumerate(clusterIndices):
                if len(indices) == 0:
                    clusterCenters.append(self.centroids[i])
                else:
                    clusterCenters.append(np.mean(x[indices], axis=0)[0])

            # When there is no centroids which has more change then 0.0001, then break
            if (np.max(self.centroids - np.array(clusterCenters)) < 0.0001):
                break
            else:
                self.centroids = np.array(clusterCenters)

        return y


randomPoints = np.random.randint(0, 100, (100, 2))
kmeans = KMansClustering(k=3)
labels = kmeans.fit(randomPoints)

plt.scatter(randomPoints[:, 0], randomPoints[:, 1], c=labels)
plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], c=range(len(kmeans.centroids)), marker = "*", s = 200)
plt.show()