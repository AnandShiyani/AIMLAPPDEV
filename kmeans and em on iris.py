import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")

iris = datasets.load_iris()
X = pd.DataFrame(iris.data)
X.columns = ['Sepal_length', 'Sepal_width', 'Petal_length', 'Petal_width']
y = pd.DataFrame(iris.target)
y.columns = ['Targets']

model = KMeans(n_clusters = 3)
model.fit(X)

plt.figure(figsize=(14,14))
colormap = np.array(['red','lime', 'black'])

plt.subplot(2,2,1)
plt.scatter(X.Petal_length, X.Petal_width, c = colormap[y.Targets], s = 40)
plt.title('Real clusters')#home shift end
plt.xlabel('Petal length')
plt.ylabel('Petal width')

plt.subplot(2,2,2)
plt.scatter(X.Petal_length, X.Petal_width, c = colormap[model.labels_], s = 40)
plt.title('K-means clustering')#home shift end
plt.xlabel('Petal length')
plt.ylabel('Petal width')


#EM
from sklearn import preprocessing
scaler = preprocessing.StandardScaler()
scaler.fit(X)
xsa = scaler.transform(X)
xs = pd.DataFrame(xsa, columns = X.columns)

from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components = 3)
gmm.fit(xs)
gmm_y = gmm.predict(xs)
plt.subplot(2,2,3)
plt.scatter(X.Petal_length, X.Petal_width, c = colormap[gmm_y], s = 40)
plt.title('GMM clustering')#home shift end
plt.xlabel('Petal length')
plt.ylabel('Petal width')
plt.subplot_tool()
plt.show()
print("Observation: the gmm using em algorithm based clustering matched the true labels more closely than the kmeans")
