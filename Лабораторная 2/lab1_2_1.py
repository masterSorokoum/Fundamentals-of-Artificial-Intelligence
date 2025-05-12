import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

data = pd.read_csv('WholesaleCustomersData.csv')
X = data.values[:, :].astype(float)

pca = PCA(n_components=2)
X = pca.fit_transform(X)
wcss = []

for k in range(1, 11):
    kmeans_model = KMeans(n_clusters=k, init='k-means++', random_state=0)
    kmeans_model.fit(X)
    wcss.append(kmeans_model.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('K-Means Elbow Metod')
plt.xlabel('Number of clusters')
plt.xticks([i for i in range(11)])
plt.ylabel('WCSS')
plt.show()

