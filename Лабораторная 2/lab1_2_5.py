import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

data = pd.read_csv('WholesaleCustomersData.csv')
X = data.values[:, :].astype(float)

pca = PCA(n_components=2)
X = pca.fit_transform(X)

kmeans_model = KMeans(n_clusters=5, init='k-means++', random_state=0)
kmeans_model.fit(X)

data['cluster'] = kmeans_model.labels_
pd.set_option('display.max_columns', None)
print(data.groupby('cluster').mean())

clusters = (0, 1, 2)
colors = ('r', 'g', 'b', 'y', 'c', 'm', 'pink')
for cluster, color in zip(clusters, colors):
    plt.scatter(X[data['cluster'] == cluster, 0],
                X[data['cluster'] == cluster, 1],
                s=50, c=color, label='Cluster '+str(cluster))
    
plt.scatter(kmeans_model.cluster_centers_[:, 0],
            kmeans_model.cluster_centers_[:, 1],
            s=100, c='black', marker='x', label='Centroid') # вывод центроидов

test_data = pd.read_csv('test.csv')
test_data = test_data.values[:, :].astype(float)  # Исправлено на срез всех столбцов
test_data = pca.transform(test_data)

plt.scatter(test_data[0, 0], test_data[0, 1],
            s=200, c='black', marker='o', label='Object')

print('Объект ближе к кластеру {0}.'.format(*kmeans_model.predict(test_data)))

plt.title('Cluster 5')
plt.xlabel('Principal component 1')
plt.ylabel('Principal component 2')
plt.legend()
plt.show()
