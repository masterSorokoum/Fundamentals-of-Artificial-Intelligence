import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

data = pd.read_csv('WholesaleCustomersData.csv')
X = data.values[:, :].astype(float) 

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)  # Результат преобразования

wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=0)
    kmeans.fit(X_pca)
    wcss.append(kmeans.inertia_)  # WCSS для текущего k

plt.plot(range(1, 11), wcss)
plt.title('Метод "локтя"')
plt.xlabel('Число кластеров')
plt.ylabel('WCSS')
plt.xticks(range(1, 11))
plt.show()

k = 3
kmeans = KMeans(n_clusters=k, init='k-means++', random_state=0)
clusters = kmeans.fit_predict(X_pca)

data['cluster'] = clusters

print(data.groupby('cluster').mean())  # Средние значения признаков по кластерам

colors = ['red', 'green', 'blue']  # Цвета для кластеров
for i in range(k):
    plt.scatter(X_pca[clusters == i, 0], X_pca[clusters == i, 1], 
                c=colors[i], label=f'Кластер {i}')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
           s=100, marker='X', c='black', label='Центроиды')
plt.title('Кластеризация клиентов (k=3)')
plt.xlabel('Главная компонента 1')
plt.ylabel('Главная компонента 2')
plt.legend()
plt.show()
