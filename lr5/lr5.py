import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
import numpy as np

# 1.  Відкрити та зчитати наданий файл з даними. 
file_name = 'dataset2_l4.txt' 
df = pd.read_csv(file_name) 

# 2.  Визначити та вивести кількість записів. 
num = df.shape
num_rows, num_columns = num
print(f"Кількість записів: {num_rows}")
print(f"Кількість полів: {num_columns}")

# 3.  Видалити атрибут Class. 
df = df.drop(['Class'], axis=1) 

# 4.  Вивести атрибути, що залишилися. 
print(df.columns)


# # 5. 1) elbow method
# wcss = []
# for i in range(1, 11):
#     kmeans = KMeans(n_clusters=i, init='random', random_state=42)
#     kmeans.fit(df)
#     wcss.append(kmeans.inertia_)

# plt.plot(range(1, 11), wcss)
# plt.title('Метод ліктя')
# plt.xlabel('Кількість кластерів')
# plt.ylabel('WCSS')
# plt.show()

# # 5. 2) average silhouette method
# silhouette_scores = []
# for i in range(2, 11):
#     kmeans = KMeans(n_clusters=i, init='random', random_state=42)
#     kmeans.fit(df)
#     silhouette_scores.append(silhouette_score(df, kmeans.labels_))

# plt.plot(range(2, 11), silhouette_scores)
# plt.title('Середня силуетна оцінка')
# plt.xlabel('Кількість кластерів')
# plt.ylabel('Silhouette Score')
# plt.show()


# # 5. 3)
# ps_scores = []

# for k in range(2, 11):
#     X1, X2 = train_test_split(df, test_size=0.5, random_state=42)
    
#     kmeans1 = KMeans(n_clusters=k, init='random', random_state=42).fit(X1)
#     kmeans2 = KMeans(n_clusters=k, init='random', random_state=42).fit(X2)
    
#     labels1 = kmeans1.predict(df)
#     labels2 = kmeans2.predict(df)
    
#     pairs1 = set()
#     for i in range(len(labels1)):
#         for j in range(i + 1, len(labels1)):
#             if labels1[i] == labels1[j]:
#                 pairs1.add((i, j))
    
#     common_pairs = sum(1 for i, j in pairs1 if labels2[i] == labels2[j])
    
#     ps_scores.append(common_pairs / len(pairs1))
    

# plt.plot(range(2, 11), ps_scores)
# plt.title('Прогностична сила')
# plt.xlabel('Кількість кластерів')
# plt.ylabel('Prediction Strength')
# plt.show()



# 6.
optimal_clusters = 7

kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', n_init=10, max_iter=300, random_state=42)
kmeans.fit(df)

print("Координати центрів кластерів KMeans:")
print(kmeans.cluster_centers_)


# 7.
agg_clustering = AgglomerativeClustering(n_clusters=optimal_clusters)
agg_clustering.fit(df)
agg_centers = np.array([df[agg_clustering.labels_ == i].mean(axis=0) for i in range(optimal_clusters)])

print("Координати центрів кластерів Agglomerative Clustering:")
print(agg_centers)

# 8.
plt.figure(figsize=(12, 6))

# Візуалізація KMeans
plt.subplot(1, 2, 1)
plt.scatter(df.iloc[:, 5], df.iloc[:, 6], c=kmeans.labels_, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 5], kmeans.cluster_centers_[:, 6], s=30, c='red')
plt.title('KMeans Clustering')

# Візуалізація Agglomerative Clustering
plt.subplot(1, 2, 2)
plt.scatter(df.iloc[:, 5], df.iloc[:, 6], c=agg_clustering.labels_, cmap='viridis')
plt.scatter(agg_centers[:, 5], agg_centers[:, 6], s=30, c='red')
plt.title('Agglomerative Clustering')

plt.show()
