# K-MEANS CLUSTERING - MALL CUSTOMER SEGMENTATION

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# ✅ Load dataset from local file (NO ERROR)
df = pd.read_csv("Mall_Customers.csv")

print("Dataset Preview:")
print(df.head().to_string())

# ✅ Select features
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# ✅ Elbow Method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title("Elbow Method")
plt.xlabel("Clusters")
plt.ylabel("WCSS")
plt.show()

# ✅ K-Means clustering
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
y_kmeans = kmeans.fit_predict(X)

# ✅ Visualization
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y_kmeans)
plt.scatter(kmeans.cluster_centers_[:, 0],
            kmeans.cluster_centers_[:, 1],
            s=200, marker='X')

plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("Customer Segments")
plt.show()