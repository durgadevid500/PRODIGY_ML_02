

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


df = pd.read_csv(r"C:\Users\hp\Downloads\archive\Mall_Customers.csv")

print("\n========== DATASET ==========")
print(df.head())

print("\nDataset Shape:", df.shape)

print("\nDataset Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())



df = df.dropna()

print("\nAfter Cleaning:", df.shape)




print("\n========== DATA ANALYSIS ==========")

print("\nStatistical Summary:")
print(df.describe())

plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Gender")
plt.title("Customer Gender Distribution")
plt.savefig("gender_distribution.png")
plt.show()

plt.figure(figsize=(7, 4))
sns.histplot(df["Age"], bins=15, kde=True)
plt.title("Customer Age Distribution")
plt.savefig("age_distribution.png")
plt.show()
plt.figure(figsize=(7, 5))
sns.scatterplot(
    data=df,
    x="Annual Income (k$)",
    y="Spending Score (1-100)"
)
plt.title("Annual Income vs Spending Score")
plt.savefig("customer_scatter.png")
plt.show()


X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


inertia = []

for k in range(1, 11):

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    kmeans.fit(X_scaled)

    inertia.append(kmeans.inertia_)


plt.figure(figsize=(7, 5))
plt.plot(range(1, 11), inertia, marker="o")
plt.title("Elbow Method")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.xticks(range(1, 11))
plt.savefig("elbow_method.png")
plt.show()


kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(X_scaled)


score = silhouette_score(X_scaled, df["Cluster"])

print("\n========== MODEL EVALUATION ==========")
print("Silhouette Score:", round(score, 4))


plt.figure(figsize=(9, 6))

sns.scatterplot(
    data=df,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    hue="Cluster",
    palette="viridis",
    s=100
)

plt.title("Customer Segmentation using K-Means")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend(title="Customer Cluster")

plt.savefig("customer_clusters.png")
plt.show()


print("\n========== CLUSTER ANALYSIS ==========")

cluster_analysis = df.groupby("Cluster")[
    ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
].mean()

print(cluster_analysis)


df.to_csv("segmented_customers.csv", index=False)

print("\nCustomer Segmentation Completed Successfully!")

print("Results saved to segmented_customers.csv")
print("All graphs saved successfully.")
