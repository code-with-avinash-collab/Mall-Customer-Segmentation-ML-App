import pandas as pd
import joblib

from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import silhouette_score

# Load Dataset
data = pd.read_csv("E:\Mall_Customers.csv")

# Encode Gender
le = LabelEncoder()
data["Gender"] = le.fit_transform(data["Gender"])

# Save encoder
joblib.dump(le, "gender_encoder.pkl")

# Features for Clustering
X = data[["Annual Income (k$)", "Spending Score (1-100)"]]

# Train KMeans
kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

kmeans.fit(X)

# Save Model
joblib.dump(kmeans, "mall_customer_model.pkl")

# Cluster Labels
labels = kmeans.labels_

# Silhouette Score
score = silhouette_score(X, labels)

print("Training Completed")
print("Silhouette Score:", round(score,3))
print("Cluster Centers:")
print(kmeans.cluster_centers_)