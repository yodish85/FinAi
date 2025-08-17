import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import re
import os

# Load data
file_path = "/Users/admin/FinAi/train-val-data/test_balanced_data_20250811_170036.npy"
labels_path = "/Users/admin/FinAi/train-val-data/test_balanced_labels_20250811_170036.npy"
data = np.load(file_path)

# Extract prefix (first part before "_balanced_data")
prefix_match = re.match(r"([a-zA-Z0-9]+)_balanced_data", os.path.basename(file_path))
prefix = prefix_match.group(1) if prefix_match else "data"

# Extract timestamp
timestamp_match = re.search(r"(\d{8}_\d{6})", os.path.basename(file_path))
timestamp = timestamp_match.group(1) if timestamp_match else "no_timestamp"

# Reshape: each sample becomes a row
X = data.reshape(data.shape[0], -1)

# Apply K-means clustering
n_clusters = 3  # Change as needed
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(X)

print("Cluster labels:", labels)
print("Cluster centers shape:", kmeans.cluster_centers_.shape)

# Prepare output path for npy (same folder, with prefix + timestamp)
output_dir = os.path.dirname(file_path)
output_filename = f"{prefix}_cluster_labels_{timestamp}.npy"
output_path = os.path.join(output_dir, output_filename)

# Save labels as npy
np.save(output_path, labels)
print(f"Labels saved to {output_path}")

# Reduce to 2D for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# --- Load existing labels if available ---
if os.path.exists(labels_path):
    old_labels = np.load(labels_path)
    print("Loaded existing labels from file.")

    # Compare new vs old labels
    differences = labels != old_labels
    diff_count = np.sum(differences)
    total = len(labels)
    if diff_count == 0:
        print("✅ New labels are identical to existing labels.")
    else:
        print(f"⚠️ Labels differ: {diff_count} out of {total} ({diff_count/total:.2%} difference)")

    # --- Plot comparison ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # New labels plot
    for cluster_id in range(n_clusters):
        axes[0].scatter(
            X_pca[labels == cluster_id, 0],
            X_pca[labels == cluster_id, 1],
            label=f"Cluster {cluster_id}"
        )
    axes[0].set_title("New K-means Labels")
    axes[0].legend()
    axes[0].grid(True)

    # Old labels plot
    for cluster_id in range(n_clusters):
        axes[1].scatter(
            X_pca[old_labels == cluster_id, 0],
            X_pca[old_labels == cluster_id, 1],
            label=f"Cluster {cluster_id}"
        )
    axes[1].set_title("Previous Labels")
    axes[1].legend()
    axes[1].grid(True)

    # Differences plot
    axes[2].scatter(
        X_pca[~differences, 0],
        X_pca[~differences, 1],
        color="lightgray",
        label="Same"
    )
    axes[2].scatter(
        X_pca[differences, 0],
        X_pca[differences, 1],
        color="red",
        label="Different"
    )
    axes[2].set_title("Differences (Red = Changed)")
    axes[2].legend()
    axes[2].grid(True)

    plt.suptitle("Cluster Label Comparison", fontsize=16)
    plt.show()

else:
    print("No previous label file found. This is the first run.")


# Plot clusters
plt.figure(figsize=(8, 6))
for cluster_id in range(n_clusters):
    plt.scatter(
        X_pca[labels == cluster_id, 0],
        X_pca[labels == cluster_id, 1],
        label=f"Cluster {cluster_id}"
    )

plt.scatter(
    pca.transform(kmeans.cluster_centers_)[:, 0],
    pca.transform(kmeans.cluster_centers_)[:, 1],
    color="black", marker="X", s=200, label="Centers"
)

plt.title("K-means Clustering (PCA 2D projection)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()
plt.grid(True)
plt.show()
