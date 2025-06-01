import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load features and labels
X = np.load("/Users/admin/Desktop/financial_ai_model/train_20250510_211806_data.npy")  # features
y = np.load("/Users/admin/Desktop/financial_ai_model/train_20250510_211806_labels.npy")  # labels (optional)
print("Original shape:", X.shape)

# Reduce temporal dimension: take mean across time for each feature
X_reduced = X.mean(axis=1)  # shape: (66286, 41)
print("Reduced shape (mean over time):", X_reduced.shape)

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_reduced)

# Run PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Plot explained variance
plt.figure(figsize=(10, 5))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance by PCA Components')
plt.grid(True)
plt.tight_layout()
plt.show()

# Analyze contributions
n_components = 10  # look at top components
component_weights = np.abs(pca.components_[:n_components])  # shape: (10, 41)
feature_importance = component_weights.sum(axis=0)  # shape: (41,)

# Rank features
num_to_drop = int(0.2 * len(feature_importance))  # drop bottom 20%
drop_indices = np.argsort(feature_importance)[:num_to_drop]

print(f"\nNumber of features: {len(feature_importance)}")
print(f"Suggest dropping {num_to_drop} least important features (indices shown):")
print(drop_indices)