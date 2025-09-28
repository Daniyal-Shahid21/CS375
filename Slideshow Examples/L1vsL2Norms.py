import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap
import numpy as np

# Generate 2D dataset
X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                           n_clusters_per_class=1, n_samples=200, random_state=42)

# Define classifiers
knn_l1 = KNeighborsClassifier(n_neighbors=5, metric='manhattan').fit(X, y)
knn_l2 = KNeighborsClassifier(n_neighbors=5, metric='euclidean').fit(X, y)

# Mesh grid
x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1
y_min, y_max = X[:,1].min() - 1, X[:,1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))

# Predictions
Z1 = knn_l1.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
Z2 = knn_l2.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(10,4))
cm = ListedColormap(['#FFAAAA', '#AAAAFF'])

for ax, Z, title in zip(axes, [Z1, Z2], ['L1 norm (Manhattan)', 'L2 norm (Euclidean)']):
    ax.contourf(xx, yy, Z, cmap=cm, alpha=0.6)
    ax.scatter(X[:,0], X[:,1], c=y, edgecolor='k', cmap=cm)
    ax.set_title(title)

plt.savefig("knn_boundaries.png")