import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll, fetch_openml
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas


X, t = make_swiss_roll(n_samples=1000, noise=0.05)
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
p = ax.scatter(X[:,0], X[:,1], X[:,2], c=t, cmap=plt.cm.viridis)
fig.colorbar(p, ax=ax)
ax.set_title("Swiss Roll - 3D")
plt.show()

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
plt.scatter(X_pca[:,0], X_pca[:,1], c=t, cmap=plt.cm.viridis)
plt.title("Swiss Roll - PCA 2D")
plt.colorbar()
plt.show()

tsne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=30)
X_tsne = tsne.fit_transform(X)
plt.scatter(X_tsne[:,0], X_tsne[:,1], c=t, cmap=plt.cm.viridis)
plt.title("Swiss Roll - t-SNE 2D")
plt.colorbar()
plt.show()

reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1)
X_umap = reducer.fit_transform(X)
plt.scatter(X_umap[:,0], X_umap[:,1], c=t, cmap=plt.cm.viridis)
plt.title("Swiss Roll - UMAP 2D")
plt.colorbar()
plt.show()

mnist = fetch_openml('mnist_784', version=1)
X_mnist, y_mnist = mnist['data'], mnist['target']
y_mnist = y_mnist.astype(int)
X_mnist = X_mnist[:10000]
y_mnist = y_mnist[:10000]

X_train, X_test, y_train, y_test = train_test_split(X_mnist, y_mnist, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
acc_no_reduction = accuracy_score(y_test, y_pred)
print("accuracy no reduction:", acc_no_reduction)

pca_mnist = PCA(n_components=50)
X_mnist_pca = pca_mnist.fit_transform(X_mnist)
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_mnist_pca, y_mnist, test_size=0.2, random_state=42)

clf_pca = RandomForestClassifier(n_estimators=100, random_state=42)
clf_pca.fit(X_train_pca, y_train_pca)
y_pred_pca = clf_pca.predict(X_test_pca)
acc_pca = accuracy_score(y_test_pca, y_pred_pca)
print("accuracy with PCA: ", acc_pca)