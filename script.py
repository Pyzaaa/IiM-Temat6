import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll, fetch_openml, make_blobs
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, accuracy_score, classification_report
import pandas as pd
from sklearn.preprocessing import StandardScaler
import seaborn as sns

scaler = StandardScaler()

#    WIZUALIZACJE

X_swiss, y_swiss = make_swiss_roll(n_samples=1000, noise=0.05)
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
p = ax.scatter(X_swiss[:,0], X_swiss[:,1], X_swiss[:,2], c=y_swiss, cmap=plt.cm.viridis)
fig.colorbar(p, ax=ax)
ax.set_title("Swiss Roll - 3D")
plt.show()

pca_swiss = PCA(n_components=2)
X_pca_swiss = pca_swiss.fit_transform(X_swiss)
plt.scatter(X_pca_swiss[:,0], X_pca_swiss[:,1], c=y_swiss, cmap=plt.cm.viridis)
plt.title("Swiss Roll - PCA 2D")
plt.colorbar()
plt.show()

tsne_swiss = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=30, random_state=42)
X_tsne_swiss = tsne_swiss.fit_transform(X_swiss)
plt.scatter(X_tsne_swiss[:,0], X_tsne_swiss[:,1], c=y_swiss, cmap=plt.cm.viridis)
plt.title("Swiss Roll - t-SNE 2D")
plt.colorbar()
plt.show()

umap_swiss = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
X_umap_swiss = umap_swiss.fit_transform(X_swiss)
plt.scatter(X_umap_swiss[:,0], X_umap_swiss[:,1], c=y_swiss, cmap=plt.cm.viridis)
plt.title("Swiss Roll - UMAP 2D")
plt.colorbar()
plt.show()

X_blobs, y_blobs = make_blobs(n_samples=1000, centers=4, n_features=3, random_state=0)
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
p = ax.scatter(X_blobs[:,0], X_blobs[:,1], X_blobs[:,2], c=y_blobs, cmap=plt.cm.viridis)
fig.colorbar(p, ax=ax)
ax.set_title("Blobs - 3D")
plt.show()

pca_blobs = PCA(n_components=2)
X_pca_blobs = pca_blobs.fit_transform(X_blobs)
plt.scatter(X_pca_blobs[:,0], X_pca_blobs[:,1], c=y_blobs, cmap=plt.cm.viridis)
plt.title("Blobs - PCA 2D")
plt.colorbar()
plt.show()

tsne_blobs = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=30, random_state=42)
X_tsne_blobs = tsne_blobs.fit_transform(X_blobs)
plt.scatter(X_tsne_blobs[:,0], X_tsne_blobs[:,1], c=y_blobs, cmap=plt.cm.viridis)
plt.title("Blobs - t-SNE 2D")
plt.colorbar()
plt.show()

umap_blobs = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
X_umap_blobs = umap_blobs.fit_transform(X_blobs)
plt.scatter(X_umap_blobs[:,0], X_umap_blobs[:,1], c=y_blobs, cmap=plt.cm.viridis)
plt.title("Blobs - UMAP 2D")
plt.colorbar()
plt.show()

#    MNIST

mnist = fetch_openml('mnist_784', version=1)
X_mnist, y_mnist = mnist['data'], mnist['target']
y_mnist = y_mnist.astype(int)
X_mnist = X_mnist[:10000]
y_mnist = y_mnist[:10000]

X_train, X_test, y_train, y_test = train_test_split(X_mnist, y_mnist, test_size=0.2, random_state=42)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf = RandomForestClassifier(n_estimators=100, random_state=42)

clf.fit(X_train_scaled, y_train)
y_pred = clf.predict(X_test_scaled)
print("Accuracy no reduction:", accuracy_score(y_test, y_pred))
print("Precision no reduction:", precision_score(y_test, y_pred, average='macro'))
print("Recall no reduction:", recall_score(y_test, y_pred, average='macro'))
print("F1-score no reduction:", f1_score(y_test, y_pred, average='macro'))
print("Confusion Matrix no reduction:\n", confusion_matrix(y_test, y_pred))

pca_mnist = PCA(n_components=50)
X_mnist_pca = pca_mnist.fit_transform(X_mnist)
plt.figure(figsize=(8, 6))
plt.scatter(X_mnist_pca[:, 0], X_mnist_pca[:, 1], c=y_mnist, cmap='tab10', s=10)
plt.title("MNIST - PCA 2D")
plt.colorbar(label='Cyfra')
plt.tight_layout()
plt.show()

X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_mnist_pca, y_mnist, test_size=0.2, random_state=42)
X_train_pca_scaled = scaler.fit_transform(X_train_pca)
X_test_pca_scaled = scaler.transform(X_test_pca)

clf.fit(X_train_pca_scaled, y_train_pca)
y_pred_pca = clf.predict(X_test_pca_scaled)
print("Accuracy with PCA:", accuracy_score(y_test_pca, y_pred_pca))
print("Precision with PCA:", precision_score(y_test_pca, y_pred_pca, average='macro'))
print("Recall with PCA:", recall_score(y_test_pca, y_pred_pca, average='macro'))
print("F1-score with PCA:", f1_score(y_test_pca, y_pred_pca, average='macro'))
print("Confusion Matrix with PCA:\n", confusion_matrix(y_test_pca, y_pred_pca))

tsne_mnist = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=30, random_state=42)
X_mnist_tsne = tsne_mnist.fit_transform(X_mnist)
plt.figure(figsize=(8, 6))
plt.scatter(X_mnist_tsne[:, 0], X_mnist_tsne[:, 1], c=y_mnist, cmap='tab10', s=10)
plt.title("MNIST - t-SNE 2D")
plt.colorbar(label='Cyfra')
plt.tight_layout()
plt.show()

X_train_tsne, X_test_tsne, y_train_tsne, y_test_tsne = train_test_split(X_mnist_tsne, y_mnist, test_size=0.2, random_state=42)
X_train_tsne_scaled = scaler.fit_transform(X_train_tsne)
X_test_tsne_scaled = scaler.transform(X_test_tsne)

clf.fit(X_train_tsne_scaled, y_train_tsne)
y_pred_tsne = clf.predict(X_test_tsne_scaled)
print("Accuracy with t-SNE:", accuracy_score(y_test_tsne, y_pred_tsne))
print("Precision with t-SNE:", precision_score(y_test_tsne, y_pred_tsne, average='macro'))
print("Recall with t-SNE:", recall_score(y_test_tsne, y_pred_tsne, average='macro'))
print("F1-score with t-SNE:", f1_score(y_test_tsne, y_pred_tsne, average='macro'))
print("Confusion Matrix with t-SNE:\n", confusion_matrix(y_test_tsne, y_pred_tsne))

umap_mnist = umap.UMAP(n_components=50, n_neighbors=15, min_dist=0.1, random_state=42)
X_mnist_umap = umap_mnist.fit_transform(X_mnist)
plt.figure(figsize=(8, 6))
plt.scatter(X_mnist_umap[:, 0], X_mnist_umap[:, 1], c=y_mnist, cmap='tab10', s=10)
plt.title("MNIST - UMAP 2D")
plt.colorbar(label='Cyfra')
plt.tight_layout()
plt.show()

X_train_umap, X_test_umap, y_train_umap, y_test_umap = train_test_split(X_mnist_umap, y_mnist, test_size=0.2, random_state=42)
X_train_umap_scaled = scaler.fit_transform(X_train_umap)
X_test_umap_scaled = scaler.transform(X_test_umap)

clf.fit(X_train_umap_scaled, y_train_umap)
y_pred_umap = clf.predict(X_test_umap_scaled)
print("Accuracy with UMAP:", accuracy_score(y_test_umap, y_pred_umap))
print("Precision with UMAP:", precision_score(y_test_umap, y_pred_umap, average='macro'))
print("Recall with UMAP:", recall_score(y_test_umap, y_pred_umap, average='macro'))
print("F1-score with UMAP:", f1_score(y_test_umap, y_pred_umap, average='macro'))
print("Confusion Matrix with UMAP:\n", confusion_matrix(y_test_umap, y_pred_umap))

results = {
    'Metoda': ['Brak redukcji', 'PCA', 't-SNE', 'UMAP'],
    'Accuracy': [
        accuracy_score(y_test, y_pred),
        accuracy_score(y_test_pca, y_pred_pca),
        accuracy_score(y_test_tsne, y_pred_tsne),
        accuracy_score(y_test_umap, y_pred_umap)
    ],
    'Precision': [
        precision_score(y_test, y_pred, average='macro'),
        precision_score(y_test_pca, y_pred_pca, average='macro'),
        precision_score(y_test_tsne, y_pred_tsne, average='macro'),
        precision_score(y_test_umap, y_pred_umap, average='macro')
    ],
    'Recall': [
        recall_score(y_test, y_pred, average='macro'),
        recall_score(y_test_pca, y_pred_pca, average='macro'),
        recall_score(y_test_tsne, y_pred_tsne, average='macro'),
        recall_score(y_test_umap, y_pred_umap, average='macro')
    ],
    'F1-score': [
        f1_score(y_test, y_pred, average='macro'),
        f1_score(y_test_pca, y_pred_pca, average='macro'),
        f1_score(y_test_tsne, y_pred_tsne, average='macro'),
        f1_score(y_test_umap, y_pred_umap, average='macro')
    ]
}

df_results = pd.DataFrame(results)
df_results.set_index('Metoda', inplace=True)

plt.figure(figsize=(8, 5))
sns.heatmap(df_results, annot=True, cmap='YlGnBu', fmt=".3f")
plt.title("Porównanie metryk klasyfikacji dla różnych metod redukcji")
plt.tight_layout()
plt.show()
