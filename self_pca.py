from sklearn.datasets import make_swiss_roll, make_blobs, fetch_openml
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np


class SelfPCA:
    def __init__(self, n_components, return_eigenvectors=False):
        self.n_components = n_components
        self.return_eigenvectors = return_eigenvectors

    def fit_transform(self, x):
        x_meaned = x - np.mean(x, axis=0)
        cov_matrix = np.cov(x_meaned, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, sorted_indices]
        eigenvalues = eigenvalues[sorted_indices]
        principal_components = eigenvectors[:, :self.n_components]
        x_reduced = np.dot(x_meaned, principal_components)
        return (x_reduced, eigenvectors, eigenvalues) if self.return_eigenvectors else x_reduced


if __name__ == "__main__":

    xy_swiss = make_swiss_roll(n_samples=1000, noise=0.05, random_state=None)
    xy_blobs = make_blobs(n_samples=1000, centers=4, n_features=3, random_state=0)

    n_components = 2

    for x, y in (xy_swiss, xy_blobs):
        pca = PCA(n_components=n_components)
        x_pca = pca.fit_transform(x)

        self_pca = SelfPCA(n_components=n_components, return_eigenvectors=True)
        x_self_pca, e_vectors, e_values = self_pca.fit_transform(x)

        fig = plt.figure(figsize=(18, 5))

        ax1 = fig.add_subplot(1, 3, 1, projection='3d')
        p1 = ax1.scatter(x[:, 0], x[:, 1], x[:, 2], c=y, cmap=plt.cm.viridis)
        ax1.set_title("3D")
        fig.colorbar(p1, ax=ax1, shrink=0.5)

        origin = np.mean(x, axis=0)
        scale = 5
        for vec in e_vectors.T[:n_components]:
            ax1.quiver(*origin, *(vec * scale), color='red', linewidth=2, arrow_length_ratio=0.3)
        for vec in e_vectors.T[n_components:]:
            ax1.quiver(*origin, *(vec * scale), color='blue', linewidth=2, arrow_length_ratio=0.3)

        ax2 = fig.add_subplot(1, 3, 2)
        p2 = ax2.scatter(x_pca[:, 0], x_pca[:, 1], c=y, cmap=plt.cm.viridis)
        ax2.set_title("PCA 2D")
        fig.colorbar(p2, ax=ax2)

        ax3 = fig.add_subplot(1, 3, 3)
        p3 = ax3.scatter(x_self_pca[:, 0], x_self_pca[:, 1], c=y, cmap=plt.cm.viridis)
        ax3.set_title("self PCA 2D")
        fig.colorbar(p3, ax=ax3)

        plt.tight_layout()
        plt.show()

    mnist = fetch_openml('mnist_784', version=1)
    X_mnist, y_mnist = mnist['data'], mnist['target']
    y_mnist = y_mnist.astype(int)
    X_mnist = X_mnist[:10000]
    y_mnist = y_mnist[:10000]

    pca_mnist = PCA(n_components=50)
    X_mnist_pca = pca_mnist.fit_transform(X_mnist)

    self_pca_mnist = SelfPCA(n_components=50)
    X_mnist_self_pca = self_pca_mnist.fit_transform(X_mnist)

    fig = plt.figure(figsize=(12, 5))

    ax2 = fig.add_subplot(1, 2, 1)
    p2 = ax2.scatter(X_mnist_pca[:, 0], X_mnist_pca[:, 1], c=y_mnist, cmap='tab10', s=10)
    ax2.set_title("PCA 2D")
    fig.colorbar(p2, ax=ax2, label='Cyfra')

    ax3 = fig.add_subplot(1, 2, 2)
    p3 = ax3.scatter(X_mnist_self_pca[:, 0], X_mnist_self_pca[:, 1], c=y_mnist, cmap='tab10', s=10)
    ax3.set_title("self PCA 2D")
    fig.colorbar(p3, ax=ax3, label='Cyfra')

    plt.tight_layout()
    plt.show()


