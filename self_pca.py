from sklearn.datasets import make_swiss_roll
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


class SelfPCA():
    def __init__(self, n_components, return_eigen_vectors=False):
        self.n_components = n_components
        self.return_eigen_vectors = return_eigen_vectors

    def fit_transform(self, X):
        return None


if __name__ == "__main__":

    X_swiss, y_swiss = make_swiss_roll(n_samples=1000, noise=0.05)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    p = ax.scatter(X_swiss[:, 0], X_swiss[:, 1], X_swiss[:, 2], c=y_swiss, cmap=plt.cm.viridis)
    fig.colorbar(p, ax=ax)
    ax.set_title("Swiss Roll - 3D")
    plt.show()

    pca_swiss = PCA(n_components=2)
    X_pca_swiss = pca_swiss.fit_transform(X_swiss)
    plt.scatter(X_pca_swiss[:, 0], X_pca_swiss[:, 1], c=y_swiss, cmap=plt.cm.viridis)
    plt.title("Swiss Roll - PCA 2D")
    plt.colorbar()
    plt.show()

    self_pca_swiss = SelfPCA(n_components=2)
    X_self_pca_swiss = self_pca_swiss.fit_transform(X_swiss)
