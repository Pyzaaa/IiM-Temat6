import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.manifold import SpectralEmbedding
from sklearn.neighbors import NearestNeighbors
from umap.umap_ import UMAP, smooth_knn_dist, find_ab_params


class self_UMAP:
    def __init__(self, n_neighbors=15, min_dist=0.1, n_epochs=200, learning_rate=0.01):
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate

    def fit_transform(self, X):
        # ETAP I
        # 1. Znajdowanie najbliższych sąsiadów w przestrzeni wysokowymiarowej
        nn = NearestNeighbors(n_neighbors=self.n_neighbors)
        nn.fit(X)

        # 2. Wyznaczanie odległosci
        distances, labels = nn.kneighbors(X) # distances - macierz d_ij, labels - labele sąsiadów (indeksy)

        # 3. Przeliczanie odległości na prawdopodobieństwo
        sigmas, rhos = smooth_knn_dist(distances, self.n_neighbors)
        graph = {}
        for i in range(X.shape[0]):
            for point_lbl, j in enumerate(labels[i]):
                if i == j:
                    continue
                d = distances[i, point_lbl]
                rho = rhos[i]
                sigma = sigmas[i]
                p_ij = np.exp(-(d - rho) / sigma) if d > rho else 1.0
                graph[(i, j)] = p_ij

        # 4. Łączenie par (p(ij), p(ji)) w jedno symetryczne prawdopodobieństwo
        symetric_graph = {}
        for (i, j), p_ij in graph.items():
            p_ji = graph.get((j, i), 0.0)
            p = 1 - (1 - p_ij) * (1 - p_ji)
            symetric_graph[(i, j)] = p
            symetric_graph[(j, i)] = p

        # ETAP II

        # PCA zamiast SpectralEmbedding do inicjalizacji
        # from sklearn.decomposition import PCA
        # X_embedded = PCA(n_components=2).fit_transform(X).copy()

        # 1. użycie spectral embedding do inicjalizacji punktów w przesztrzeni niskowymiarowej
        embedding = SpectralEmbedding(n_components=2, n_neighbors=self.n_neighbors)
        X_embedded = embedding.fit_transform(X)

        # 2. Dobieranie a i b na podstawie min_distance
        a, b = find_ab_params(spread=1.0, min_dist=self.min_dist)
        def similarity_embedded(distance):
            return 1.0 / (1.0 + a * (distance ** (2 * b)))
        

        # 3. Zdefiniowanie funkcji strat: Binary Cross-Entropy (gradient)
        def cross_entropy_gradient(i, j, X_embedded):
            distance = np.linalg.norm(X_embedded[i] - X_embedded[j])
            p_ij = symetric_graph.get((i, j), 0.0)
            q_ij = similarity_embedded(distance)

            # Gradient Binary Cross-Entropy
            grad = (X_embedded[i] - X_embedded[j]) * (p_ij - q_ij) * 2 * a * distance ** (2 * b) / (1 + a * distance ** (2 * b)) ** 2
            return grad

        # 4. Minimalizacja funkcji strat za pomocą Stochastic Gradient Descent
        graph_edges = list(symetric_graph.keys())
        for epoch in range(self.n_epochs):
            for (i, j) in graph_edges:
                gradient = cross_entropy_gradient(i, j, X_embedded)
                X_embedded[i] -= self.learning_rate * gradient
                X_embedded[j] += self.learning_rate * gradient

        return X_embedded


if __name__ == "__main__":
    # 1. Dane testowe
    X, y = make_blobs(n_samples=300, centers=4, n_features=10, random_state=42)

    # 2. Własny UMAP
    my_umap = self_UMAP(n_neighbors=15, min_dist=0.1, n_epochs=200, learning_rate=0.05)
    X_my = my_umap.fit_transform(X)

    # 3. Gotowy UMAP
    umap_model = UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    X_umap = umap_model.fit_transform(X)

    # 4. Wizualizacja
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(X_my[:, 0], X_my[:, 1], c=y, cmap='Spectral', s=10)
    plt.title("Twoja implementacja UMAP")

    plt.subplot(1, 2, 2)
    plt.scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap='Spectral', s=10)
    plt.title("Gotowy UMAP z biblioteki")

    plt.suptitle("Porównanie UMAP – własna vs gotowa implementacja", fontsize=14)
    plt.tight_layout()
    plt.show()
