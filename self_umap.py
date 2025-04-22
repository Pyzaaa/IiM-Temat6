import numpy as np
from sklearn.neighbors import NearestNeighbors
from umap.umap_ import smooth_knn_dist, find_ab_params
from sklearn.datasets import make_blobs

class self_UMAP:
    def __init__(self, n_neighbors=15, min_dist=0.1, n_epochs=200, learnig_rate = 0.01):
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
        graph = {}
        sigmas, rhos = smooth_knn_dist(dists, k)
        for i in range(X.shape[0]):
            for point_lbl, j in enumerate(labels[i]):
                if i == j:
                    continue

                d = distances[i, point_lbl]
                rho = rhos[i]
                sigma = sigmas[i]

                if d > rho:
                    p_ij = np.exp(-(d - rho) / sigma)
                else:
                    p_ij = 1.0

                graph[(i, j)] = p_ij

        # 4. Łączenie par (p(ij), p(ji)) w jedno symetryczne prawdopodobieństwo
        symetric_graph = []
        for (i, j), p_ij in graph.items():
            p_ji = graph.get((j,i),0.0)
            p = 1 - (1 - p_ij) * (1 - p_ji)
            symetric_graph[(i, j)] = p

        # ETAP II

        # 1. użycie spectral embedding do inicjalizacji punktów w przesztrzni niskowymiarowej
        embedding = SpectralEmbedding(n_components=2, n_neighbors=self.n_neighbors)
        X_embedded = embedding.fit_transform(X)

        # 2. Dobieranie a i b na podstawie min_distance
        a, b = find_ab_parmas(spread=1.0, min_dist=0.1)
        def similarity_embedded(distance):
            return 1.0 / (1.0 + a * (distance ** (2*b)))

        # 3. Zdefiniowanie funkcji strat: Binary Cross-Entropy (gradient)
        def cross_entropy_gradient(i, j):
            distance = np.linalg.norm(X_embedded[i] - X_embedded[j])
            q_ij = similarity_embedded(distance)

            gradient = ((2*b*a)/((1+(a*(distance**(2*b))))**2)) * (distance) * 
            # TO DO: poprawić tą funkcję

        # 4. Minimalizacja funkcji strat za pomocą Stochastic Gradient Descent
        graph_edges = list(symetric_graph.keys())
        for epoch in range(self.n_epochs):
            for (i, j) in graph_edges:
                gradient = cross_entropy_gradient(i, j)
                X_embedded[i] -= self.learning_rate * gradient
                X_embedded[j] += self.learning_rate * gradient

        return X_embedded

if __name__ == "__main__":
    # X_blobs, y_blobs = make_blobs(n_samples=1000, centers=4, n_features=3, random_state=0)
    pass