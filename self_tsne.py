import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_s_curve, make_swiss_roll
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import trustworthiness
import time


class self_TSNE:
    def __init__(self, n_components=2, perplexity=30.0, max_iter=500, learning_rate=200.0, random_state=None):
        self.n_components = n_components
        self.perplexity = perplexity
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.random_state = random_state

    def _compute_pairwise_affinities(self, X, tol=1e-5):
        """Obliczanie podobieństw P_ij w przestrzeni wysokowymiarowej."""
        n = X.shape[0]
        D = np.square(np.linalg.norm(X[:, np.newaxis] - X[np.newaxis, :], axis=-1))  # kwadraty odległości
        P = np.zeros((n, n))

        # Dla każdego punktu szukamy sigmy takiej, by uzyskać zadaną perplexity
        for i in range(n):
            beta = 1.0
            betamin = -np.inf
            betamax = np.inf
            H_target = np.log2(self.perplexity)

            Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]
            H, thisP = self._Hbeta(Di, beta)

            # Szukanie optymalnego beta (odpowiednika sigma)
            tries = 0
            while np.abs(H - H_target) > tol and tries < 50:
                if H > H_target:
                    betamin = beta
                    beta = beta * 2 if betamax == np.inf else (beta + betamax) / 2
                else:
                    betamax = beta
                    beta = beta / 2 if betamin == -np.inf else (beta + betamin) / 2
                H, thisP = self._Hbeta(Di, beta)
                tries += 1

            P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP

        P = (P + P.T) / (2 * n)  # symetryzacja i normalizacja
        return P

    def _Hbeta(self, D, beta):
        P = np.exp(-D * beta)
        sumP = np.sum(P)
        H = np.log2(sumP) + beta * np.sum(D * P) / sumP
        return H, P / sumP

    def fit_transform(self, X):
        n, d = X.shape
        P = self._compute_pairwise_affinities(X)
        P *= 4.0  # Early exaggeration

        early_iterations = self.max_iter // 4

        # Inicjalizacja embeddingu (np. losowo)
        if self.random_state is not None:
            np.random.seed(self.random_state)

        Y = np.random.randn(n, self.n_components) * 1e-4

        Y_inc = np.zeros_like(Y)

        for iter in range(self.max_iter):
            if iter == early_iterations:
                P /= 4.0

            momentum = 0.5 if iter < early_iterations else 0.8

            # Obliczanie podobieństw Q_ij w przestrzeni niskowymiarowej
            sum_Y = np.sum(np.square(Y), 1)
            num = 1 / (1 + np.add.outer(sum_Y, sum_Y) - 2 * np.dot(Y, Y.T))
            np.fill_diagonal(num, 0)
            Q = num / np.sum(num)
            Q = np.maximum(Q, 1e-12)

            PQ = P - Q
            grad = np.zeros_like(Y)
            for i in range(n):
                grad[i] = 4 * np.sum((PQ[:, i][:, np.newaxis] * num[:, i][:, np.newaxis]) * (Y[i] - Y), axis=0)

            Y_inc = momentum * Y_inc - self.learning_rate * grad
            Y += Y_inc

        return Y


def knn_preservation(X_high, X_low, k=10):
    knn_high = NearestNeighbors(n_neighbors=k).fit(X_high)
    knn_low = NearestNeighbors(n_neighbors=k).fit(X_low)

    _, neigh_high = knn_high.kneighbors(X_high)
    _, neigh_low = knn_low.kneighbors(X_low)

    overlap = [len(set(neigh_high[i]) & set(neigh_low[i])) / k for i in range(X_high.shape[0])]
    return np.mean(overlap)


def test_and_plot(X, y, title_prefix, tsne_params):
    # Upewnij się, że X jest numpy array
    X = np.array(X)

    my_tsne = self_TSNE(**tsne_params)

    start = time.time()
    X_my = my_tsne.fit_transform(X)
    print(f"Własna t-SNE: {time.time() - start:.2f}s")

    tsne = TSNE(n_components=2, perplexity=tsne_params['perplexity'], learning_rate=tsne_params['learning_rate'],
                max_iter=tsne_params['max_iter'], init='random', random_state= tsne_params['random_state'])

    start = time.time()
    X_sklearn = tsne.fit_transform(X)
    print(f"Sklearn t-SNE: {time.time() - start:.2f}s")

    score_my = knn_preservation(X, X_my)
    score_sklearn = knn_preservation(X, X_sklearn)

    print(f"KNN-preservation (Własna):  {score_my:.3f}")
    print(f"KNN-preservation (Sklearn): {score_sklearn:.3f}")

    trust_my = trustworthiness(X, X_my, n_neighbors=10)
    trust_sklearn = trustworthiness(X, X_sklearn, n_neighbors=10)

    print(f"Trustworthiness (Własna):  {trust_my:.3f}")
    print(f"Trustworthiness (Sklearn): {trust_sklearn:.3f}")

    # Plot
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(X_my[:, 0], X_my[:, 1], c=y, cmap='Spectral', s=5)
    plt.title(f"{title_prefix} – Własna implementacja")

    plt.subplot(1, 2, 2)
    plt.scatter(X_sklearn[:, 0], X_sklearn[:, 1], c=y, cmap='Spectral', s=5)
    plt.title(f"{title_prefix} – Gotowy t-SNE")

    plt.suptitle(f"{title_prefix}: Własna vs Gotowa implementacja t-SNE", fontsize=14)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    tsne_params = {'perplexity': 30.0, 'max_iter': 500, 'learning_rate': 100.0, 'random_state': 42}

    # Test 1 – make_blobs
    X1, y1 = make_blobs(n_samples=300, centers=4, n_features=10, random_state=1)
    test_and_plot(X1, y1, "Blobs", tsne_params)

    # Test 2 – Swiss Roll
    X2, t2 = make_swiss_roll(n_samples=300, noise=0.01, random_state=1)
    test_and_plot(X2, t2, "Swiss Roll", tsne_params)

    # Test 3 – make_s_curve
    X3, y3 = make_s_curve(n_samples=300, random_state=1)
    test_and_plot(X3, y3, "S curve", tsne_params)