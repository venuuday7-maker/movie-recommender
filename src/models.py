"""Rating predictors: global mean, user+item biases, and biased matrix factorisation (SGD, NumPy only)."""
import numpy as np


class Index:
    """Map raw ids to dense 0..n-1 indices."""

    def __init__(self, ids):
        self.values = np.unique(ids)
        self.lookup = {v: i for i, v in enumerate(self.values)}

    def __call__(self, ids):
        return np.fromiter((self.lookup[i] for i in ids), dtype=np.int64, count=len(ids))

    def __len__(self):
        return len(self.values)


class GlobalMean:
    def fit(self, u, i, r):
        self.mu = r.mean()
        return self

    def predict(self, u, i):
        return np.full(len(u), self.mu)


class BiasModel:
    """r_ui ~ mu + b_u + b_i, with shrinkage (damping) so rare users/items stay near the mean."""

    def __init__(self, n_users, n_items, damping=10.0):
        self.nu, self.ni, self.d = n_users, n_items, damping

    def fit(self, u, i, r):
        self.mu = r.mean()
        self.bi = np.bincount(i, weights=r - self.mu, minlength=self.ni) / (np.bincount(i, minlength=self.ni) + self.d)
        res = r - self.mu - self.bi[i]
        self.bu = np.bincount(u, weights=res, minlength=self.nu) / (np.bincount(u, minlength=self.nu) + self.d)
        return self

    def predict(self, u, i):
        return np.clip(self.mu + self.bu[u] + self.bi[i], 0.5, 5.0)

    def score_all(self, u):
        return self.mu + self.bu[u] + self.bi


class MatrixFactorisation:
    """r_ui ~ mu + b_u + b_i + p_u . q_i, trained with mini-batch SGD and L2 regularisation."""

    def __init__(self, n_users, n_items, k=32, lr=0.02, reg=0.05, epochs=30, batch=2048, seed=0):
        self.k, self.lr, self.reg, self.epochs, self.batch = k, lr, reg, epochs, batch
        self.rng = np.random.default_rng(seed)
        self.P = self.rng.normal(0, 0.1, (n_users, k))
        self.Q = self.rng.normal(0, 0.1, (n_items, k))
        self.bu, self.bi = np.zeros(n_users), np.zeros(n_items)

    def fit(self, u, i, r, val=None):
        self.mu = r.mean()
        self.history = []
        for _ in range(self.epochs):
            order = self.rng.permutation(len(r))
            for s in range(0, len(order), self.batch):
                idx = order[s:s + self.batch]
                uu, ii, rr = u[idx], i[idx], r[idx]
                err = rr - (self.mu + self.bu[uu] + self.bi[ii] + (self.P[uu] * self.Q[ii]).sum(1))
                pu, qi = self.P[uu].copy(), self.Q[ii].copy()
                np.add.at(self.bu, uu, self.lr * (err - self.reg * self.bu[uu]))
                np.add.at(self.bi, ii, self.lr * (err - self.reg * self.bi[ii]))
                np.add.at(self.P, uu, self.lr * (err[:, None] * qi - self.reg * pu))
                np.add.at(self.Q, ii, self.lr * (err[:, None] * pu - self.reg * qi))
            if val is not None:
                self.history.append(rmse(val[2], self.predict(val[0], val[1])))
        return self

    def predict(self, u, i):
        return np.clip(self.mu + self.bu[u] + self.bi[i] + (self.P[u] * self.Q[i]).sum(1), 0.5, 5.0)

    def score_all(self, u):
        """Predicted score for every item for user index u (used for top-N ranking)."""
        return self.mu + self.bu[u] + self.bi + self.Q @ self.P[u]


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))
