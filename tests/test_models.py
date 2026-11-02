import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from models import BiasModel, MatrixFactorisation, rmse  # noqa: E402


def synthetic(n_users=40, n_items=30, k=3, seed=1):
    rng = np.random.default_rng(seed)
    P, Q = rng.normal(0, 1, (n_users, k)), rng.normal(0, 1, (n_items, k))
    u, i = np.meshgrid(np.arange(n_users), np.arange(n_items), indexing="ij")
    u, i = u.ravel(), i.ravel()
    r = np.clip(3 + (P[u] * Q[i]).sum(1) * 0.6, 0.5, 5)
    keep = rng.random(len(u)) < 0.6
    return u, i, r, keep, n_users, n_items


def test_mf_recovers_low_rank_structure_better_than_biases():
    u, i, r, keep, nu, ni = synthetic()
    mf = MatrixFactorisation(nu, ni, k=3, lr=0.05, reg=0.01, epochs=200, batch=64).fit(u[keep], i[keep], r[keep])
    bias = BiasModel(nu, ni, damping=1).fit(u[keep], i[keep], r[keep])
    assert rmse(r[~keep], mf.predict(u[~keep], i[~keep])) < 0.7 * rmse(r[~keep], bias.predict(u[~keep], i[~keep]))


def test_predictions_stay_in_rating_range():
    u, i, r, keep, nu, ni = synthetic()
    p = MatrixFactorisation(nu, ni, epochs=3).fit(u, i, r).predict(u, i)
    assert p.min() >= 0.5 and p.max() <= 5.0
