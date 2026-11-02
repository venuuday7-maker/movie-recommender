"""Compare rating prediction (RMSE) and top-10 recommendation quality (precision/recall@10)."""
import json
from pathlib import Path

import numpy as np

from data import load_ratings, split_ratings
from models import BiasModel, GlobalMean, Index, MatrixFactorisation, rmse

K_TOP = 10
LIKE = 4.0
OUT = Path(__file__).resolve().parent.parent / "results"


def topn_metrics(score_fn, train_u, train_i, test_u, test_i, test_r):
    """For every user with liked held-out items: rank unseen items, score precision/recall@K."""
    seen, liked = {}, {}
    for u, i in zip(train_u, train_i):
        seen.setdefault(u, set()).add(i)
    for u, i, r in zip(test_u, test_i, test_r):
        if r >= LIKE:
            liked.setdefault(u, set()).add(i)
    prec, rec = [], []
    for u in sorted(liked):
        scores = np.array(score_fn(u), dtype=float)
        scores[list(seen.get(u, ()))] = -np.inf  # never recommend what the user already rated
        top = set(np.argpartition(-scores, K_TOP)[:K_TOP])
        hits = len(top & liked[u])
        prec.append(hits / K_TOP)
        rec.append(hits / len(liked[u]))
    return float(np.mean(prec)), float(np.mean(rec)), len(prec)


def zs(x):
    return (x - x.mean()) / (x.std() + 1e-9)


def main():
    train, test = split_ratings(load_ratings())
    users, items = Index(train.userId.values), Index(train.movieId.values)
    tu, ti, tr = users(train.userId.values), items(train.movieId.values), train.rating.values
    vu, vi, vr = users(test.userId.values), items(test.movieId.values), test.rating.values
    nu, ni = len(users), len(items)
    print(f"train={len(train)} test={len(test)} users={nu} items={ni}")

    # ---- model selection on a validation slice of TRAIN only; the test set is touched once at the end ----
    rng = np.random.default_rng(7)
    vmask = rng.random(len(tr)) < 0.1
    best = (9.0, None, None)
    for reg in (0.05, 0.1, 0.2):
        m = MatrixFactorisation(nu, ni, reg=reg, epochs=40).fit(tu[~vmask], ti[~vmask], tr[~vmask], val=(tu[vmask], ti[vmask], tr[vmask]))
        ep = int(np.argmin(m.history)) + 1
        print(f"  reg={reg}: best val RMSE {min(m.history):.4f} at epoch {ep}")
        if min(m.history) < best[0]:
            best = (min(m.history), reg, ep)
    _, reg, epochs = best
    print(f"selected reg={reg}, epochs={epochs}")

    mean = GlobalMean().fit(tu, ti, tr)
    bias = BiasModel(nu, ni).fit(tu, ti, tr)
    mf = MatrixFactorisation(nu, ni, reg=reg, epochs=epochs).fit(tu, ti, tr)
    pop = np.bincount(ti, minlength=ni).astype(float)  # items with the most ratings
    logpop = zs(np.log1p(pop))

    # blend weight for "MF + popularity" chosen on the validation slice (models refit without it)
    mf_val = MatrixFactorisation(nu, ni, reg=reg, epochs=epochs).fit(tu[~vmask], ti[~vmask], tr[~vmask])
    best_a, best_p = 0.0, -1.0
    for a in (0.0, 0.25, 0.5, 1.0, 2.0):
        p, _, _ = topn_metrics(lambda u, a=a: zs(mf_val.score_all(u)) + a * logpop, tu[~vmask], ti[~vmask], tu[vmask], ti[vmask], tr[vmask])
        print(f"  blend alpha={a}: val precision@{K_TOP}={p:.4f}")
        if p > best_p:
            best_a, best_p = a, p
    print(f"selected blend alpha={best_a}")

    results = {"global mean": {"rmse": round(rmse(vr, mean.predict(vu, vi)), 4)},
               "popularity": {},
               "user+item bias": {"rmse": round(rmse(vr, bias.predict(vu, vi)), 4)},
               "matrix factorisation": {"rmse": round(rmse(vr, mf.predict(vu, vi)), 4)},
               "MF + popularity blend": {}}
    rankers = {"popularity": lambda u: pop, "user+item bias": bias.score_all, "matrix factorisation": mf.score_all,
               "MF + popularity blend": lambda u: zs(mf.score_all(u)) + best_a * logpop}
    for name, fn in rankers.items():
        p, r, n = topn_metrics(fn, tu, ti, vu, vi, vr)
        results[name].update({f"precision@{K_TOP}": round(p, 4), f"recall@{K_TOP}": round(r, 4), "users_evaluated": n})
    for name, row in results.items():
        print(f"{name:26s} {row}")
    OUT.mkdir(exist_ok=True)
    (OUT / "metrics.json").write_text(json.dumps({"selected": {"reg": reg, "epochs": epochs, "blend_alpha": best_a}, "results": results}, indent=2))


if __name__ == "__main__":
    main()
