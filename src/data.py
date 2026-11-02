"""Download and load MovieLens "latest-small" (100k ratings). Not redistributed here; see the GroupLens licence."""
import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
DATA = Path(__file__).resolve().parent.parent / "data"


def load_ratings():
    csv = DATA / "ml-latest-small" / "ratings.csv"
    if not csv.exists():
        DATA.mkdir(exist_ok=True)
        with urllib.request.urlopen(URL) as r:
            zipfile.ZipFile(io.BytesIO(r.read())).extractall(DATA)
    return pd.read_csv(csv)


def split_ratings(df, test_frac=0.2, seed=42):
    """Random 80/20 split of ratings; test rows with an unseen user or item are dropped (cold start is out of scope)."""
    rng = np.random.default_rng(seed)
    mask = rng.random(len(df)) < test_frac
    train, test = df[~mask], df[mask]
    test = test[test.userId.isin(train.userId) & test.movieId.isin(train.movieId)]
    return train.reset_index(drop=True), test.reset_index(drop=True)
