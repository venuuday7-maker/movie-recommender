# Movie Recommender (MovieLens, NumPy only)

Collaborative filtering built from scratch on MovieLens *latest-small* (100k ratings, 610 users, ~9k movies): a global-mean baseline, shrunk user+item biases, and biased **matrix factorisation trained with mini-batch SGD** (no recommender libraries). It is evaluated two ways, because predicting ratings and recommending items are different problems.

```bash
pip install -r requirements.txt
python src/evaluate.py     # downloads the dataset on first run (not redistributed here)
pytest -q
```

## Method
- 80/20 random split of ratings; test rows with unseen users or items are dropped (cold start is out of scope).
- **Model selection uses only a validation slice of the training data** (regularisation, number of epochs, popularity-blend weight). The test set is used once, at the end.
- Rating quality: **RMSE**. Recommendation quality: **precision@10 / recall@10** over 595 users, ranking only movies the user had not rated, with "liked" meaning a held-out rating of 4 or more.

## Results
| Model | RMSE | precision@10 | recall@10 |
|---|---|---|---|
| Global mean | 1.038 | n/a | n/a |
| Popularity (most-rated) | n/a | 0.114 | 0.096 |
| User+item bias | 0.868 | 0.081 | 0.064 |
| Matrix factorisation (reg 0.1, 36 epochs) | **0.848** | 0.006 | 0.004 |
| MF + popularity blend | n/a | **0.126** | **0.108** |

## What I learned
- MF wins on **rating prediction** (RMSE 0.848 vs 0.868 for biases) but is **terrible at top-10 ranking** (precision 0.006). Ranking by predicted rating pushes obscure movies with a few enthusiastic raters to the top. Low RMSE does not mean good recommendations.
- A simple fix that is well known in practice: blend the standardised MF score with standardised log-popularity. That beat the popularity baseline (0.126 vs 0.114). The blend weight (2.0) sits at the edge of the grid I searched, so a wider search might do better.
- The first version without validation-based early stopping overfit: test RMSE bottomed out around epoch 10 and then rose.

## Limitations
Single random split (no repeated runs or confidence intervals), explicit ratings only, no implicit-feedback model (e.g. BPR/ALS), no cold-start handling. Dataset licence: see GroupLens; the data is downloaded, not stored in this repo.
