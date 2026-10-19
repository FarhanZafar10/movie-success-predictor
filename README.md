# Movie Revenue Predictor API

A regression model that predicts a movie's box-office revenue from information known before release, deployed as a REST API.

> **Note:** This is a proof-of-concept exploring what drives box-office revenue. It is not a production tool for real greenlighting decisions (see Limitations).

## Problem
Given a movie's budget, runtime, genres, release timing, and production scale, predict how much revenue it will earn. This is a regression problem: the output is a dollar amount, not a category.

## Dataset
[Full TMDB Movies Dataset (Kaggle, asaniczka)](https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies). It has about 471,000 movies, but only **10,761** had real budget and revenue data, a released status, and valid genres, runtime, and release date. Those rows form the working dataset. I chose it over the older TMDB 5000 dataset (2017) because it is far more recent and larger.

## Approach
- **Filtering:** Kept only released movies with budget > 0 and revenue > 0. Zeros in this dataset mean "never recorded," not "earned $0."
- **Avoiding data leakage:** Excluded `vote_average`, `vote_count`, and `popularity`. They only exist after a movie is released and watched, so a real pre-release prediction would never have them.
- **Feature engineering:**
  - Release year and month extracted from the release date
  - Genres multi-label one-hot encoded (19 genre columns)
  - Counts of production companies, countries, and spoken languages, instead of one-hot encoding thousands of unique companies
- **Handling skew:** Revenue is heavily right-skewed (mean $63.8M vs median $10.7M). I trained on `log1p(revenue)` and reversed it with `expm1` at prediction time.
- **Models compared:** Linear Regression (baseline) and Random Forest Regressor, evaluated on an 80/20 train/test split.

## Results (held-out test set, log-revenue scale)
| Model | RMSE | R² |
|---|---|---|
| Linear Regression | 3.05 | 0.46 |
| Random Forest | 1.89 | 0.79 |

Random Forest explains about 79% of the variance in revenue, much better than the linear baseline. That suggests revenue depends on interactions between features (for example budget combined with genre and release season), which a linear model can't capture.

### Feature importance
**Budget accounts for about 81% of the Random Forest's predictive power.** Runtime, release year, release month, and production scale each contribute modestly. This matches how the industry works: studios set budgets based on expected returns, so budget already encodes a lot of "expected success."

## API
Built with Flask. Send a POST request to `/predict`:

```json
{
  "budget": 150000000,
  "runtime": 148,
  "release_year": 2024,
  "release_month": 7,
  "genres": ["Action", "Science Fiction", "Adventure"],
  "num_production_companies": 3,
  "num_production_countries": 2,
  "num_spoken_languages": 2
}
```

Response:
```json
{ "predicted_revenue": 350046219.85 }
```

## Running locally                                                                                     The API runs on `http://127.0.0.1:5001`. Run `python test_api.py` to send a sample request.

## Limitations
- Revenue is inherently hard to predict. About 21% of the variance comes from factors not in this dataset: marketing spend, star power, franchise or sequel status, competing releases, critical buzz, and luck.
- The model leans heavily on budget, so it works best as a budget-to-revenue relationship, not a deep understanding of what makes a film succeed.
- TMDB is crowdsourced, and only movies with community-entered financials were usable, so the data may skew toward better-known films.
- Historical patterns may not capture recent shifts, such as the effect of streaming on theatrical revenue.

## Tech stack
Python, pandas, NumPy, scikit-learn, Flask, joblib