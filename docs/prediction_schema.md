# Prediction schema

All models, built-in and external, exchange forecasts through one table. A table can be
stored as CSV, gzipped CSV or Parquet (Parquet requires `pip install "tsfm-mre[parquet]"`).

## Columns

| Column | Type | Meaning |
| :--- | :--- | :--- |
| `model` | text | Model name; it labels the model in every table and figure |
| `series` | text | Series identifier, e.g. `DGS10` |
| `origin` | date | Last trading day of information used by the forecast |
| `horizon` | integer ≥ 1 | Forecast horizon in trading days |
| `target_date` | date | Trading day being forecast, `horizon` trading days after `origin` |
| `mean` | number | Predictive mean of the target |
| `q_0.001`, `q_0.0025`, `q_0.005`, `q_0.01`, `q_0.025`, `q_0.05`, `q_0.1`, `q_0.2`, …, `q_0.9`, `q_0.95`, `q_0.975`, `q_0.99`, `q_0.995`, `q_0.9975`, `q_0.999` | number | Predictive quantiles (21 levels, in the units of the data) |
| `es_0.01`, `es_0.025` | number | Lower-tail expected shortfall $\mathbb E[Y \mid Y \le q_\tau]$ |
| `es_0.975`, `es_0.99` | number | Upper-tail expected shortfall $\mathbb E[Y \mid Y \ge q_\tau]$ |

The names are available in Python as `tsfm_mre.schema.REQUIRED_COLUMNS`,
`QUANTILE_COLUMNS` and `ES_COLUMNS`; the levels as `QUANTILE_LEVELS` and `ES_LEVELS`.

## Validation

`tsfm-mre validate FILE` and `tsfm_mre.validate_predictions` check that:

- all columns are present and every forecast value is finite;
- horizons are positive integers and every target date follows its origin;
- quantiles are non-decreasing in the level;
- each expected shortfall lies in its tail (`es_0.01 ≤ q_0.01`, `es_0.99 ≥ q_0.99`, and so on);
- the key `(model, series, origin, horizon)` is unique.

## Producing the table from model output

**Sample paths.** For models that generate predictive draws, `summarize_samples`
returns the mean, the 21 quantiles and the four ES values for every forecast:

```python
from tsfm_mre import summarize_samples

mean, quantiles, es = summarize_samples(samples)  # samples: (n_forecasts, n_draws)
```

Lower-tail ES at level $\tau$ is the mean of the $\lceil \tau m \rceil$ smallest of $m$
draws, and upper-tail ES the mean of the $\lceil (1-\tau) m \rceil$ largest.

**Quantile forecasts.** Models that output a coarser grid, for example deciles, map
their quantile function onto the 21 levels and compute ES from it; a parametric tail or
sampling from the fitted distribution gives the outer levels.

**Parametric forecasts.** Gaussian, Student-t and similar forecasts evaluate the quantile
function and the tail expectation in closed form; `tsfm_mre.models.gaussian_forecast` is
an example.

## Evaluating external forecasts

```bash
tsfm-mre run configs/fred_h15.yaml --predictions my_tsfm.csv.gz other_model.parquet
```

External tables are validated, written next to the baseline forecasts in
`runs/<name>/predictions/` and included in all metrics, backtests, comparisons, figures
and reports. Model comparisons use the origins that every model forecasts; scores and
backtests use all rows of each model. Model names must differ from the configured
baselines.
