import numpy as np
import pandas as pd
import pytest

from tsfm_mre.models import gaussian_forecast
from tsfm_mre.schema import (
    ES_COLUMNS,
    QUANTILE_COLUMNS,
    QUANTILE_LEVELS,
    REQUIRED_COLUMNS,
    SchemaError,
    read_predictions,
    summarize_samples,
    tail_count,
    validate_predictions,
    write_predictions,
)


def make_table(n: int = 5) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=n + 1)
    fc = gaussian_forecast(np.full(n, 4.0), np.full(n, 0.05))
    data = {
        "model": "m",
        "series": "DGS10",
        "origin": dates[:-1],
        "horizon": 1,
        "target_date": dates[1:],
        "mean": fc.mean,
    }
    data.update({c: fc.quantiles[:, j] for j, c in enumerate(QUANTILE_COLUMNS)})
    data.update({c: fc.es[:, j] for j, c in enumerate(ES_COLUMNS)})
    return pd.DataFrame(data)


def test_column_names():
    assert QUANTILE_COLUMNS[1] == "q_0.0025"
    assert QUANTILE_COLUMNS[10] == "q_0.5"
    assert ES_COLUMNS == ("es_0.01", "es_0.025", "es_0.975", "es_0.99")
    levels = np.asarray(QUANTILE_LEVELS)
    assert np.allclose(levels + levels[::-1], 1.0)


def test_tail_count_is_robust_to_rounding():
    assert tail_count(1.0 - 0.99, 500) == 5
    assert tail_count(0.025, 500) == 13
    assert tail_count(0.001, 100) == 1


def test_summarize_samples_follows_definitions():
    samples = np.arange(1.0, 1001.0)[None, :]
    mean, quantiles, es = summarize_samples(samples)
    assert mean[0] == pytest.approx(500.5)
    assert quantiles[0, QUANTILE_LEVELS.index(0.5)] == pytest.approx(500.5)
    assert es[0].tolist() == pytest.approx([5.5, 13.0, 988.0, 995.5])


def test_validate_accepts_and_sorts():
    table = make_table().iloc[::-1]
    out = validate_predictions(table)
    assert list(out.columns) == list(REQUIRED_COLUMNS)
    assert out["origin"].is_monotonic_increasing


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda t: t.drop(columns=["q_0.5"]), "missing columns"),
        (lambda t: t.assign(**{"q_0.6": t["q_0.4"] - 1.0}), "non-decreasing"),
        (lambda t: t.assign(**{"es_0.01": t["q_0.01"] + 0.01}), "tail beyond"),
        (lambda t: t.assign(target_date=t["origin"]), "follow origin"),
        (lambda t: pd.concat([t, t.iloc[:1]]), "duplicated"),
        (lambda t: t.assign(mean=np.nan), "finite"),
        (lambda t: t.assign(horizon=0), "positive integer"),
    ],
)
def test_validate_rejects_invalid_tables(mutate, message):
    with pytest.raises(SchemaError, match=message):
        validate_predictions(mutate(make_table()))


def test_gzip_output_is_byte_reproducible(tmp_path):
    table = make_table()
    a = write_predictions(table, tmp_path / "a.csv.gz").read_bytes()
    b = write_predictions(table, tmp_path / "b.csv.gz").read_bytes()
    assert a == b
    back = read_predictions(tmp_path / "a.csv.gz")
    assert np.allclose(back[list(QUANTILE_COLUMNS)], table[list(QUANTILE_COLUMNS)])


def test_unsupported_extension(tmp_path):
    with pytest.raises(SchemaError, match="unsupported"):
        read_predictions(tmp_path / "x.json")
