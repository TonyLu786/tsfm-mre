import json

import numpy as np
import pandas as pd
import pytest

from tsfm_mre import data
from tsfm_mre.data import (
    CALIBRATION_CORR,
    CALIBRATION_SD,
    csv_panel,
    fred_panel,
    parse_fred_csv,
    synthetic_panel,
)

SERIES = ["DGS3MO", "DGS2", "DGS10"]


def test_synthetic_panel_is_reproducible():
    a = synthetic_panel(SERIES, "2010-01-01", "2012-12-31", seed=1)
    b = synthetic_panel(SERIES, "2010-01-01", "2012-12-31", seed=1)
    c = synthetic_panel(SERIES, "2010-01-01", "2012-12-31", seed=2)
    assert a.digest() == b.digest() != c.digest()


def test_synthetic_panel_matches_its_calibration():
    panel = synthetic_panel(SERIES, "2003-01-01", "2025-12-31", seed=20251017)
    changes = panel.values.diff().dropna()
    assert (panel.values >= 0).all().all()
    np.testing.assert_allclose(changes.std().to_numpy(), CALIBRATION_SD, rtol=0.2)
    np.testing.assert_allclose(changes.corr().to_numpy(), CALIBRATION_CORR, atol=0.1)
    assert panel.values.mean().is_monotonic_increasing  # upward-sloping curve on average


@pytest.mark.parametrize("header", ["observation_date", "DATE"])
def test_parse_fred_csv(header):
    text = f"{header},DGS10\n2025-01-02,4.57\n2025-01-03,.\n2025-01-06,4.62\n"
    series = parse_fred_csv(text, "DGS10")
    assert series.index[0] == pd.Timestamp("2025-01-02")
    assert np.isnan(series.iloc[1]) and series.iloc[2] == pytest.approx(4.62)


def test_fred_panel_caches_and_verifies_downloads(tmp_path, monkeypatch):
    calls = []

    def fake_fetch(series, start, end, timeout=60.0):
        calls.append(series)
        return f"observation_date,{series}\n2025-01-02,4.0\n2025-01-03,.\n2025-01-06,4.1\n".encode()

    monkeypatch.setattr(data, "fetch_fred_csv", fake_fetch)
    panel = fred_panel(["DGS2", "DGS10"], "2025-01-01", "2025-01-31", tmp_path)
    assert calls == ["DGS2", "DGS10"]
    assert len(panel.values) == 2 and panel.provenance["non_trading_dates_dropped"] == 1
    record = json.loads((tmp_path / "DGS10_2025-01-01_2025-01-31.json").read_text())
    assert record["sha256"] == panel.provenance["files"][1]["sha256"]

    fred_panel(["DGS2", "DGS10"], "2025-01-01", "2025-01-31", tmp_path)
    assert calls == ["DGS2", "DGS10"]  # served from the cache

    (tmp_path / "DGS10_2025-01-01_2025-01-31.csv").write_text("observation_date,DGS10\n")
    with pytest.raises(ValueError, match="SHA-256"):
        fred_panel(["DGS2", "DGS10"], "2025-01-01", "2025-01-31", tmp_path)


def test_csv_panel_aligns_series(tmp_path):
    path = tmp_path / "yields.csv"
    path.write_text("date,DGS2,DGS10\n2024-01-02,4.3,3.9\n2024-01-03,,3.95\n2024-01-04,4.4,4.0\n")
    panel = csv_panel(path, ["DGS2", "DGS10"], "2024-01-01", "2024-12-31")
    assert len(panel.values) == 2
    assert panel.provenance["incomplete_dates_dropped"] == 1
    with pytest.raises(ValueError, match="lacks columns"):
        csv_panel(path, ["DGS30"], "2024-01-01", "2024-12-31")
