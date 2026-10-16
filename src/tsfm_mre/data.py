"""Yield panels: FRED H.15 downloads, local CSV files and a calibrated synthetic generator.

All sources return a :class:`Panel` of daily constant-maturity yields in percent on a
common trading calendar. Dates on which any series is missing are dropped, so a
forecast horizon of ``h`` always refers to the ``h``-th next trading day.
"""

from __future__ import annotations

import hashlib
import io
import json
import urllib.request
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}&cosd={start}&coed={end}"

MATURITY_MONTHS: dict[str, int] = {
    "DGS1MO": 1,
    "DGS3MO": 3,
    "DGS6MO": 6,
    "DGS1": 12,
    "DGS2": 24,
    "DGS3": 36,
    "DGS5": 60,
    "DGS7": 84,
    "DGS10": 120,
    "DGS20": 240,
    "DGS30": 360,
}
"""Maturities (months) of the FRED constant-maturity Treasury series."""


@dataclass(frozen=True)
class Panel:
    """Daily yields (percent) indexed by trading date, one column per series."""

    values: pd.DataFrame
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def dates(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.values.index)

    @property
    def series(self) -> list[str]:
        return [str(c) for c in self.values.columns]

    def digest(self) -> str:
        """SHA-256 of the panel in a canonical CSV representation."""
        text = self.values.to_csv(float_format="%.10g", date_format="%Y-%m-%d", lineterminator="\n")
        return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _align(columns: dict[str, pd.Series], start: str, end: str) -> tuple[pd.DataFrame, int]:
    frame = pd.concat(columns, axis=1).sort_index()
    frame = frame.loc[pd.Timestamp(start) : pd.Timestamp(end)]
    complete = frame.dropna(how="any")
    complete.index.name = "date"
    return complete.astype(float), int(len(frame) - len(complete))


# ---------------------------------------------------------------------------
# FRED H.15
# ---------------------------------------------------------------------------


def parse_fred_csv(content: bytes | str, series: str) -> pd.Series:
    """Parse a FRED ``fredgraph.csv`` download (first column dates, second values)."""
    text = content.decode("utf-8") if isinstance(content, bytes) else content
    frame = pd.read_csv(io.StringIO(text), na_values=["."])
    if frame.shape[1] < 2:
        raise ValueError(f"unexpected FRED file layout for {series}")
    dates = pd.to_datetime(frame.iloc[:, 0])
    values = pd.to_numeric(frame.iloc[:, 1], errors="coerce").to_numpy(dtype=float)
    return pd.Series(values, index=pd.DatetimeIndex(dates), name=series)


def fetch_fred_csv(series: str, start: str, end: str, timeout: float = 60.0) -> bytes:
    """Download one series from FRED's public CSV endpoint."""
    url = FRED_CSV_URL.format(series=series, start=start, end=end)
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read()


def fred_panel(series: list[str], start: str, end: str, cache_dir: str | Path) -> Panel:
    """FRED H.15 yields with a local cache.

    Each download is stored unchanged next to a JSON record of its URL, retrieval
    time and SHA-256; cached files are checked against that digest before use.
    """
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    columns: dict[str, pd.Series] = {}
    files = []
    for name in series:
        path = cache / f"{name}_{start}_{end}.csv"
        record_path = path.with_suffix(".json")
        url = FRED_CSV_URL.format(series=name, start=start, end=end)
        if path.exists() and record_path.exists():
            content = path.read_bytes()
            record = json.loads(record_path.read_text(encoding="utf-8"))
            if record.get("sha256") != _sha256(content):
                raise ValueError(f"{path.name} does not match its recorded SHA-256")
        else:
            content = fetch_fred_csv(name, start, end)
            record = {
                "series": name,
                "url": url,
                "retrieved_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
                "sha256": _sha256(content),
            }
            path.write_bytes(content)
            record_path.write_text(
                json.dumps(record, indent=2) + "\n", encoding="utf-8", newline="\n"
            )
        columns[name] = parse_fred_csv(content, name)
        files.append({**record, "cache_file": path.name})
    values, dropped = _align(columns, start, end)
    provenance = {
        "source": "fred",
        "description": "FRED, H.15 Selected Interest Rates: market yield on U.S. Treasury "
        "securities at constant maturity, daily, percent",
        "files": files,
        "non_trading_dates_dropped": dropped,
    }
    return Panel(values, provenance)


# ---------------------------------------------------------------------------
# Local CSV
# ---------------------------------------------------------------------------


def csv_panel(path: str | Path, series: list[str], start: str, end: str) -> Panel:
    """Wide CSV with a ``date`` column and one column of yields (percent) per series."""
    path = Path(path)
    content = path.read_bytes()
    frame = pd.read_csv(io.BytesIO(content))
    if "date" not in frame.columns:
        raise ValueError(f"{path.name} needs a 'date' column")
    missing = [s for s in series if s not in frame.columns]
    if missing:
        raise ValueError(f"{path.name} lacks columns: {', '.join(missing)}")
    index = pd.DatetimeIndex(pd.to_datetime(frame["date"]))
    columns = {
        s: pd.Series(pd.to_numeric(frame[s], errors="coerce").to_numpy(), index=index)
        for s in series
    }
    values, dropped = _align(columns, start, end)
    provenance = {
        "source": "csv",
        "files": [{"file": path.name, "sha256": _sha256(content)}],
        "incomplete_dates_dropped": dropped,
    }
    return Panel(values, provenance)


# ---------------------------------------------------------------------------
# Synthetic H.15-calibrated panel
# ---------------------------------------------------------------------------

NS_DECAY = 0.0609
"""Nelson-Siegel decay for maturities in months (Diebold and Li, 2006)."""

CALIBRATION_MATURITIES = (3, 24, 120)
CALIBRATION_SD = np.array([0.044, 0.052, 0.057])
"""Standard deviation of daily changes (percentage points), H.15 3m/2y/10y, 2003-2025."""
CALIBRATION_CORR = np.array(
    [
        [1.00, 0.33, 0.19],
        [0.33, 1.00, 0.75],
        [0.19, 0.75, 1.00],
    ]
)
"""Correlation of daily changes, H.15 3m/2y/10y, 2003-2025."""

FACTOR_MEAN = np.array([3.45, -1.75, -1.0])
FACTOR_PERSISTENCE = np.array([0.999, 0.9995, 0.998])
GARCH_ALPHA, GARCH_BETA = 0.05, 0.94
STUDENT_DOF = 5.0
STRESS_EPISODES = (("2008-09-15", "2009-03-31", 2.5), ("2020-03-02", "2020-04-30", 2.5))
FLOOR_SHARPNESS = 8.0


def nelson_siegel_loadings(months: list[int] | np.ndarray) -> np.ndarray:
    """Level, slope and curvature loadings for the given maturities (months)."""
    x = NS_DECAY * np.asarray(months, dtype=float)
    slope = (1.0 - np.exp(-x)) / x
    return np.column_stack([np.ones_like(x), slope, slope - np.exp(-x)])


def synthetic_panel(series: list[str], start: str, end: str, seed: int) -> Panel:
    """Synthetic constant-maturity yields calibrated to daily H.15 dynamics.

    Three Nelson-Siegel factors follow persistent AR(1) processes. Their shocks
    share a GARCH(1,1) variance, have Student-t tails, are scaled up during two
    stress episodes and have a covariance chosen so that daily changes of the
    3-month, 2-year and 10-year yields match H.15 volatilities and correlations.
    A smooth floor keeps short rates near zero during low-rate regimes.
    """
    unknown = [s for s in series if s not in MATURITY_MONTHS]
    if unknown:
        raise ValueError(f"no maturity known for {', '.join(unknown)}")
    dates = pd.bdate_range(start, end)
    n = len(dates)
    rng = np.random.default_rng(seed)

    calibration = nelson_siegel_loadings(list(CALIBRATION_MATURITIES))
    cov_y = CALIBRATION_CORR * np.outer(CALIBRATION_SD, CALIBRATION_SD)
    inverse = np.linalg.inv(calibration)
    chol = np.linalg.cholesky(inverse @ cov_y @ inverse.T)

    z = rng.standard_t(STUDENT_DOF, size=(n, 3)) * np.sqrt((STUDENT_DOF - 2.0) / STUDENT_DOF)
    scale = np.ones(n)
    for first, last, factor in STRESS_EPISODES:
        scale[(dates >= first) & (dates <= last)] = factor
    scale /= np.sqrt(np.mean(scale**2))  # stress reallocates, not adds, variance

    omega = 1.0 - GARCH_ALPHA - GARCH_BETA
    factors = np.empty((n, 3))
    factors[0] = FACTOR_MEAN
    variance = 1.0
    for t in range(1, n):
        variance = omega + (GARCH_ALPHA * z[t - 1, 0] ** 2 + GARCH_BETA) * variance
        shock = scale[t] * np.sqrt(variance) * (chol @ z[t])
        factors[t] = FACTOR_MEAN + FACTOR_PERSISTENCE * (factors[t - 1] - FACTOR_MEAN) + shock

    latent = factors @ nelson_siegel_loadings([MATURITY_MONTHS[s] for s in series]).T
    yields = np.logaddexp(0.0, FLOOR_SHARPNESS * latent) / FLOOR_SHARPNESS
    values = pd.DataFrame(yields, index=pd.DatetimeIndex(dates, name="date"), columns=list(series))
    provenance = {
        "source": "synthetic",
        "description": "Nelson-Siegel three-factor model with GARCH(1,1) Student-t shocks, "
        "stress episodes and a smooth zero floor, calibrated to daily H.15 dynamics",
        "seed": int(seed),
    }
    return Panel(values, provenance)


DEFAULT_RESOLUTION = {"fred": 0.01, "csv": 0.0, "synthetic": 0.0}
"""Precision to which observations are recorded (percentage points); H.15 uses 0.01."""


def load_panel(data: Any, seed: int) -> Panel:
    """Load the panel described by a :class:`tsfm_mre.config.DataConfig`."""
    series = list(data.series)
    if data.source == "synthetic":
        panel = synthetic_panel(series, data.start, data.end, seed)
    elif data.source == "fred":
        panel = fred_panel(series, data.start, data.end, data.cache_dir)
    elif data.source == "csv":
        panel = csv_panel(data.path, series, data.start, data.end)
    else:
        raise ValueError(f"unknown data source '{data.source}'")
    resolution = data.resolution
    if resolution is None:
        resolution = DEFAULT_RESOLUTION[data.source]
    return Panel(panel.values, {**panel.provenance, "resolution": float(resolution)})
