"""Rolling-origin forecasting engine."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from tsfm_mre.data import Panel
from tsfm_mre.models import Forecast, Forecaster
from tsfm_mre.schema import ES_COLUMNS, QUANTILE_COLUMNS


def evaluation_origins(dates: pd.DatetimeIndex, start: str, horizon: int) -> np.ndarray:
    """Indices of all origins on or after ``start`` whose target lies inside the sample."""
    first = int(dates.searchsorted(pd.Timestamp(start)))
    return np.arange(first, len(dates) - horizon, dtype=np.int64)


def forecast_table(
    model: str,
    series: str,
    horizon: int,
    dates: pd.DatetimeIndex,
    origins: np.ndarray,
    forecast: Forecast,
) -> pd.DataFrame:
    """Arrange a batch of forecasts as rows of the prediction schema."""
    data: dict[str, object] = {
        "model": model,
        "series": series,
        "origin": dates[origins],
        "horizon": horizon,
        "target_date": dates[origins + horizon],
        "mean": forecast.mean,
    }
    data.update({c: forecast.quantiles[:, j] for j, c in enumerate(QUANTILE_COLUMNS)})
    data.update({c: forecast.es[:, j] for j, c in enumerate(ES_COLUMNS)})
    return pd.DataFrame(data)


def rolling_origin_forecasts(
    panel: Panel,
    models: Sequence[Forecaster],
    evaluation_start: str,
    horizons: Sequence[int],
) -> pd.DataFrame:
    """Forecast every series at every origin from ``evaluation_start`` onwards.

    Each model is re-estimated at each origin on the data observed up to that
    origin, and forecasts ``y[t + h]`` for every horizon ``h``.
    """
    dates = panel.dates
    frames = []
    for series in panel.series:
        y = panel.values[series].to_numpy(dtype=float)
        for horizon in horizons:
            origins = evaluation_origins(dates, evaluation_start, horizon)
            if origins.size == 0:
                raise ValueError(f"no forecast origins for horizon {horizon}")
            for model in models:
                required = model.min_history(horizon)
                if origins[0] < required:
                    raise ValueError(
                        f"model '{model.name}' needs {required} observations before the first "
                        f"origin; the evaluation starts after {origins[0]}"
                    )
                forecast = model.forecast(y, origins, horizon)
                frames.append(forecast_table(model.name, series, horizon, dates, origins, forecast))
    return pd.concat(frames, ignore_index=True)
