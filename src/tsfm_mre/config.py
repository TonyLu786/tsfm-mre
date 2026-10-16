"""Experiment configuration: YAML files mapped onto validated dataclasses."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from tsfm_mre.models import MODEL_REGISTRY
from tsfm_mre.schema import ES_LEVELS, QUANTILE_LEVELS

DATA_SOURCES = ("synthetic", "fred", "csv")


class ConfigError(ValueError):
    """Raised for an invalid experiment configuration."""


@dataclass(frozen=True)
class DataConfig:
    """Data source (``synthetic``, ``fred`` or ``csv``), series and sample period.

    ``cache_dir`` (FRED downloads) and ``path`` (CSV input) are relative to the working
    directory; ``resolution`` is the recording precision used by the randomised PIT.
    """

    source: str
    series: tuple[str, ...]
    start: str
    end: str
    cache_dir: str = "data/raw/fred"
    path: str | None = None
    resolution: float | None = None


@dataclass(frozen=True)
class EvaluationConfig:
    """First forecast origin and forecast horizons in trading days."""

    start: str
    horizons: tuple[int, ...] = (1, 5, 10, 20)


@dataclass(frozen=True)
class RiskConfig:
    """Horizon and levels of the VaR and ES backtests."""

    horizon: int = 1
    var_alpha: float = 0.01
    es_alpha: float = 0.025
    basel_window: int = 250
    es_simulations: int = 2000


@dataclass(frozen=True)
class ComparisonConfig:
    """Benchmark model and stationary-bootstrap settings of the comparison tests."""

    benchmark: str
    bootstrap_reps: int = 1000
    mean_block_length: float = 10.0


@dataclass(frozen=True)
class ReportConfig:
    """Series shown in the figures (default: the last configured series)."""

    focus_series: str | None = None


@dataclass(frozen=True)
class Config:
    """A complete experiment configuration."""

    name: str
    seed: int
    data: DataConfig
    evaluation: EvaluationConfig
    models: tuple[dict[str, Any], ...]
    risk: RiskConfig
    comparison: ComparisonConfig
    report: ReportConfig

    @property
    def model_names(self) -> list[str]:
        """Model labels in configuration order."""
        return [str(m.get("label", m["name"])) for m in self.models]

    def to_dict(self) -> dict[str, Any]:
        """Plain-dictionary form recorded in run manifests."""
        return asdict(self)


def _section(cls: type, raw: Any, name: str) -> Any:
    if not isinstance(raw, dict):
        raise ConfigError(f"'{name}' must be a mapping")
    allowed = {f.name for f in fields(cls)}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ConfigError(f"unknown keys in '{name}': {', '.join(unknown)}")
    values = {k: tuple(v) if isinstance(v, list) else v for k, v in raw.items()}
    values = {k: str(v) if k in {"start", "end"} else v for k, v in values.items()}
    try:
        return cls(**values)
    except TypeError as exc:
        raise ConfigError(f"invalid '{name}' section: {exc}") from exc


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


def parse_config(raw: dict[str, Any]) -> Config:
    """Validate a configuration mapping and build a :class:`Config`."""
    _check(isinstance(raw, dict), "configuration must be a mapping")
    allowed = {f.name for f in fields(Config)}
    unknown = sorted(set(raw) - allowed)
    _check(not unknown, f"unknown top-level keys: {', '.join(unknown)}")
    for key in ("name", "seed", "data", "evaluation", "models", "comparison"):
        _check(key in raw, f"missing top-level key '{key}'")

    data = _section(DataConfig, raw["data"], "data")
    evaluation = _section(EvaluationConfig, raw["evaluation"], "evaluation")
    risk = _section(RiskConfig, raw.get("risk", {}), "risk")
    comparison = _section(ComparisonConfig, raw["comparison"], "comparison")
    report = _section(ReportConfig, raw.get("report", {}), "report")

    _check(data.source in DATA_SOURCES, f"data.source must be one of {DATA_SOURCES}")
    _check(data.source != "csv" or bool(data.path), "data.path is required for csv data")
    _check(
        data.resolution is None or float(data.resolution) >= 0.0,
        "data.resolution must be non-negative",
    )
    _check(len(data.series) > 0, "data.series must list at least one series")
    _check(len(set(data.series)) == len(data.series), "data.series contains duplicates")
    try:
        start, end = pd.Timestamp(data.start), pd.Timestamp(data.end)
        eval_start = pd.Timestamp(evaluation.start)
    except ValueError as exc:
        raise ConfigError(f"invalid date: {exc}") from exc
    _check(start < eval_start < end, "evaluation.start must lie strictly inside the data sample")

    horizons = evaluation.horizons
    _check(len(horizons) > 0, "evaluation.horizons must not be empty")
    _check(
        all(isinstance(h, int) and h >= 1 for h in horizons), "horizons must be positive integers"
    )
    _check(len(set(horizons)) == len(horizons), "evaluation.horizons contains duplicates")

    models = raw["models"]
    _check(isinstance(models, list) and len(models) > 0, "models must be a non-empty list")
    for spec in models:
        _check(isinstance(spec, dict) and "name" in spec, "each model needs a 'name'")
        _check(spec["name"] in MODEL_REGISTRY, f"unknown model '{spec['name']}'")
    labels = [str(m.get("label", m["name"])) for m in models]
    _check(len(set(labels)) == len(labels), "model labels must be unique")

    _check(risk.horizon in horizons, "risk.horizon must be one of evaluation.horizons")
    for alpha in (risk.var_alpha, risk.es_alpha):
        _check(
            any(abs(alpha - q) < 1e-12 for q in QUANTILE_LEVELS)
            and any(abs(1.0 - alpha - q) < 1e-12 for q in QUANTILE_LEVELS),
            f"risk level {alpha} needs quantiles at {alpha} and {1 - alpha}",
        )
    _check(
        any(abs(risk.es_alpha - q) < 1e-12 for q in ES_LEVELS)
        and any(abs(1.0 - risk.es_alpha - q) < 1e-12 for q in ES_LEVELS),
        f"risk.es_alpha must be one of {[x for x in ES_LEVELS if x < 0.5]}",
    )
    _check(risk.basel_window >= 1, "risk.basel_window must be positive")
    _check(risk.es_simulations >= 99, "risk.es_simulations must be at least 99")
    _check(comparison.benchmark in labels, "comparison.benchmark must be one of the models")
    _check(comparison.bootstrap_reps >= 99, "comparison.bootstrap_reps must be at least 99")
    _check(comparison.mean_block_length >= 1.0, "comparison.mean_block_length must be >= 1")
    if report.focus_series is None:
        report = ReportConfig(focus_series=data.series[-1])
    _check(report.focus_series in data.series, "report.focus_series must be one of data.series")

    return Config(
        name=str(raw["name"]),
        seed=int(raw["seed"]),
        data=data,
        evaluation=evaluation,
        models=tuple(dict(m) for m in models),
        risk=risk,
        comparison=comparison,
        report=report,
    )


def load_config(path: str | Path) -> Config:
    """Read and validate a YAML experiment configuration."""
    with open(path, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return parse_config(raw)
