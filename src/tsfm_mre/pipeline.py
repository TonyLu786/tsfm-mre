"""End-to-end experiment: data, rolling-origin forecasts, evaluation, figures, reports."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from tsfm_mre import plots
from tsfm_mre.config import Config
from tsfm_mre.data import Panel, load_panel
from tsfm_mre.evaluate import EvaluationTables, evaluate
from tsfm_mre.forecast import rolling_origin_forecasts
from tsfm_mre.models import build_model
from tsfm_mre.provenance import build_manifest, write_manifest
from tsfm_mre.report import write_reports
from tsfm_mre.schema import read_predictions, validate_predictions, write_predictions


@dataclass(frozen=True)
class RunResult:
    """Outputs of :func:`run`: the panel, all forecasts, the evaluation tables and the manifest."""

    out_dir: Path
    panel: Panel
    predictions: pd.DataFrame
    tables: EvaluationTables
    manifest: dict


def _overview(
    config: Config, panel: Panel, predictions: pd.DataFrame, models: list[str]
) -> list[tuple[str, str]]:
    dates = panel.dates
    origins = predictions.loc[predictions["horizon"] == config.risk.horizon, "origin"]
    model_text = ", ".join(
        f"{m} (benchmark)" if m == config.comparison.benchmark else m for m in models
    )
    return [
        ("Data", panel.provenance.get("description", panel.provenance.get("source", ""))),
        ("Series", ", ".join(panel.series)),
        ("Sample", f"{dates[0]:%Y-%m-%d} to {dates[-1]:%Y-%m-%d} ({len(dates):,} trading days)"),
        (
            "Forecast origins",
            f"{origins.min():%Y-%m-%d} to {origins.max():%Y-%m-%d}, every trading day; "
            "models re-estimated at each origin",
        ),
        ("Horizons", ", ".join(str(h) for h in config.evaluation.horizons) + " trading days"),
        ("Models", model_text),
    ]


def run(
    config: Config,
    out_dir: str | Path,
    extra_predictions: Sequence[str | Path] = (),
    log: Callable[[str], None] = print,
) -> RunResult:
    """Run the experiment described by ``config`` and write all outputs to ``out_dir``.

    ``extra_predictions`` are prediction tables of further models (for example a
    time-series foundation model) in the schema of :mod:`tsfm_mre.schema`; they are
    evaluated and compared together with the configured models.
    """
    start = time.perf_counter()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    panel = load_panel(config.data, config.seed)
    log(f"data: {panel.provenance['source']}, {len(panel.dates)} days x {len(panel.series)} series")

    models = [build_model(spec) for spec in config.models]
    predictions = rolling_origin_forecasts(
        panel, models, config.evaluation.start, config.evaluation.horizons
    )
    names = [m.name for m in models]
    for path in extra_predictions:
        extra = read_predictions(path)
        clash = sorted(set(extra["model"]) & set(predictions["model"]))
        if clash:
            raise ValueError(f"{Path(path).name} repeats model names: {', '.join(clash)}")
        predictions = pd.concat([predictions, extra], ignore_index=True)
        names += [m for m in extra["model"].unique() if m not in names]
    predictions = validate_predictions(predictions)
    log(f"forecasts: {len(predictions):,} rows for models {', '.join(names)}")

    for model in names:
        write_predictions(
            predictions[predictions["model"] == model], out / "predictions" / f"{model}.csv.gz"
        )

    tables = evaluate(
        predictions,
        panel,
        risk_horizon=config.risk.horizon,
        var_alpha=config.risk.var_alpha,
        es_alpha=config.risk.es_alpha,
        basel_window=config.risk.basel_window,
        es_simulations=config.risk.es_simulations,
        benchmark=config.comparison.benchmark,
        bootstrap_reps=config.comparison.bootstrap_reps,
        mean_block_length=config.comparison.mean_block_length,
        seed=config.seed,
    )
    for name, table in (
        ("metrics.csv", tables.metrics),
        ("risk.csv", tables.risk),
        ("comparison_pairwise.csv", tables.pairwise),
        ("comparison_joint.csv", tables.joint),
    ):
        table.to_csv(out / name, index=False, float_format="%.8g", lineterminator="\n")
    log("evaluation: metrics, risk backtests and comparison tests written")

    focus, horizon = config.report.focus_series, config.risk.horizon
    alpha, figdir = config.risk.var_alpha, out / "figures"
    figures = [
        (
            f"{focus}: predictive intervals and VaR exceedances",
            plots.predictive_intervals(
                tables.scored, names, focus, horizon, alpha, figdir / "intervals.png"
            ),
        ),
        (
            f"{focus}: rolling VaR exceedances",
            plots.rolling_exceedances(
                tables.scored,
                names,
                focus,
                horizon,
                alpha,
                config.risk.basel_window,
                figdir / "rolling_exceedances.png",
            ),
        ),
        (
            "Scores by horizon",
            plots.scores_by_horizon(
                tables.metrics, names, config.comparison.benchmark, figdir / "scores_by_horizon.png"
            ),
        ),
        (
            "PIT histograms",
            plots.pit_histograms(tables.scored, names, horizon, figdir / "pit.png"),
        ),
    ]
    write_reports(
        out,
        f"TSFM-MRE results: {config.name}",
        _overview(config, panel, predictions, names),
        tables,
        names,
        panel.series,
        config.comparison.benchmark,
        horizon,
        figures,
    )

    manifest = build_manifest(
        config=config.to_dict(),
        data={**panel.provenance, "panel_sha256": panel.digest()},
        models={m.name: {"class": type(m).__name__, **m.params()} for m in models},
        out_dir=out,
        runtime_seconds=time.perf_counter() - start,
    )
    write_manifest(manifest, out)
    log(f"done in {manifest['runtime_seconds']:.1f} s: {out.as_posix()}")
    return RunResult(out, panel, predictions, tables, manifest)
