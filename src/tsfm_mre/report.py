"""Markdown and self-contained HTML reports of an evaluation run."""

from __future__ import annotations

import base64
import html
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from tsfm_mre.evaluate import INTERVALS, POOLED, EvaluationTables


@dataclass(frozen=True)
class Table:
    """A report table: caption, header, formatted rows and column alignment ("l" or "r")."""

    caption: str
    header: list[str]
    rows: list[list[str]]
    align: str


def fmt_p(p: float) -> str:
    """Format a p-value with three decimals, or as "<0.001"."""
    if pd.isna(p):
        return "n/a"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def fmt_bp(x: float) -> str:
    """Format a value in percentage points as basis points."""
    return f"{100.0 * x:.2f}"


def fmt_pct(x: float, signed: bool = False) -> str:
    """Format a fraction as a percentage."""
    return f"{100.0 * x:+.1f}%" if signed else f"{100.0 * x:.1f}%"


LOSS_CAPTIONS = {
    "crps": "Continuous ranked probability score",
    "tail_ql": "Tail quantile loss at the two VaR levels",
}


def accuracy_section(
    tables: EvaluationTables, models: Sequence[str], benchmark: str, loss: str
) -> Table:
    """Series-pooled loss by horizon with Reality Check and SPA p-values."""
    pooled = tables.metrics[tables.metrics["series"] == POOLED].set_index(["horizon", "model"])
    joint = tables.joint[(tables.joint["scope"] == POOLED) & (tables.joint["loss"] == loss)]
    joint = joint.set_index("horizon")
    rows = []
    for horizon in sorted(pooled.index.get_level_values("horizon").unique()):
        base = pooled.loc[(horizon, benchmark), loss]
        cells = [str(horizon)]
        for model in models:
            value = pooled.loc[(horizon, model), loss]
            change = fmt_pct(value / base - 1.0, signed=True)
            cells.append(fmt_bp(value) if model == benchmark else f"{fmt_bp(value)} ({change})")
        has_joint = horizon in joint.index
        cells.append(fmt_p(joint.loc[horizon, "rc_p"]) if has_joint else "n/a")
        cells.append(fmt_p(joint.loc[horizon, "spa_p"]) if has_joint else "n/a")
        rows.append(cells)
    header = [
        "Horizon",
        *[f"{m} (benchmark)" if m == benchmark else m for m in models],
        "RC p",
        "SPA p",
    ]
    caption = (
        f"{LOSS_CAPTIONS[loss]} (basis points), pooled over series, with the change relative to "
        "the benchmark; lower is better. RC (White's Reality Check) and SPA (Hansen's test, "
        "consistent version) give p-values for the null hypothesis that no model has a lower "
        "expected loss than the benchmark."
    )
    return Table(caption, header, rows, "r" * len(header))


def calibration_section(tables: EvaluationTables, models: Sequence[str], horizon: int) -> Table:
    """Interval coverage and point accuracy at one horizon, pooled over series."""
    pooled = tables.metrics[
        (tables.metrics["series"] == POOLED) & (tables.metrics["horizon"] == horizon)
    ]
    pooled = pooled.set_index("model")
    rows = []
    for model in models:
        rows.append(
            [model]
            + [fmt_pct(pooled.loc[model, f"coverage_{round(c * 100)}"]) for c in INTERVALS]
            + [fmt_bp(pooled.loc[model, "mae_median"]), fmt_bp(pooled.loc[model, "rmse_mean"])]
        )
    header = ["Model", *[f"{round(c * 100)}% interval" for c in INTERVALS], "MAE (bp)", "RMSE (bp)"]
    caption = (
        f"Empirical coverage of central predictive intervals and point accuracy at the "
        f"{horizon}-day horizon, pooled over series (median for MAE, mean for RMSE)."
    )
    return Table(caption, header, rows, "l" + "r" * (len(header) - 1))


def risk_section(
    tables: EvaluationTables, models: Sequence[str], series_order: Sequence[str]
) -> Table:
    """VaR and ES backtests by series, tail and model."""
    risk = tables.risk.assign(
        _series=tables.risk["series"].map({s: i for i, s in enumerate(series_order)}),
        _tail=tables.risk["tail"].map({"upper": 0, "lower": 1}),
        _model=tables.risk["model"].map({m: i for i, m in enumerate(models)}),
    ).sort_values(["_series", "_tail", "_model"])
    rows = [
        [
            r.series,
            r.tail,
            r.model,
            f"{r.exceedances} / {r.n}",
            fmt_pct(r.exceedance_rate),
            fmt_p(r.kupiec_p),
            fmt_p(r.conditional_coverage_p),
            f"{r.basel_zone} ({r.basel_exceedances})",
            f"{r.es_z2:+.2f}",
            fmt_p(r.es_p),
        ]
        for r in risk.itertuples()
    ]
    first = risk.iloc[0]
    header = [
        "Series",
        "Tail",
        "Model",
        "Exceedances",
        "Rate",
        "Kupiec p",
        "CC p",
        f"Basel zone (last {first.basel_window})",
        "ES Z2",
        "ES p",
    ]
    caption = (
        f"{first.horizon}-day VaR at level {1 - first.var_alpha:.0%} and ES at level "
        f"{1 - first.es_alpha:.1%}. The upper tail is the loss from rising yields, the lower tail "
        "the loss from falling yields. Kupiec: unconditional coverage; CC: Christoffersen "
        "conditional coverage; ES: Acerbi–Székely Z2 with Monte Carlo p-value (negative Z2 "
        "signals underestimated tail risk)."
    )
    return Table(caption, header, rows, "lll" + "r" * (len(header) - 3))


def _markdown(table: Table) -> str:
    align = {"l": ":---", "r": "---:"}
    lines = [table.caption, "", "| " + " | ".join(table.header) + " |"]
    lines.append("| " + " | ".join(align[a] for a in table.align) + " |")
    lines += ["| " + " | ".join(row) + " |" for row in table.rows]
    return "\n".join(lines)


def _html_table(table: Table) -> str:
    head = "".join(f"<th>{html.escape(h)}</th>" for h in table.header)
    body = "".join(
        "<tr>"
        + "".join(
            f'<td class="{"num" if a == "r" else "txt"}">{html.escape(c)}</td>'
            for c, a in zip(row, table.align, strict=True)
        )
        + "</tr>"
        for row in table.rows
    )
    return (
        f'<figure class="table"><figcaption>{html.escape(table.caption)}</figcaption>'
        f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></figure>"
    )


def write_reports(
    out_dir: Path,
    title: str,
    overview: list[tuple[str, str]],
    tables: EvaluationTables,
    models: Sequence[str],
    series: Sequence[str],
    benchmark: str,
    risk_horizon: int,
    figures: list[tuple[str, Path]],
) -> tuple[Path, Path]:
    """Write ``summary.md`` (figures linked) and ``report.html`` (figures embedded)."""
    sections = [
        ("Probabilistic accuracy", accuracy_section(tables, models, benchmark, "crps")),
        ("Tail accuracy", accuracy_section(tables, models, benchmark, "tail_ql")),
        ("Calibration", calibration_section(tables, models, risk_horizon)),
        ("Risk backtests", risk_section(tables, models, series)),
    ]
    md = [f"# {title}", "", "| | |", "| :--- | :--- |"]
    md += [f"| {k} | {v} |" for k, v in overview]
    for heading, table in sections:
        md += ["", f"## {heading}", "", _markdown(table)]
    md += ["", "## Figures", ""]
    md += [f"![{caption}]({path.relative_to(out_dir).as_posix()})\n" for caption, path in figures]
    summary = out_dir / "summary.md"
    summary.write_text("\n".join(md).rstrip() + "\n", encoding="utf-8", newline="\n")

    overview_rows = "".join(
        f"<tr><th>{html.escape(k)}</th><td>{html.escape(v)}</td></tr>" for k, v in overview
    )
    blocks = [f"<h2>{html.escape(h)}</h2>{_html_table(t)}" for h, t in sections]
    images = "".join(
        f'<figure><img alt="{html.escape(c)}" src="data:image/png;base64,'
        f'{base64.b64encode(p.read_bytes()).decode("ascii")}"><figcaption>{html.escape(c)}'
        "</figcaption></figure>"
        for c, p in figures
    )
    page = HTML_TEMPLATE.format(
        title=html.escape(title),
        overview=overview_rows,
        sections="".join(blocks),
        figures=images,
    )
    report = out_dir / "report.html"
    report.write_text(page, encoding="utf-8", newline="\n")
    return summary, report


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
:root {{ --fg: #1d2733; --muted: #5b6776; --line: #d9dee5; --head: #f3f5f8; --bg: #ffffff; }}
@media (prefers-color-scheme: dark) {{
  :root {{ --fg: #e6e9ee; --muted: #a3adba; --line: #39424e; --head: #1f262f; --bg: #12171d; }}
}}
body {{ margin: 0 auto; max-width: 1080px; padding: 32px 20px 64px; background: var(--bg);
       color: var(--fg); font: 15px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif; }}
h1 {{ font-size: 26px; margin: 0 0 16px; }}
h2 {{ font-size: 19px; margin: 36px 0 10px; padding-bottom: 4px;
      border-bottom: 1px solid var(--line); }}
table {{ border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }}
th, td {{ padding: 6px 10px; border-bottom: 1px solid var(--line); text-align: left; }}
thead th {{ background: var(--head); font-weight: 600; }}
td.num {{ text-align: right; white-space: nowrap; }}
.overview th {{ width: 200px; color: var(--muted); font-weight: 500; }}
figure {{ margin: 18px 0; overflow-x: auto; }}
figcaption {{ color: var(--muted); font-size: 13.5px; margin: 6px 0; }}
img {{ max-width: 100%; background: #ffffff; border-radius: 4px; }}
</style>
</head>
<body>
<h1>{title}</h1>
<table class="overview">{overview}</table>
{sections}
<h2>Figures</h2>
{figures}
</body>
</html>
"""
