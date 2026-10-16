"""Compare the H.15 panel with the synthetic panel (the table in docs/data.md).

Usage::

    python scripts/describe_data.py --fred configs/fred_h15.yaml --synthetic configs/synthetic.yaml
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from tsfm_mre.config import load_config
from tsfm_mre.data import Panel, load_panel


def describe(panel: Panel) -> pd.DataFrame:
    levels = panel.values
    changes = levels.diff().dropna() * 100.0  # basis points
    rows = {}
    for s in panel.series:
        c = changes[s]
        rows[s] = {
            "level mean (%)": levels[s].mean(),
            "level sd (%)": levels[s].std(),
            "level min (%)": levels[s].min(),
            "level max (%)": levels[s].max(),
            "daily change sd (bp)": c.std(),
            "daily change kurtosis": c.kurt(),
            "acf(1) of squared changes": (c**2).autocorr(1),
        }
    return pd.DataFrame(rows)


def correlations(panel: Panel) -> dict[str, float]:
    corr = panel.values.diff().dropna().corr()
    names = panel.series
    return {f"corr({a}, {b})": corr.loc[a, b] for i, a in enumerate(names) for b in names[i + 1 :]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fred", default="configs/fred_h15.yaml")
    parser.add_argument("--synthetic", default="configs/synthetic.yaml")
    args = parser.parse_args()
    panels = {}
    for label, path in (("H.15", args.fred), ("synthetic", args.synthetic)):
        config = load_config(path)
        panels[label] = load_panel(config.data, config.seed)
    series = panels["H.15"].series
    header = "| Statistic | " + " | ".join(f"{s} H.15 | {s} synthetic" for s in series) + " |"
    print(header)
    print("| :--- |" + " ---: |" * (2 * len(series)))
    tables = {k: describe(p) for k, p in panels.items()}
    for stat in tables["H.15"].index:
        cells = []
        for s in series:
            cells += [
                f"{tables['H.15'].loc[stat, s]:.2f}",
                f"{tables['synthetic'].loc[stat, s]:.2f}",
            ]
        print(f"| {stat} | " + " | ".join(cells) + " |")
    real, synthetic = correlations(panels["H.15"]), correlations(panels["synthetic"])
    print()
    print("| Correlation of daily changes | H.15 | synthetic |")
    print("| :--- | ---: | ---: |")
    for key in real:
        print(f"| {key} | {real[key]:.2f} | {synthetic[key]:.2f} |")
    share = {k: float(np.mean(p.values.iloc[:, 0] < 0.25)) for k, p in panels.items()}
    print()
    print(
        f"Share of days with {series[0]} below 0.25%: H.15 {share['H.15']:.0%}, "
        f"synthetic {share['synthetic']:.0%}"
    )


if __name__ == "__main__":
    main()
