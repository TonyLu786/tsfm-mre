import json

import numpy as np
import pandas as pd

from tsfm_mre.cli import main
from tsfm_mre.config import load_config
from tsfm_mre.pipeline import run
from tsfm_mre.provenance import file_digest
from tsfm_mre.schema import QUANTILE_COLUMNS, read_predictions, write_predictions

TABLES = ["metrics.csv", "risk.csv", "comparison_pairwise.csv", "comparison_joint.csv"]


def test_smoke_run_writes_complete_and_reproducible_outputs(smoke_config_path, tmp_path):
    config = load_config(smoke_config_path)
    first = run(config, tmp_path / "a", log=lambda m: None)
    second = run(config, tmp_path / "b", log=lambda m: None)

    out = first.out_dir
    for name in [*TABLES, "summary.md", "report.html", "manifest.json"]:
        assert (out / name).exists(), name
    figures = sorted(p.name for p in (out / "figures").glob("*.png"))
    assert figures == [
        "intervals.png",
        "pit.png",
        "rolling_exceedances.png",
        "scores_by_horizon.png",
    ]

    manifest = json.loads((out / "manifest.json").read_text())
    for rel, digest in manifest["outputs"].items():
        assert file_digest(out / rel) == digest
    assert manifest["data"]["panel_sha256"] == first.panel.digest()

    for name in [*TABLES, "predictions/ar1.csv.gz"]:
        assert (out / name).read_bytes() == (second.out_dir / name).read_bytes(), name
    for name in [*TABLES, "summary.md", "report.html", "manifest.json"]:
        assert b"\r\n" not in (out / name).read_bytes(), name

    risk = pd.read_csv(out / "risk.csv")
    assert set(risk["tail"]) == {"upper", "lower"}
    assert risk["es_p"].between(0, 1).all()
    joint = pd.read_csv(out / "comparison_joint.csv")
    assert set(joint["loss"]) == {"crps", "tail_ql"}
    summary = (out / "summary.md").read_text()
    assert "## Risk backtests" in summary and "(benchmark)" in summary


def test_external_predictions_join_the_evaluation(smoke_config_path, tmp_path):
    config = load_config(smoke_config_path)
    base = run(config, tmp_path / "base", log=lambda m: None)
    external = read_predictions(base.out_dir / "predictions" / "rw_ewma.csv.gz")
    external["model"] = "tsfm_demo"
    centre = external["q_0.5"].to_numpy()[:, None]
    external[list(QUANTILE_COLUMNS)] = centre + 1.1 * (
        external[list(QUANTILE_COLUMNS)].to_numpy() - centre
    )
    for col, q in [("es_0.01", "q_0.01"), ("es_0.025", "q_0.025")]:
        external[col] = np.minimum(external[col], external[q])
    for col, q in [("es_0.975", "q_0.975"), ("es_0.99", "q_0.99")]:
        external[col] = np.maximum(external[col], external[q])
    path = write_predictions(external, tmp_path / "tsfm_demo.csv.gz")

    result = run(config, tmp_path / "with_external", extra_predictions=[path], log=lambda m: None)
    assert "tsfm_demo" in set(result.tables.metrics["model"])
    pairwise = result.tables.pairwise
    assert "tsfm_demo" in set(pairwise["model"])
    assert (result.out_dir / "predictions" / "tsfm_demo.csv.gz").exists()


def test_manifest_command_has_no_local_paths(tmp_path):
    from tsfm_mre.provenance import command_line

    module = tmp_path / "site-packages" / "tsfm_mre" / "__main__.py"
    argv = [str(module), "run", "configs/smoke.yaml", "--out", str(tmp_path / "smoke")]
    assert command_line(argv) == "tsfm-mre run configs/smoke.yaml --out smoke"
    script = tmp_path / "bin" / "tsfm-mre"
    assert command_line([str(script), "validate", "x.csv"]) == "tsfm-mre validate x.csv"


def test_cli_run_and_validate(smoke_config_path, tmp_path, capsys):
    out = tmp_path / "cli"
    assert main(["run", str(smoke_config_path), "--out", str(out), "--quiet"]) == 0
    assert main(["validate", str(out / "predictions" / "hs.csv.gz")]) == 0
    assert "valid forecasts" in capsys.readouterr().out
    bad = tmp_path / "bad.csv"
    bad.write_text("model,series\nx,y\n")
    assert main(["validate", str(bad)]) == 2
