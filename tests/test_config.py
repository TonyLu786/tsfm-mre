import copy

import pytest
import yaml

from tests.conftest import CONFIGS
from tsfm_mre.config import ConfigError, load_config, parse_config


@pytest.mark.parametrize("name", ["smoke", "synthetic", "fred_h15"])
def test_shipped_configurations_are_valid(name):
    config = load_config(CONFIGS / f"{name}.yaml")
    assert config.name == name
    assert config.comparison.benchmark in config.model_names


def _raw():
    return yaml.safe_load((CONFIGS / "smoke.yaml").read_text())


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda r: r.update(extra=1), "unknown top-level"),
        (lambda r: r["data"].update(sources="fred"), "unknown keys"),
        (lambda r: r["data"].update(source="excel"), "data.source"),
        (lambda r: r["comparison"].update(benchmark="tsfm"), "benchmark"),
        (lambda r: r["risk"].update(horizon=10), "risk.horizon"),
        (lambda r: r["risk"].update(var_alpha=0.03), "needs quantiles"),
        (lambda r: r["risk"].update(es_alpha=0.05), "es_alpha"),
        (lambda r: r["evaluation"].update(start="2020-01-02"), "inside the data sample"),
        (lambda r: r["models"].append({"name": "ar1"}), "unique"),
        (lambda r: r["models"].append({"name": "lstm"}), "unknown model"),
    ],
)
def test_invalid_configurations(change, message):
    raw = copy.deepcopy(_raw())
    change(raw)
    with pytest.raises(ConfigError, match=message):
        parse_config(raw)


def test_focus_series_defaults_to_last_series():
    raw = _raw()
    raw.pop("report")
    assert parse_config(raw).report.focus_series == "DGS10"
