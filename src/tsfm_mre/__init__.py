"""TSFM-MRE: reproducible, risk-aware evaluation of probabilistic forecasts in finance."""

from tsfm_mre._version import __version__
from tsfm_mre.config import Config, load_config
from tsfm_mre.pipeline import RunResult, run
from tsfm_mre.schema import (
    ES_LEVELS,
    QUANTILE_LEVELS,
    read_predictions,
    summarize_samples,
    validate_predictions,
    write_predictions,
)

__all__ = [
    "ES_LEVELS",
    "QUANTILE_LEVELS",
    "Config",
    "RunResult",
    "__version__",
    "load_config",
    "read_predictions",
    "run",
    "summarize_samples",
    "validate_predictions",
    "write_predictions",
]
