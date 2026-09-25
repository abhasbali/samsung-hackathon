"""Optional MLflow experiment tracking. A no-op when disabled or when mlflow is missing."""

from __future__ import annotations

from typing import Any

from codefusion.config import CodeFusionConfig, config_to_dict
from codefusion.logging_utils import get_logger

log = get_logger(__name__)


def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = str(v)[:500]
    return out


def log_run(cfg: CodeFusionConfig, run_name: str, metrics: dict[str, Any], extra: dict[str, Any] | None = None) -> bool:
    """Log params/metrics to MLflow if ``tracking.mlflow`` is enabled. Returns True if logged."""
    if not cfg.tracking.mlflow:
        return False
    try:
        import mlflow
    except ImportError:
        log.warning("tracking.mlflow=true but mlflow is not installed; skipping")
        return False
    try:
        if cfg.tracking.tracking_uri:
            mlflow.set_tracking_uri(cfg.tracking.tracking_uri)
        mlflow.set_experiment(cfg.tracking.experiment)
        with mlflow.start_run(run_name=run_name):
            params = _flatten(config_to_dict(cfg))
            # MLflow caps params per batch; log in chunks.
            items = list(params.items())
            for i in range(0, len(items), 90):
                mlflow.log_params(dict(items[i : i + 90]))
            mlflow.set_tag("enabled_components", ",".join(cfg.enabled_components()))
            mlflow.set_tag("model", cfg.dense.model)
            for k, v in (extra or {}).items():
                mlflow.set_tag(k, str(v))
            mlflow.log_metrics({k: float(v) for k, v in metrics.items() if isinstance(v, (int, float)) and v is not None})
        return True
    except Exception as exc:  # noqa: BLE001
        log.warning("mlflow logging failed", extra={"data": {"error": repr(exc)[:200]}})
        return False
