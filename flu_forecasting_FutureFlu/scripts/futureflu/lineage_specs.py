"""Load FutureFlu lineage specs from config/futureflu/lineages.yaml.

English: Resolve lineage predictors and optional $FLU_FORECASTING_ROOT paths.
中文：解析各谱系预测因子，以及可选的 $FLU_FORECASTING_ROOT 路径。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "PyYAML is required to load config/futureflu/lineages.yaml "
        "(pip install pyyaml)."
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "futureflu" / "lineages.yaml"


def load_lineage_config(path: Path | None = None) -> dict[str, Any]:
    """Load lineages.yaml. / 加载 lineages.yaml。"""
    config_path = path or DEFAULT_CONFIG
    with config_path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"invalid lineage config: {config_path}")
    return data


def resolve_flu_forecasting_root() -> Path:
    """Locate an external blab/flu-forecasting checkout.

    English: Search FLU_FORECASTING_ROOT, then ../flu-forecasting next to this
    package, then optional local ./flu-forecasting (gitignored).
    中文：依次查找 FLU_FORECASTING_ROOT、包旁的 ../flu-forecasting，以及可选的
    本地 ./flu-forecasting（已 gitignore）。
    """
    env_value = os.environ.get("FLU_FORECASTING_ROOT", "").strip()
    candidates: list[Path] = []
    if env_value:
        candidates.append(Path(env_value).expanduser())
    candidates.extend(
        [
            PROJECT_ROOT.parent / "flu-forecasting",
            PROJECT_ROOT / "flu-forecasting",
        ]
    )
    for candidate in candidates:
        if (candidate / "scripts" / "frequencies.py").exists() and (candidate / "src").exists():
            return candidate.resolve()
    searched = ", ".join(str(path) for path in candidates) or "(none)"
    raise FileNotFoundError(
        "flu-forecasting upstream checkout not found. Clone "
        "https://github.com/blab/flu-forecasting and set FLU_FORECASTING_ROOT. "
        f"Searched: {searched}"
    )


def expand_path(value: str | Path, *, require_upstream: bool = False) -> Path:
    """Resolve package-relative paths or `$FLU_FORECASTING_ROOT/...` tokens.

    English: Expand upstream tokens before joining package-relative paths.
    中文：先展开上游路径占位符，再拼接包内相对路径。
    """
    text = str(value)
    if text.startswith("$FLU_FORECASTING_ROOT"):
        root = resolve_flu_forecasting_root()
        remainder = text[len("$FLU_FORECASTING_ROOT") :].lstrip("/").lstrip("\\")
        path = root / remainder if remainder else root
        return path.resolve()

    path = Path(text).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def resolved_lineages(config: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    """Return lineage specs with predictors and distance-map paths.

    English: Expand named predictor sets and resolve distance_map_root paths.
    中文：展开命名预测因子集合，并解析 distance_map_root 路径。
    """
    cfg = config or load_lineage_config()
    predictor_sets = cfg.get("predictors") or {}
    defaults = cfg.get("defaults") or {}
    out: dict[str, dict[str, Any]] = {}
    for key, raw in (cfg.get("lineages") or {}).items():
        spec = dict(raw)
        pred_key = spec.get("predictors")
        if isinstance(pred_key, str):
            if pred_key not in predictor_sets:
                raise KeyError(f"unknown predictors set {pred_key!r} for lineage {key}")
            spec["predictors"] = list(predictor_sets[pred_key])
        elif not isinstance(spec.get("predictors"), list):
            raise TypeError(f"lineage {key} predictors must be a list or named set")

        spec["distance_map_root"] = expand_path(spec["distance_map_root"])

        for field in ("global_start", "issue_start", "issue_end", "forecast_years"):
            if field not in spec and field in defaults:
                spec[field] = defaults[field]
        out[key] = spec
    return out


def workflow_defaults(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return workflow defaults from lineages.yaml. / 返回 lineages.yaml 中的流程默认项。"""
    cfg = config or load_lineage_config()
    return dict(cfg.get("defaults") or {})
