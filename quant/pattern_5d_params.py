"""5 日路径规律可调参数（move_pattern_5d_param_search 产出）。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_5D_OPTIMIZED = ROOT / "research" / "move_pattern_5d_optimized.json"


@dataclass
class LiquidityFilter:
    min_dvol_m: float = 50.0
    min_avg_dvol_m: float = 30.0
    min_vol_ratio: float = 1.0
    min_turnover_pct: float = 0.3
    require_turnover: bool = True


@dataclass
class PathThreshold:
    up_pct: float = 3.0
    down_pct: float = 3.0
    horizon: int = 5


@dataclass
class ExtendedUpFilters:
    min_vol_ratio: float = 2.5
    min_ret_5d: float = 0.15
    min_close_strength: float = 0.60
    min_turnover_pct: float = 0.5
    min_dvol_m: float = 200.0
    require_above_ma50: bool = True
    require_above_ma20: bool = True
    min_up_vol_share: float = 0.0


@dataclass
class ExtendedDownFilters:
    min_vol_ratio: float = 2.5
    max_ret_5d: float = -0.05
    max_close_strength: float = 0.40
    min_turnover_pct: float = 0.5
    min_dvol_m: float = 50.0
    require_below_ma50: bool = True


@dataclass
class Optimized5dRules:
    liquidity: LiquidityFilter = field(default_factory=LiquidityFilter)
    threshold: PathThreshold = field(default_factory=PathThreshold)
    up: ExtendedUpFilters = field(default_factory=ExtendedUpFilters)
    down: ExtendedDownFilters = field(default_factory=ExtendedDownFilters)
    min_samples: int = 60
    min_hit_is: float = 0.68
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "liquidity": asdict(self.liquidity),
            "threshold": asdict(self.threshold),
            "up": asdict(self.up),
            "down": asdict(self.down),
            "min_samples": self.min_samples,
            "min_hit_is": self.min_hit_is,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Optimized5dRules:
        return cls(
            liquidity=LiquidityFilter(**(d.get("liquidity") or {})),
            threshold=PathThreshold(**(d.get("threshold") or {})),
            up=ExtendedUpFilters(**(d.get("up") or {})),
            down=ExtendedDownFilters(**(d.get("down") or {})),
            min_samples=int(d.get("min_samples", 60)),
            min_hit_is=float(d.get("min_hit_is", 0.68)),
            meta=d.get("meta") or {},
        )


def load_optimized_5d(path: Path | None = None) -> Optimized5dRules:
    p = path or DEFAULT_5D_OPTIMIZED
    if not p.exists():
        return Optimized5dRules()
    return Optimized5dRules.from_dict(json.loads(p.read_text(encoding="utf-8-sig")))


def save_optimized_5d(rules: Optimized5dRules, path: Path | None = None) -> Path:
    p = path or DEFAULT_5D_OPTIMIZED
    p.write_text(json.dumps(rules.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def up_mask(df, p: ExtendedUpFilters) -> "pd.Series":
    import pandas as pd

    turn = pd.to_numeric(df.get("换手率%"), errors="coerce")
    m = (
        (pd.to_numeric(df["vol_ratio"], errors="coerce") >= p.min_vol_ratio)
        & (pd.to_numeric(df["ret_5d"], errors="coerce") >= p.min_ret_5d)
        & (pd.to_numeric(df["close_strength"], errors="coerce") >= p.min_close_strength)
        & (pd.to_numeric(df["dvol_m"], errors="coerce") >= p.min_dvol_m)
        & (turn >= p.min_turnover_pct)
    )
    if p.require_above_ma50:
        m &= df["above_ma50"].astype(bool)
    if p.require_above_ma20:
        m &= df["above_ma20"].astype(bool)
    if p.min_up_vol_share > 0:
        m &= pd.to_numeric(df.get("up_vol_share", 0), errors="coerce") >= p.min_up_vol_share
    return m


def down_mask(df, p: ExtendedDownFilters) -> "pd.Series":
    import pandas as pd

    turn = pd.to_numeric(df.get("换手率%"), errors="coerce")
    m = (
        (pd.to_numeric(df["vol_ratio"], errors="coerce") >= p.min_vol_ratio)
        & (pd.to_numeric(df["ret_5d"], errors="coerce") <= p.max_ret_5d)
        & (pd.to_numeric(df["close_strength"], errors="coerce") <= p.max_close_strength)
        & (pd.to_numeric(df["dvol_m"], errors="coerce") >= p.min_dvol_m)
        & (turn >= p.min_turnover_pct)
    )
    if p.require_below_ma50:
        m &= ~df["above_ma50"].astype(bool)
    return m
