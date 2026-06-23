"""涨跌规律可调参数（由 pattern_param_search 网格寻优产出）。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OPTIMIZED = ROOT / "research" / "pattern_rules_optimized.json"

# 规则 → 评估 horizon（寻优产出会写入 rule_horizons 覆盖）
DEFAULT_RULE_HORIZONS: dict[str, str] = {
    "D1_blowoff": "5d",
    "D2_dump": "5d",
    "D3_shrink": "20d",
    "D4_mega": "20d",
    "D4_off_ma": "20d",
    "D_parabolic": "20d",
    "D_outflow": "5d",
}

SHORT_TERM_AVOID_RULES = ("D1_blowoff", "D2_dump", "D_outflow")

CANON_AVOID_RULE: dict[str, str] = {
    "D3_shrink_vol_top": "D3_shrink",
    "D4_extended_mega": "D4_mega",
    "D4_extended_off_ma": "D4_off_ma",
    "D1_vol_blowoff": "D1_blowoff",
    "D2_vol_dump": "D2_dump",
    "D_outflow": "D_outflow",
    "D_parabolic_20d": "D_parabolic",
    "D_parabolic_5d": "D_parabolic",
}


def canon_avoid_rule(rule_id: str) -> str:
    return CANON_AVOID_RULE.get(rule_id, rule_id)


@dataclass
class LongParams:
    """腿① 做多：次日胜率优化参数（真实量价）。"""

    min_gain_pct: float = 2.5
    max_gain_pct: float = 4.5
    min_vol_ratio: float = 1.3
    max_vol_ratio: float = 1.65
    min_close_strength: float = 0.60
    min_ret_5d_pct: float = 4.0
    max_ret_5d_pct: float = 15.0
    min_gain_20d_pct: float = 4.0
    max_gain_20d_pct: float = 22.0
    min_rs_spy_20d_pct: float = 3.0
    max_rs_spy_20d_pct: float = 18.0
    min_dvol_m: float = 500.0
    min_setup_win_rate: float = 0.65
    min_setup_samples: int = 5
    require_above_ma50: bool = True
    require_spy_ma20: bool = True
    require_spy_positive_5d: bool = True
    require_spy_positive_1d: bool = False
    min_spy_1d_pct: float = 0.0
    require_green_candle: bool = True
    top_n: int = 2

    def to_gainer_filters_kwargs(self) -> dict[str, Any]:
        from research.gainer_daily_backtest import GainerProFilters

        return {
            "min_gain_pct": self.min_gain_pct,
            "max_gain_pct": self.max_gain_pct,
            "min_vol_ratio": self.min_vol_ratio,
            "max_vol_ratio": self.max_vol_ratio,
            "min_close_strength": self.min_close_strength,
            "min_gain_20d_pct": self.min_gain_20d_pct,
            "max_gain_20d_pct": self.max_gain_20d_pct,
            "min_rs_20d_pct": self.min_rs_spy_20d_pct,
            "max_rs_20d_pct": self.max_rs_spy_20d_pct,
            "min_setup_win_rate": self.min_setup_win_rate,
            "min_setup_samples": self.min_setup_samples,
            "require_above_ma50": self.require_above_ma50,
            "require_spy_above_ma20": self.require_spy_ma20,
            "require_spy_positive_5d": self.require_spy_positive_5d,
            "require_spy_positive_1d": self.require_spy_positive_1d,
            "min_spy_1d_pct": self.min_spy_1d_pct,
            "require_green_candle": self.require_green_candle,
            "top_n": self.top_n,
            "min_candidates": 1,
            "use_recent_setup_win": True,
        }

    def build_gainer_filters(self):
        from research.gainer_daily_backtest import GainerProFilters

        return GainerProFilters(**self.to_gainer_filters_kwargs())

    def mask_on_panel(self, p: pd.DataFrame) -> pd.Series:
        """高置信日频面板的布尔掩码。"""
        gain = pd.to_numeric(p["涨幅%"], errors="coerce")
        vr = pd.to_numeric(p["量比"], errors="coerce")
        cs = pd.to_numeric(p["收盘强度"], errors="coerce")
        r5 = pd.to_numeric(p.get("ret_5d", p.get("涨幅20d%", 0)), errors="coerce")
        if "ret_5d" not in p.columns and "涨幅20d%" in p.columns:
            r5 = pd.to_numeric(p["涨幅20d%"], errors="coerce") / 100.0
        r20 = pd.to_numeric(p.get("ret_20d", 0), errors="coerce")
        if "ret_20d" not in p.columns and "涨幅20d%" in p.columns:
            r20 = pd.to_numeric(p["涨幅20d%"], errors="coerce") / 100.0
        rs = pd.to_numeric(p.get("相对SPY20d%", 0), errors="coerce")
        dvol = pd.to_numeric(p.get("成交额USD", 0), errors="coerce") / 1e6
        wr8 = pd.to_numeric(p.get("近8次胜率", 1), errors="coerce")
        n8 = pd.to_numeric(p.get("近8次样本", 0), errors="coerce")
        ma50 = p.get("站上MA50", True)
        if ma50.dtype == object:
            ma50 = ma50.astype(bool)
        spy_ma = p.get("SPY站上MA20", True)
        spy5 = pd.to_numeric(p.get("SPY5d涨%", 0), errors="coerce")
        spy1 = pd.to_numeric(p.get("SPY1d涨%", 0), errors="coerce")
        green = p.get("收阳", True)
        if green.dtype == object:
            green = green.astype(bool)

        m = (
            (gain >= self.min_gain_pct)
            & (gain <= self.max_gain_pct)
            & (vr >= self.min_vol_ratio)
            & (vr <= self.max_vol_ratio)
            & (cs >= self.min_close_strength)
            & (r5 * 100 >= self.min_ret_5d_pct)
            & (r5 * 100 <= self.max_ret_5d_pct)
            & (r20 * 100 >= self.min_gain_20d_pct)
            & (r20 * 100 <= self.max_gain_20d_pct)
            & (rs >= self.min_rs_spy_20d_pct)
            & (rs <= self.max_rs_spy_20d_pct)
            & (dvol >= self.min_dvol_m)
            & (wr8 >= self.min_setup_win_rate)
            & (n8 >= self.min_setup_samples)
        )
        if self.require_above_ma50:
            m &= ma50.astype(bool)
        if self.require_spy_ma20:
            m &= spy_ma.astype(bool)
        if self.require_spy_positive_5d:
            m &= spy5 > 0
        if self.require_spy_positive_1d:
            m &= spy1 >= self.min_spy_1d_pct
        if self.require_green_candle:
            m &= green.astype(bool)
        return m


@dataclass
class DownParams:
    """腿② 回避：后 20 日下跌率优化阈值。"""

    parabolic_ret_5d: float = 0.15
    parabolic_ret_20d: float = 0.40
    shrink_vol_max: float = 1.0
    shrink_vol_min_ret_5d: float = 0.12
    vol_dump_min_ratio: float = 2.5
    vol_dump_max_ret_5d: float = -0.05
    blowoff_min_ratio: float = 2.5
    blowoff_ret_5d_min: float = 0.05
    blowoff_ret_5d_max: float = 0.15
    blowoff_dvol_min_m: float = 200.0
    blowoff_dvol_max_m: float = 1000.0
    weak_close_max: float = 0.35
    weak_close_min_vol: float = 1.5
    mega_dvol_m: float = 500.0
    mega_ret_5d: float = 0.15
    extended_off_ma_ret_5d: float = 0.15
    extended_off_ma_max_vol: float = 1.5
    active_avoid_rules: list[str] = field(default_factory=lambda: [
        "D4_mega", "D1_blowoff", "D2_dump", "D_parabolic",
    ])
    rule_horizons: dict[str, str] = field(default_factory=dict)
    short_term_avoid_rules: list[str] = field(
        default_factory=lambda: list(SHORT_TERM_AVOID_RULES),
    )
    income_conflict_min_ret_20d: float = 0.35

    def horizon_for(self, rule_id: str) -> str:
        canon = canon_avoid_rule(rule_id)
        if self.rule_horizons and canon in self.rule_horizons:
            return self.rule_horizons[canon]
        return DEFAULT_RULE_HORIZONS.get(canon, "20d")

    def is_short_term(self, rule_id: str) -> bool:
        return canon_avoid_rule(rule_id) in self.short_term_avoid_rules

    def describe(self) -> str:
        st = ",".join(self.short_term_avoid_rules[:2]) if self.short_term_avoid_rules else "—"
        return (
            f"抛物线5d>{self.parabolic_ret_5d:.0%}/20d>{self.parabolic_ret_20d:.0%} · "
            f"缩量顶<{self.shrink_vol_max} · 放量跌>{self.vol_dump_min_ratio} · "
            f"短周期规则={st}"
        )


@dataclass
class OptimizedPatternRules:
    long: LongParams = field(default_factory=LongParams)
    down: DownParams = field(default_factory=DownParams)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "long": asdict(self.long),
            "down": asdict(self.down),
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict) -> OptimizedPatternRules:
        return cls(
            long=LongParams(**(d.get("long") or {})),
            down=DownParams(**(d.get("down") or {})),
            meta=d.get("meta") or {},
        )


def load_optimized_rules(path: Path | None = None) -> OptimizedPatternRules:
    p = path or DEFAULT_OPTIMIZED
    if not p.exists():
        return OptimizedPatternRules()
    return OptimizedPatternRules.from_dict(json.loads(p.read_text(encoding="utf-8")))


def save_optimized_rules(rules: OptimizedPatternRules, path: Path | None = None) -> Path:
    p = path or DEFAULT_OPTIMIZED
    p.write_text(json.dumps(rules.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return p
