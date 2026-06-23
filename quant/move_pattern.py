"""涨跌前的资金轨迹特征：成交额、量比、涨幅等（无未来函数）。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class MovePatternRule:
    id: str
    direction: str  # up | down
    description: str
    conditions: dict[str, Any]
    sample_n: int
    fwd_mean: float
    win_rate: float
    median_fwd: float
    action: str
    win_horizon: str = "20d"  # 1d=次日 | 5d | 20d
    tier: str = "B"  # S=高置信 A/B/C

    def to_dict(self) -> dict:
        hz = self.win_horizon
        return {
            "id": self.id,
            "direction": self.direction,
            "pattern": self.description,
            "conditions": self.conditions,
            "sample_n": self.sample_n,
            "win_horizon": hz,
            "win_label": "次日胜率" if hz == "1d" else f"后{hz}胜率",
            "fwd_mean": round(self.fwd_mean, 4),
            "fwd_20d_mean": round(self.fwd_mean, 4),
            "fwd_median": round(self.median_fwd, 4),
            "fwd_20d_median": round(self.median_fwd, 4),
            "win_rate": round(self.win_rate, 4),
            "tier": self.tier,
            "action": self.action,
        }


def _dollar_vol(close: pd.Series, volume: pd.Series) -> pd.Series:
    return close.astype(float) * volume.astype(float)


def extract_trajectory_features(df: pd.DataFrame, *, forward_days: int = 20) -> pd.DataFrame:
    """逐日计算「上涨/下跌前」可观测轨迹特征 + 后向标签（仅用于挖掘）。"""
    if df is None or df.empty or len(df) < 80:
        return pd.DataFrame()

    close = df["Close"].astype(float)
    high = df["High"].astype(float) if "High" in df.columns else close
    low = df["Low"].astype(float) if "Low" in df.columns else close
    vol = df["Volume"].astype(float)
    dvol = _dollar_vol(close, vol)

    vol_ma20 = vol.rolling(20, min_periods=10).mean()
    dvol_ma20 = dvol.rolling(20, min_periods=10).mean()
    ma20 = close.rolling(20, min_periods=10).mean()
    ma50 = close.rolling(50, min_periods=25).mean()

    ret_1d = close.pct_change()
    ret_5d = close.pct_change(5)
    ret_20d = close.pct_change(20)

    vol_ratio = vol / vol_ma20.replace(0, np.nan)
    vol_trend_5_20 = vol.rolling(5, min_periods=3).mean() / vol_ma20.replace(0, np.nan)
    dvol_m = dvol / 1e6

    hl = (high - low).replace(0, np.nan)
    close_strength = ((close - low) / hl).clip(0, 1)

    up_mask = ret_1d > 0
    up_vol_10 = vol.where(up_mask).rolling(10, min_periods=5).sum()
    tot_vol_10 = vol.rolling(10, min_periods=5).sum()
    up_vol_share = up_vol_10 / tot_vol_10.replace(0, np.nan)

    # 后向收益（挖掘标签，实盘不可用）
    fwd = close.shift(-forward_days) / close - 1.0

    out = pd.DataFrame({
        "日期": close.index,
        "收盘价": close.values,
        "ret_1d": ret_1d.values,
        "ret_5d": ret_5d.values,
        "ret_20d": ret_20d.values,
        "vol_ratio": vol_ratio.values,
        "vol_trend_5_20": vol_trend_5_20.values,
        "dvol_m": dvol_m.values,
        "dvol_ma20_m": (dvol_ma20 / 1e6).values,
        "above_ma20": (close > ma20).values,
        "above_ma50": (close > ma50).values,
        "close_strength": close_strength.values,
        "up_vol_share": up_vol_share.values,
        f"fwd_{forward_days}d": fwd.values,
    })
    out = out.dropna(subset=["vol_ratio", f"fwd_{forward_days}d"])
    return out


def compute_forward_path_labels(
    df: pd.DataFrame,
    *,
    horizon: int = 5,
    up_threshold: float = 0.03,
    down_threshold: float = 0.03,
) -> pd.DataFrame:
    """未来 horizon 日内路径涨/跌标签（真实 High/Low，无 BS）。

    path_up: 自信号日收盘起，未来 horizon 日内最高价相对涨幅
    path_down: 未来 horizon 日内最低价相对跌幅（≤0）
    hit_up / hit_down: 路径是否达到阈值比例
    fwd_close: horizon 日收盘涨跌幅
    """
    if df is None or df.empty:
        return pd.DataFrame()
    close = df["Close"].astype(float)
    high = df["High"].astype(float) if "High" in df.columns else close
    low = df["Low"].astype(float) if "Low" in df.columns else close
    n = len(close)
    path_up = np.full(n, np.nan)
    path_down = np.full(n, np.nan)
    fwd_close = np.full(n, np.nan)
    for i in range(n):
        j = i + 1
        k = min(n, i + 1 + horizon)
        if j >= k:
            continue
        base = float(close.iloc[i])
        if base <= 0:
            continue
        path_up[i] = float(high.iloc[j:k].max() / base - 1.0)
        path_down[i] = float(low.iloc[j:k].min() / base - 1.0)
        if i + horizon < n:
            fwd_close[i] = float(close.iloc[i + horizon] / base - 1.0)
    return pd.DataFrame({
        f"path_up_{horizon}d": path_up,
        f"path_down_{horizon}d": path_down,
        f"fwd_{horizon}d": fwd_close,
        f"hit_up_{horizon}d": path_up >= up_threshold,
        f"hit_down_{horizon}d": path_down <= -down_threshold,
    }, index=df.index)


def extract_trajectory_features_5d(
    df: pd.DataFrame,
    *,
    shares_out: float | None = None,
    horizon: int = 5,
    up_threshold: float = 0.03,
    down_threshold: float = 0.03,
    forward_days: int = 20,
) -> pd.DataFrame:
    """轨迹特征 + 5 日路径标签 + 换手率（真实 OHLCV）。"""
    base = extract_trajectory_features(df, forward_days=forward_days)
    if base.empty:
        return base
    paths = compute_forward_path_labels(
        df, horizon=horizon, up_threshold=up_threshold, down_threshold=down_threshold,
    )
    paths = paths.copy()
    paths["日期"] = pd.to_datetime(paths.index)
    base = base.copy()
    base["日期"] = pd.to_datetime(base["日期"])
    out = base.merge(paths, on="日期", how="inner")
    if shares_out and shares_out > 0:
        vol_s = df["Volume"].astype(float)
        turn = []
        for d in out["日期"]:
            try:
                v = float(vol_s.loc[d]) if d in vol_s.index else np.nan
            except (KeyError, TypeError):
                v = np.nan
            turn.append(v / float(shares_out) * 100.0 if np.isfinite(v) else np.nan)
        out["换手率%"] = turn
    else:
        out["换手率%"] = np.nan
    out["shares_out"] = shares_out
    pu = f"path_up_{horizon}d"
    pd_col = f"path_down_{horizon}d"
    out["强涨"] = out.get(f"hit_up_{horizon}d", False)
    out["强跌"] = out.get(f"hit_down_{horizon}d", False)
    return out.dropna(subset=["vol_ratio", pu, pd_col])


def _bucket_vol_ratio(v: float) -> str:
    if v < 1.0:
        return "<1.0"
    if v < 1.5:
        return "1.0-1.5"
    if v < 2.5:
        return "1.5-2.5"
    return ">2.5"


def _bucket_ret5(v: float) -> str:
    if v < -0.05:
        return "<-5%"
    if v < 0:
        return "-5~0%"
    if v < 0.05:
        return "0~5%"
    if v < 0.15:
        return "5~15%"
    return ">15%"


def _bucket_dvol(v: float) -> str:
    if v < 50:
        return "<50M"
    if v < 200:
        return "50-200M"
    if v < 1000:
        return "200M-1B"
    return ">1B"


def enrich_buckets(feat: pd.DataFrame) -> pd.DataFrame:
    f = feat.copy()
    f["vol_ratio桶"] = f["vol_ratio"].map(_bucket_vol_ratio)
    f["ret_5d桶"] = f["ret_5d"].map(_bucket_ret5)
    f["dvol桶"] = f["dvol_m"].map(_bucket_dvol)
    f["强涨"] = f["fwd_20d"] >= 0.10
    f["强跌"] = f["fwd_20d"] <= -0.10
    return f


def mine_rules_from_panel(
    panel: pd.DataFrame,
    *,
    min_samples: int = 40,
    min_win_rate: float = 0.58,
    forward_col: str = "fwd_20d",
) -> list[MovePatternRule]:
    """从全市场事件面板挖掘可重复规则（网格 + 单因子）。"""
    if panel.empty:
        return []

    rules: list[MovePatternRule] = []
    rid = 0

    def _add(direction: str, sub: pd.DataFrame, desc: str, action: str) -> None:
        nonlocal rid
        if len(sub) < min_samples:
            return
        fwd = sub[forward_col]
        win = fwd > 0 if direction == "up" else fwd < 0
        wr = float(win.mean())
        if wr < min_win_rate:
            return
        rid += 1
        rules.append(MovePatternRule(
            id=f"{direction}_{rid}",
            direction=direction,
            description=desc,
            conditions={"note": desc},
            sample_n=len(sub),
            fwd_mean=float(fwd.mean()),
            win_rate=wr,
            median_fwd=float(fwd.median()),
            action=action,
        ))

    _add("up", panel[panel["vol_ratio"] >= 2.5], "量比>2.5（资金明显介入）", "堆量后关注突破")
    _add("down", panel[(panel["vol_ratio"] >= 2.5) & (panel["ret_5d"] < 0)], "量比>2.5 + 5日已跌", "放量下跌 → 回避")
    _add("up", panel[(panel["dvol_m"] >= 1000) & (panel["ret_5d"] > 0)], "成交额>1B + 5日正动量", "大盘子正动量")
    _add("up", panel[(panel["close_strength"] >= 0.65) & (panel["vol_ratio"] >= 1.5)],
         "收在日内高位(>65%) + 量比≥1.5", "尾盘抢筹 → 延续概率高")
    _add("down", panel[(panel["close_strength"] <= 0.35) & (panel["vol_ratio"] >= 1.5) & (panel["ret_5d"] < 0)],
         "收在日内低位 + 放量 + 5日跌", "出货形态 → 回避")

    vol_buckets = ["1.0-1.5", "1.5-2.5", ">2.5"]
    ret_buckets_up = ["0~5%", "5~15%", ">15%"]
    ret_buckets_down = ["-5~0%", "<-5%"]
    dvol_buckets = ["50-200M", "200M-1B", ">1B"]

    for direction, ret_bs in [("up", ret_buckets_up), ("down", ret_buckets_down)]:
        for vb in vol_buckets:
            for rb in ret_bs:
                for db in dvol_buckets:
                    for ma in [True, False]:
                        sub = panel[
                            (panel["vol_ratio桶"] == vb)
                            & (panel["ret_5d桶"] == rb)
                            & (panel["dvol桶"] == db)
                            & (panel["above_ma50"] == ma)
                        ]
                        if len(sub) < min_samples:
                            continue
                        fwd = sub[forward_col]
                        win = fwd > 0 if direction == "up" else fwd < 0
                        wr = float(win.mean())
                        if wr < min_win_rate:
                            continue
                        rid += 1
                        tag = "涨幅" if direction == "up" else "跌幅"
                        desc = (
                            f"量比{vb} · 5日{tag}{rb} · 成交额{db}"
                            + (" · 站上MA50" if ma else " · MA50下方")
                        )
                        action = (
                            "资金堆量+正动量 → 做多"
                            if direction == "up"
                            else "放量走弱 → 回避/卖Put"
                        )
                        rules.append(MovePatternRule(
                            id=f"{direction}_{rid}",
                            direction=direction,
                            description=desc,
                            conditions={
                                "vol_ratio_bucket": vb,
                                "ret_5d_bucket": rb,
                                "dvol_bucket": db,
                                "above_ma50": ma,
                            },
                            sample_n=len(sub),
                            fwd_mean=float(fwd.mean()),
                            win_rate=wr,
                            median_fwd=float(fwd.median()),
                            action=action,
                        ))

    rules.sort(key=lambda r: (-r.sample_n, -r.win_rate))
    seen: set[str] = set()
    deduped: list[MovePatternRule] = []
    for r in rules:
        key = f"{r.direction}|{r.description}"
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)
    return deduped[:32]


def enrich_forward_horizons(panel: pd.DataFrame) -> pd.DataFrame:
    """为事件面板补充次日/5日标签。"""
    if panel.empty or "代码" not in panel.columns:
        return panel
    p = panel.copy()
    p["日期"] = pd.to_datetime(p["日期"])
    p = p.sort_values(["代码", "日期"])
    for h in (1, 5):
        col = f"fwd_{h}d"
        if col not in p.columns:
            p[col] = p.groupby("代码")["收盘价"].pct_change(h).shift(-h)
    return p


def _rule_from_mask(
    panel: pd.DataFrame,
    mask: pd.Series,
    *,
    desc: str,
    direction: str,
    horizon: str,
    min_samples: int,
    min_win_rate: float,
    tier: str,
    action: str,
    rule_id: str,
    extra_cond: dict | None = None,
) -> MovePatternRule | None:
    fwd_col = f"fwd_{horizon}d" if horizon in ("1", "5", "20") else "fwd_20d"
    if fwd_col not in panel.columns:
        return None
    sub = panel[mask].dropna(subset=[fwd_col])
    if len(sub) < min_samples:
        return None
    fwd = sub[fwd_col]
    win = fwd > 0 if direction == "up" else fwd < 0
    wr = float(win.mean())
    if wr < min_win_rate:
        return None
    cond = dict(extra_cond or {})
    cond["mask_desc"] = desc
    return MovePatternRule(
        id=rule_id,
        direction=direction,
        description=desc,
        conditions=cond,
        sample_n=len(sub),
        fwd_mean=float(fwd.mean()),
        win_rate=wr,
        median_fwd=float(fwd.median()),
        action=action,
        win_horizon=f"{horizon}d" if horizon in ("1", "5", "20") else horizon,
        tier=tier,
    )


def mine_high_win_rules(
    panel: pd.DataFrame,
    *,
    min_samples: int = 40,
    min_win_rate: float = 0.65,
) -> list[MovePatternRule]:
    """高置信规律：严格量价模板 + 次日/5日胜率（全市场日频面板）。"""
    p = enrich_forward_horizons(panel)
    if p.empty:
        return []

    # 统一列名（gainer panel 与 trajectory panel）
    if "涨幅%" in p.columns and "ret_1d" not in p.columns:
        p["ret_1d"] = pd.to_numeric(p["涨幅%"], errors="coerce") / 100.0
    if "量比" in p.columns and "vol_ratio" not in p.columns:
        p["vol_ratio"] = pd.to_numeric(p["量比"], errors="coerce")
    if "收盘强度" in p.columns and "close_strength" not in p.columns:
        p["close_strength"] = pd.to_numeric(p["收盘强度"], errors="coerce")
    if "成交额USD" in p.columns and "dvol_m" not in p.columns:
        p["dvol_m"] = pd.to_numeric(p["成交额USD"], errors="coerce") / 1e6
    if "站上MA50" in p.columns and "above_ma50" not in p.columns:
        p["above_ma50"] = p["站上MA50"].astype(bool)
    if "站上MA20" in p.columns and "above_ma20" not in p.columns:
        p["above_ma20"] = p["站上MA20"].astype(bool)
    if "fwd_1d" not in p.columns and "收盘价" in p.columns:
        p = enrich_forward_horizons(p)

    rules: list[MovePatternRule] = []
    templates: list[tuple[str, pd.Series, str, str]] = [
        (
            "S1·温和涨3-4% + 量比1.3-1.5 + 收强≥70% + 5日涨5-12% + 成交额>10亿 + MA50",
            (
                (p["vol_ratio"] >= 1.3) & (p["vol_ratio"] <= 1.5)
                & (p["ret_1d"] >= 0.03) & (p["ret_1d"] <= 0.04)
                & (p["close_strength"] >= 0.70)
                & (p["ret_5d"] >= 0.05) & (p["ret_5d"] <= 0.12)
                & (p["dvol_m"] >= 1000)
                & (p["above_ma50"])
            ),
            "S",
            "堆量温和突破 → 次日延续（严格）",
        ),
        (
            "S2·涨2.5-5% + 量比1.3-1.75 + 收强≥55% + MA50 + 近8次形态胜率≥62.5%",
            (
                (p["vol_ratio"] >= 1.3) & (p["vol_ratio"] <= 1.75)
                & (p["ret_1d"] >= 0.025) & (p["ret_1d"] <= 0.055)
                & (p["close_strength"] >= 0.55)
                & (p["above_ma50"])
                & (pd.to_numeric(p.get("近8次胜率", p.get("setup_wr8", 1)), errors="coerce") >= 0.625)
            ),
            "S",
            "历史形态验证通过 → 高置信做多",
        ),
        (
            "A1·涨3-5% + 量比1.3-1.5 + 收强≥65% + 成交额>5亿 + MA50",
            (
                (p["vol_ratio"] >= 1.3) & (p["vol_ratio"] <= 1.5)
                & (p["ret_1d"] >= 0.03) & (p["ret_1d"] <= 0.05)
                & (p["close_strength"] >= 0.65)
                & (p["dvol_m"] >= 500)
                & (p["above_ma50"])
            ),
            "A",
            "大盘子温和放量 → 做多",
        ),
        (
            "A2·涨2.5-5% + 量比1.3-1.75 + 20日涨4-25% + MA50 + 收阳",
            (
                (p["vol_ratio"] >= 1.3) & (p["vol_ratio"] <= 1.75)
                & (p["ret_1d"] >= 0.025) & (p["ret_1d"] <= 0.055)
                & (p["ret_20d"] >= 0.04) & (p["ret_20d"] <= 0.25)
                & (p["above_ma50"])
                & (p.get("收阳", p["ret_1d"] > 0))
            ),
            "A",
            "趋势中继放量 → 做多",
        ),
        (
            "B1·放量>2.5 + 5日跌>5% + 成交额>10亿 + MA50（回避）",
            (
                (p["vol_ratio"] >= 2.5)
                & (p["ret_5d"] < -0.05)
                & (p["dvol_m"] >= 1000)
                & (p["above_ma50"])
            ),
            "A",
            "放量杀跌 → 回避/偏空",
        ),
    ]

    if "ret_5d" not in p.columns:
        p["ret_5d"] = p.groupby("代码")["收盘价"].pct_change(5)
    if "ret_20d" not in p.columns:
        p["ret_20d"] = p.groupby("代码")["收盘价"].pct_change(20)

    rid = 0
    for desc, mask, tier, action in templates:
        direction = "down" if "回避" in action or "偏空" in action else "up"
        for hz in ("1", "5"):
            rid += 1
            r = _rule_from_mask(
                p, mask,
                desc=desc, direction=direction, horizon=hz,
                min_samples=min_samples, min_win_rate=min_win_rate,
                tier=tier, action=action, rule_id=f"hw_{rid}_{hz}d",
            )
            if r:
                rules.append(r)

    rules.sort(key=lambda x: (-{"S": 3, "A": 2, "B": 1}.get(x.tier, 0), -x.win_rate, -x.sample_n))
    return rules[:16]


def match_rule(row: pd.Series, rule: MovePatternRule | dict) -> bool:
    cond = rule["conditions"] if isinstance(rule, dict) else rule.conditions
    if "note" in cond and len(cond) == 1:
        return False  # 单因子规则需专用匹配，见 match_rule_fuzzy
    if row.get("vol_ratio桶") != cond.get("vol_ratio_bucket"):
        return False
    if row.get("ret_5d桶") != cond.get("ret_5d_bucket"):
        return False
    if row.get("dvol桶") != cond.get("dvol_bucket"):
        return False
    if bool(row.get("above_ma50")) != cond.get("above_ma50"):
        return False
    return True


def match_rule_fuzzy(row: pd.Series, rule: dict) -> bool:
    """匹配含 note 描述的规则。"""
    pat = rule.get("pattern") or rule.get("description") or ""
    vr = float(row.get("vol_ratio", 0))
    r5 = float(row.get("ret_5d", 0))
    dvol = float(row.get("dvol_m", 0))
    cs = float(row.get("close_strength", 0.5))
    ma = bool(row.get("above_ma50"))

    if "收在日内高位" in pat:
        return cs >= 0.65 and vr >= 1.5
    if "收在日内低位" in pat:
        return cs <= 0.35 and vr >= 1.5 and r5 < 0
    if "量比>2.5" in pat and "5日已跌" in pat:
        return vr >= 2.5 and r5 < 0
    if "量比>2.5" in pat:
        return vr >= 2.5
    if "量比1.5-2.5" in pat:
        return 1.5 <= vr < 2.5
    if "成交额>1B" in pat and "正动量" in pat:
        return dvol >= 1000 and r5 > 0
    if "成交额200M-1B" in pat and "正动量" in pat:
        return 200 <= dvol < 1000 and r5 > 0
    return match_rule(row, rule)


def score_today_row(row: pd.Series) -> dict:
    """单条特征 → 桶标签（供实时匹配）。"""
    r = row.copy()
    if "vol_ratio桶" not in r:
        r["vol_ratio桶"] = _bucket_vol_ratio(float(r.get("vol_ratio", 0)))
    if "ret_5d桶" not in r:
        r["ret_5d桶"] = _bucket_ret5(float(r.get("ret_5d", 0)))
    if "dvol桶" not in r:
        r["dvol桶"] = _bucket_dvol(float(r.get("dvol_m", 0)))
    return r


def assess_down_avoidance(row: pd.Series | dict, params=None) -> list[dict]:
    """下跌/回避规律（243k 事件分桶提炼，无 BS 期权）。

    params: DownParams（来自 pattern_rules_optimized.json），None 用默认。
    """
    if params is None:
        from quant.pattern_params import DownParams
        params = DownParams()
    active = set(params.active_avoid_rules or [])
    from quant.pattern_params import canon_avoid_rule
    r = row if isinstance(row, dict) else row.to_dict()
    vr = float(r.get("vol_ratio", 0) or 0)
    r5 = float(r.get("ret_5d", 0) or 0)
    r20 = float(r.get("ret_20d", 0) or 0)
    dvol = float(r.get("dvol_m", 0) or 0)
    cs = float(r.get("close_strength", 0.5) or 0.5)
    above50 = bool(r.get("above_ma50", r.get("站上MA50", False)))
    hits: list[dict] = []

    from quant.pattern_params import canon_avoid_rule

    def _add(rule_id: str, reason: str, action: str = "回避做多") -> None:
        if active:
            canon = canon_avoid_rule(rule_id)
            if canon not in active:
                return
        hz = params.horizon_for(rule_id)
        urg = "高" if params.is_short_term(rule_id) else "中"
        hits.append({
            "rule_id": rule_id,
            "reason": reason,
            "action": action,
            "horizon": hz,
            "urgency": urg,
        })

    # D3/D4: 5 日涨过多后强跌概率升高
    if r5 > params.shrink_vol_min_ret_5d and vr < params.shrink_vol_max:
        _add("D3_shrink_vol_top", f"5日涨{r5:.0%} + 量比{vr:.2f}<{params.shrink_vol_max}（缩量顶）")
    elif r5 > params.mega_ret_5d and dvol >= params.mega_dvol_m and above50:
        _add("D4_extended_mega", f"5日涨{r5:.0%} + 成交额>{dvol:.0f}M + MA50上（抛物线区）")
    elif r5 > params.extended_off_ma_ret_5d and not above50 and vr < params.extended_off_ma_max_vol:
        _add("D4_extended_off_ma", f"5日涨{r5:.0%} + MA50下 + 量比<{params.extended_off_ma_max_vol}")

    # D1: 爆量冲顶
    if (
        vr >= params.blowoff_min_ratio
        and params.blowoff_ret_5d_min <= r5 <= params.blowoff_ret_5d_max
        and params.blowoff_dvol_min_m <= dvol < params.blowoff_dvol_max_m
        and above50
    ):
        _add("D1_vol_blowoff", f"量比{vr:.1f} + 5日涨{r5:.0%}（放量冲顶）", "谨慎追多")

    # D2: 放量杀跌
    if vr >= params.vol_dump_min_ratio and r5 < params.vol_dump_max_ret_5d:
        _add("D2_vol_dump", f"量比{vr:.1f} + 5日跌{r5:.0%}（放量杀跌）")

    # 出货形态
    if cs <= params.weak_close_max and vr >= params.weak_close_min_vol and r5 < 0:
        _add("D_outflow", f"收在日内低位{cs:.0%} + 放量 + 5日跌")

    # 20 日极端涨幅
    if r20 > params.parabolic_ret_20d:
        _add("D_parabolic_20d", f"20日涨{r20:.0%}（趋势过热）")
    elif r5 > params.parabolic_ret_5d and r20 > params.parabolic_ret_20d * 0.75:
        _add("D_parabolic_5d", f"5日涨{r5:.0%} + 20日涨{r20:.0%}（加速段）")

    return hits


def vectorized_down_mask(
    panel: pd.DataFrame,
    params=None,
    *,
    include_rules: list[str] | None = None,
) -> pd.Series:
    """向量化回避掩码（与 assess_down_avoidance 逻辑一致）。

    include_rules: 仅启用指定规则 id；None=全部。
      D3_shrink, D4_mega, D4_off_ma, D1_blowoff, D2_dump, D_outflow, D_parabolic
    """
    if params is None:
        from quant.pattern_params import DownParams
        params = DownParams()
    use = set(include_rules) if include_rules else None
    def _on(rule: str, mask: pd.Series) -> pd.Series:
        return mask if use is None or rule in use else pd.Series(False, index=panel.index)

    vr = pd.to_numeric(panel["vol_ratio"], errors="coerce")
    r5 = pd.to_numeric(panel["ret_5d"], errors="coerce")
    r20 = pd.to_numeric(panel["ret_20d"], errors="coerce")
    dvol = pd.to_numeric(panel["dvol_m"], errors="coerce")
    cs_col = panel["close_strength"] if "close_strength" in panel.columns else 0.5
    cs = pd.to_numeric(cs_col, errors="coerce")
    if isinstance(cs, (int, float)):
        cs = pd.Series(float(cs), index=panel.index)
    cs = cs.fillna(0.5)
    above50 = panel["above_ma50"].astype(bool) if "above_ma50" in panel.columns else True

    d3 = _on("D3_shrink", (r5 > params.shrink_vol_min_ret_5d) & (vr < params.shrink_vol_max))
    d4_mega = _on("D4_mega", (r5 > params.mega_ret_5d) & (dvol >= params.mega_dvol_m) & above50)
    d4_off = _on(
        "D4_off_ma",
        (r5 > params.extended_off_ma_ret_5d) & (~above50) & (vr < params.extended_off_ma_max_vol),
    )
    d1 = _on(
        "D1_blowoff",
        (vr >= params.blowoff_min_ratio)
        & (r5 >= params.blowoff_ret_5d_min)
        & (r5 <= params.blowoff_ret_5d_max)
        & (dvol >= params.blowoff_dvol_min_m)
        & (dvol < params.blowoff_dvol_max_m)
        & above50,
    )
    d2 = _on("D2_dump", (vr >= params.vol_dump_min_ratio) & (r5 < params.vol_dump_max_ret_5d))
    dout = _on("D_outflow", (cs <= params.weak_close_max) & (vr >= params.weak_close_min_vol) & (r5 < 0))
    dp20 = _on("D_parabolic", r20 > params.parabolic_ret_20d)
    dp5 = _on(
        "D_parabolic",
        (r5 > params.parabolic_ret_5d) & (r20 > params.parabolic_ret_20d * 0.75),
    )
    return d3 | d4_mega | d4_off | d1 | d2 | dout | dp20 | dp5


def assess_up_favor(row: pd.Series | dict) -> list[dict]:
    """上涨规律加分项（后 20 日统计优势，非保证）。"""
    r = row if isinstance(row, dict) else row.to_dict()
    vr = float(r.get("vol_ratio", 0) or 0)
    r5 = float(r.get("ret_5d", 0) or 0)
    dvol = float(r.get("dvol_m", 0) or 0)
    above50 = bool(r.get("above_ma50", r.get("站上MA50", False)))
    tags: list[dict] = []

    if 1.0 <= vr <= 1.5 and 0 <= r5 <= 0.05 and dvol >= 1000 and above50:
        tags.append({"rule_id": "U1", "note": "温和动量+大盘子+MA50（后20d上涨率~59%）"})
    if vr >= 2.5 and 0 <= r5 <= 0.05 and 200 <= dvol < 1000 and above50:
        tags.append({"rule_id": "U2", "note": "爆量初段+成交额2-10亿+MA50（后20d上涨率~63%）"})
    return tags


def live_matches(
    ticker: str,
    feat_row: pd.Series,
    rules: list[dict],
    *,
    as_of: str = "",
) -> list[dict]:
    """今日是否命中某条挖掘规则。"""
    row = score_today_row(feat_row)
    hits: list[dict] = []
    for rule in rules:
        if match_rule_fuzzy(row, rule):
            hits.append({
                "代码": ticker,
                "日期": as_of or str(row.get("日期", "")),
                "方向": "偏多" if rule.get("direction") == "up" else "偏空",
                "规律": rule.get("pattern", rule.get("description", "")),
                "历史胜率": rule.get("win_rate"),
                "样本数": rule.get("sample_n"),
                "建议": rule.get("action", ""),
                "量比": round(float(row.get("vol_ratio", 0)), 2),
                "5日涨幅": round(float(row.get("ret_5d", 0)) * 100, 2),
                "成交额M": round(float(row.get("dvol_m", 0)), 1),
            })
    return hits
