"""资金流向 / 人为操盘痕迹识别 · 每日选股核心。

假设：大涨大跌往往伴随可观测的量价轨迹（堆量、抢筹、冲顶、出货、融资砸盘）。
本模块把轨迹归纳为「上涨操盘」与「下跌操盘」两类规则，并映射到具体策略动作。

量价特征（无未来函数）：
  · 量比、成交额、收盘强度、5/20日涨跌、MA50、上涨日量能占比
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from typing import Any, Literal

from pathlib import Path

import numpy as np
import pandas as pd

from quant.move_pattern import assess_down_avoidance, assess_up_favor, extract_trajectory_features

Direction = Literal["做多", "做空", "回避", "观望"]
Action = Literal["次日做多", "买Put价差", "卖Call价差", "回避追涨", "观望"]


@dataclass
class FlowPattern:
    """一条可匹配的操盘规律。"""

    id: str
    name: str
    side: Literal["up", "down"]
    tier: str  # S/A/B/C
    description: str
    action: Action
    hold: str
    win_rate_hint: str
    min_dvol_m: float = 30.0


# ---------------------------------------------------------------------------
# 规律目录（归纳自全市场异动 + NXTS 类案例）
# ---------------------------------------------------------------------------
FLOW_CATALOG: list[FlowPattern] = [
    # --- 上涨操盘（资金流入 / 拉升） ---
    FlowPattern(
        "U_S1", "温和堆量拉升", "up", "S",
        "涨2.5~5% + 量比1.3~1.75 + 收强≥55% + MA50上 + 5日涨4~25%",
        "次日做多", "1日", "历史次日胜率~65%",
    ),
    FlowPattern(
        "U_S2", "尾盘抢筹", "up", "S",
        "收在日内高位≥65% + 量比≥1.5 + 5日正动量",
        "次日做多", "1日", "资金尾盘抢筹延续",
    ),
    FlowPattern(
        "U_A1", "大盘子温和放量", "up", "A",
        "涨3~5% + 量比1.3~1.5 + 收强≥65% + 成交额>5亿 + MA50",
        "次日做多", "1日", "机构堆量突破",
    ),
    FlowPattern(
        "U_A2", "爆量初段", "up", "A",
        "量比>2.5 + 5日涨0~5% + 成交额2~10亿 + MA50（初段非冲顶）",
        "次日做多", "1~3日", "放量启动段",
    ),
  # --- 下跌操盘（资金流出 / 出货） ---
    FlowPattern(
        "D_S1", "抛物线出货区", "down", "S",
        "5日涨>15% 或 20日涨>40%（趋势过热，易回踩）",
        "回避追涨", "3~5日", "NXTS类暴涨后易融资砸盘",
    ),
    FlowPattern(
        "D_S2", "暴涨次日回吐", "down", "S",
        "前日涨幅>40%（动量耗尽，均值回归）",
        "买Put价差", "1日", "前日极端追涨次日回落概率高",
    ),
    FlowPattern(
        "D_OFFERING", "融资砸盘", "down", "S",
        "SEC 8-K 披露定向增发/ATM + 近期暴涨（先拉后融）",
        "买Put价差", "1~3日", "NXTS类：利好拉升后稀释",
    ),
    FlowPattern(
        "D_A1", "放量冲顶", "down", "A",
        "量比≥2.5 + 5日涨5~15% + 成交额50M~1B + MA50（冲顶形态）",
        "卖Call价差", "3~5日", "冲高后横盘或回落",
    ),
    FlowPattern(
        "D_A2", "放量杀跌", "down", "A",
        "量比≥2.5 + 5日跌（资金出逃）",
        "回避追涨", "5~20日", "放量下跌延续偏弱",
    ),
    FlowPattern(
        "D_A3", "弱收盘出货", "down", "A",
        "收在日内低位≤35% + 放量 + 5日跌",
        "买Put价差", "1~3日", "尾盘砸盘形态",
    ),
    FlowPattern(
        "D_B1", "缩量顶", "down", "B",
        "5日涨>8% + 量比<1.2（涨不动、量萎缩）",
        "回避追涨", "3~5日", "量价背离",
    ),
    FlowPattern(
        "D_B2", "小盘极端波动", "down", "B",
        "市值信号：涨>12% 或 振幅>25%（流动性真空，人为波动大）",
        "观望", "—", "不参与极端博弈",
    ),
    FlowPattern(
        "D_B3", "超涨弱市回吐", "down", "B",
        "涨7~14% + 收弱≤50% + 量比1.5~6 + 弱市SPY<MA50",
        "买Put价差", "1日", "弱市超涨次日回吐",
    ),
]


def enrich_flow_row(df: pd.DataFrame, spy_close: pd.Series | None = None) -> dict[str, Any]:
    """从 OHLCV 提取最新一条资金流向特征。"""
    if df is None or df.empty or len(df) < 25:
        return {}
    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    vol = df["Volume"].astype(float)
    s = float(close.iloc[-1])
    prev = float(close.iloc[-2])
    prev2 = float(close.iloc[-3]) if len(close) >= 3 else prev
    hi, lo = float(high.iloc[-1]), float(low.iloc[-1])
    gain_1d = s / prev - 1
    gain_prev_1d = prev / prev2 - 1 if prev2 > 0 else 0
    amp = (hi - lo) / prev if prev > 0 else 0
    vma20 = float(vol.iloc[-21:-1].mean()) if len(vol) >= 21 else float(vol.mean())
    vol_ratio = float(vol.iloc[-1] / vma20) if vma20 > 0 else 1.0
    dvol_m = float((close * vol).iloc[-20:].mean() / 1e6)
    ma50 = float(close.rolling(50, min_periods=25).mean().iloc[-1])
    ma20 = float(close.rolling(20, min_periods=10).mean().iloc[-1])
    ret_5d = float(close.iloc[-1] / close.iloc[-6] - 1) if len(close) >= 6 else 0
    ret_20d = float(close.iloc[-1] / close.iloc[-21] - 1) if len(close) >= 21 else 0
    cs = (s - lo) / (hi - lo) if hi > lo else 0.5
    # 上涨日量能占比
    ret_1d_series = close.pct_change()
    up_mask = ret_1d_series > 0
    up_vol_10 = vol.where(up_mask).rolling(10, min_periods=5).sum().iloc[-1]
    tot_vol_10 = vol.rolling(10, min_periods=5).sum().iloc[-1]
    up_vol_share = float(up_vol_10 / tot_vol_10) if tot_vol_10 and tot_vol_10 > 0 else 0.5
    rs_20d = np.nan
    if spy_close is not None and len(spy_close) >= 21:
        spy_ret = float(spy_close.iloc[-1] / spy_close.iloc[-21] - 1)
        rs_20d = ret_20d - spy_ret
    return {
        "现价": s,
        "涨幅%": gain_1d * 100,
        "前日涨幅%": gain_prev_1d * 100,
        "振幅%": amp * 100,
        "量比": vol_ratio,
        "成交额M": dvol_m,
        "收盘强度": cs,
        "涨幅5d%": ret_5d * 100,
        "涨幅20d%": ret_20d * 100,
        "相对SPY20d%": rs_20d * 100 if np.isfinite(rs_20d) else np.nan,
        "上涨量能占比": up_vol_share,
        "above_ma50": s > ma50,
        "above_ma20": s > ma20,
        "ret_1d": gain_1d,
        "ret_5d": ret_5d,
        "ret_20d": ret_20d,
        "vol_ratio": vol_ratio,
        "dvol_m": dvol_m,
        "close_strength": cs,
    }


def _match_up_patterns(r: dict[str, Any], *, spy_bull: bool) -> list[dict]:
    hits: list[dict] = []
    vr = float(r.get("vol_ratio", 0))
    g1 = float(r.get("涨幅%", 0))
    g5 = float(r.get("涨幅5d%", 0))
    cs = float(r.get("close_strength", 0.5))
    dvol = float(r.get("dvol_m", 0))
    ma50 = bool(r.get("above_ma50", False))

    if not spy_bull:
        return hits  # 弱市不做多规律

    if (
        1.3 <= vr <= 1.75
        and 2.5 <= g1 <= 5.5
        and cs >= 0.55
        and ma50
        and 4 <= g5 <= 25
    ):
        hits.append(_hit("U_S1", r))
    if cs >= 0.65 and vr >= 1.5 and g5 > 0:
        hits.append(_hit("U_S2", r))
    if (
        1.3 <= vr <= 1.5
        and 3 <= g1 <= 5
        and cs >= 0.65
        and dvol >= 500
        and ma50
    ):
        hits.append(_hit("U_A1", r))
    if vr >= 2.5 and 0 <= g5 <= 5 and 200 <= dvol < 1000 and ma50:
        hits.append(_hit("U_A2", r))
    return hits


def _match_down_patterns(
    r: dict[str, Any],
    *,
    spy_bull: bool,
    spy_1d_pct: float | None = None,
) -> list[dict]:
    hits: list[dict] = []
    vr = float(r.get("vol_ratio", 0))
    g1 = float(r.get("涨幅%", 0))
    g5 = float(r.get("涨幅5d%", 0))
    g20 = float(r.get("涨幅20d%", 0))
    g_prev = float(r.get("前日涨幅%", 0))
    amp = float(r.get("振幅%", abs(g1)))
    cs = float(r.get("close_strength", 0.5))
    dvol = float(r.get("dvol_m", 0))
    ma50 = bool(r.get("above_ma50", False))

    if g5 > 15 or g20 > 40:
        hits.append(_hit("D_S1", r))
    if g_prev > 40:
        hits.append(_hit("D_S2", r))
    if (
        vr >= 2.5
        and 5 <= g5 <= 15
        and 50 <= dvol < 1000
        and ma50
    ):
        hits.append(_hit("D_A1", r))
    if vr >= 2.5 and g5 < 0:
        hits.append(_hit("D_A2", r))
    if cs <= 0.35 and vr >= 1.5 and g5 < 0:
        hits.append(_hit("D_A3", r))
    if g5 > 8 and vr < 1.2:
        hits.append(_hit("D_B1", r))
    if g1 >= 12 or amp >= 25:
        hits.append(_hit("D_B2", r))
    spy_bear = not spy_bull
    if (
        spy_bear
        and 7 <= g1 <= 14
        and cs <= 0.50
        and 1.5 <= vr <= 6.0
    ):
        hits.append(_hit("D_B3", r))
    return hits


def _hit(pattern_id: str, r: dict[str, Any]) -> dict[str, Any]:
    pat = next((p for p in FLOW_CATALOG if p.id == pattern_id), None)
    if pat is None:
        return {"规律ID": pattern_id}
    direction: Direction = "做多" if pat.side == "up" else (
        "做空" if "Put" in pat.action or "做空" in pat.action else "回避"
    )
    return {
        "规律ID": pat.id,
        "规律名": pat.name,
        "方向": direction,
        "等级": pat.tier,
        "操盘痕迹": pat.description,
        "策略动作": pat.action,
        "持有": pat.hold,
        "胜率参考": pat.win_rate_hint,
        "涨幅%": round(float(r.get("涨幅%", 0)), 2),
        "前日涨幅%": round(float(r.get("前日涨幅%", 0)), 2),
        "量比": round(float(r.get("量比", 0)), 2),
        "收盘强度": round(float(r.get("close_strength", 0.5)), 2),
        "成交额M": round(float(r.get("dvol_m", 0)), 1),
        "5日涨%": round(float(r.get("涨幅5d%", 0)), 1),
        "20日涨%": round(float(r.get("涨幅20d%", 0)), 1),
        "MA50": "上" if r.get("above_ma50") else "下",
    }


def assess_flow_patterns(
    row: dict[str, Any] | pd.Series,
    *,
    spy_bull: bool = True,
    spy_1d_pct: float | None = None,
) -> dict[str, Any]:
    """单票：匹配上涨/下跌操盘规律 + 轨迹库加分/回避。"""
    r = dict(row) if isinstance(row, pd.Series) else row
    up_hits = _match_up_patterns(r, spy_bull=spy_bull)
    down_hits = _match_down_patterns(r, spy_bull=spy_bull, spy_1d_pct=spy_1d_pct)

    # 叠加 move_pattern 轨迹库
    traj_row = {
        "vol_ratio": r.get("vol_ratio", r.get("量比")),
        "ret_5d": r.get("ret_5d", float(r.get("涨幅5d%", 0)) / 100),
        "ret_20d": r.get("ret_20d", float(r.get("涨幅20d%", 0)) / 100),
        "dvol_m": r.get("dvol_m", r.get("成交额M")),
        "close_strength": r.get("close_strength", r.get("收盘强度")),
        "above_ma50": r.get("above_ma50", r.get("MA50") == "上"),
    }
    traj_up = assess_up_favor(traj_row)
    traj_down = assess_down_avoidance(traj_row)

    # 综合信号
    best_up = max(up_hits, key=lambda h: _tier_score(h.get("等级", "C"))) if up_hits else None
    best_down = max(down_hits, key=lambda h: _tier_score(h.get("等级", "C"))) if down_hits else None

    signal, action, reason = _resolve_signal(
        best_up, best_down, down_hits, traj_down, traj_up, spy_bull,
    )

    return {
        "信号": signal,
        "策略动作": action,
        "选股理由": reason,
        "上涨规律": up_hits,
        "下跌规律": down_hits,
        "轨迹加分": traj_up,
        "轨迹回避": traj_down,
        "特征": r,
    }


def _tier_score(tier: str) -> int:
    return {"S": 4, "A": 3, "B": 2, "C": 1}.get(str(tier), 0)


def _resolve_signal(
    best_up: dict | None,
    best_down: dict | None,
    down_hits: list[dict],
    traj_down: list[dict],
    traj_up: list[dict],
    spy_bull: bool,
) -> tuple[Direction, Action, str]:
    """冲突时：下跌 S/A 级优先于上涨（防 NXTS 类陷阱）。"""
    down_tier = _tier_score(best_down.get("等级", "C")) if best_down else 0
    up_tier = _tier_score(best_up.get("等级", "C")) if best_up else 0

    # 极端波动一律观望
    if any(h.get("规律ID") == "D_B2" for h in down_hits):
        return "观望", "观望", "极端波动不参与（流动性真空/人为博弈）"

    # 下跌信号优先
    if best_down and down_tier >= 3:
        act = best_down.get("策略动作", "回避追涨")
        action: Action = act if act in {"次日做多", "买Put价差", "卖Call价差", "回避追涨", "观望"} else "回避追涨"
        dir_: Direction = "做空" if "Put" in action else "回避"
        traj_note = "；".join(t["reason"] for t in traj_down[:2]) if traj_down else ""
        reason = f"{best_down['规律名']}：{best_down['操盘痕迹']}"
        if traj_note:
            reason += f" | 轨迹:{traj_note}"
        return dir_, action, reason

    if traj_down and not spy_bull:
        d0 = traj_down[0]
        return "回避", "回避追涨", f"轨迹回避·{d0.get('reason', '')}"

    if best_up and up_tier >= 3 and spy_bull:
        traj_note = "；".join(t["note"] for t in traj_up[:2]) if traj_up else ""
        reason = f"{best_up['规律名']}：{best_up['操盘痕迹']}"
        if traj_note:
            reason += f" | 轨迹:{traj_note}"
        return "做多", "次日做多", reason

    if best_up and spy_bull:
        traj_note = "；".join(t["note"] for t in traj_up[:2]) if traj_up else ""
        reason = f"{best_up['规律名']}：{best_up['操盘痕迹']}"
        if traj_note:
            reason += f" | 轨迹:{traj_note}"
        return "做多", "次日做多", reason

    if best_down:
        act = best_down.get("策略动作", "回避追涨")
        action = act if act in {"次日做多", "买Put价差", "卖Call价差", "回避追涨", "观望"} else "回避追涨"
        dir_ = "做空" if "Put" in action else "回避"
        return dir_, action, f"{best_down['规律名']}：{best_down['操盘痕迹']}"

    return "观望", "观望", "未命中操盘规律"


def scan_universe_flow(
    batch: dict[str, pd.DataFrame],
    *,
    spy_close: pd.Series | None = None,
    spy_bull: bool = True,
    spy_1d_pct: float | None = None,
    min_dvol_m: float = 30.0,
    min_price: float = 3.0,
) -> pd.DataFrame:
    """扫描一批 ticker，返回按信号强度排序的选股表。"""
    rows: list[dict] = []
    for tk, df in batch.items():
        if df is None or df.empty:
            continue
        feat = enrich_flow_row(df, spy_close)
        if not feat or feat.get("dvol_m", 0) < min_dvol_m or feat.get("现价", 0) < min_price:
            continue
        feat["代码"] = tk.upper()
        res = assess_flow_patterns(feat, spy_bull=spy_bull, spy_1d_pct=spy_1d_pct)
        if res["信号"] == "观望" and not res["下跌规律"] and not res["上涨规律"]:
            continue
        up_ids = [h["规律ID"] for h in res["上涨规律"]]
        down_ids = [h["规律ID"] for h in res["下跌规律"]]
        rows.append({
            "代码": tk.upper(),
            "现价": round(float(feat["现价"]), 2),
            "信号": res["信号"],
            "策略动作": res["策略动作"],
            "选股理由": res["选股理由"],
            "上涨规律": "、".join(up_ids) if up_ids else "—",
            "下跌规律": "、".join(down_ids) if down_ids else "—",
            "涨幅%": feat.get("涨幅%"),
            "量比": round(float(feat.get("量比", 0)), 2),
            "收盘强度": round(float(feat.get("close_strength", 0)), 2),
            "成交额M": round(float(feat.get("dvol_m", 0)), 1),
            "5日涨%": round(float(feat.get("涨幅5d%", 0)), 1),
            "20日涨%": round(float(feat.get("涨幅20d%", 0)), 1),
            "MA50": "上" if feat.get("above_ma50") else "下",
            "_score": _signal_score(res),
        })
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).sort_values("_score", ascending=False)
    return out.drop(columns=["_score"])


def _signal_score(res: dict) -> float:
    sig = res.get("信号", "观望")
    base = {"做多": 3, "做空": 2.5, "回避": 2, "观望": 0}.get(sig, 0)
    up = res.get("上涨规律") or []
    down = res.get("下跌规律") or []
    tier_bonus = sum(_tier_score(h.get("等级", "C")) for h in up + down)
    return base + tier_bonus


def format_catalog() -> list[str]:
    """人类可读的规律目录。"""
    lines: list[str] = []
    lines.append("【上涨操盘 · 资金流入】")
    for p in FLOW_CATALOG:
        if p.side == "up":
            lines.append(f"  {p.id} [{p.tier}] {p.name} → {p.action}（{p.hold}）")
            lines.append(f"      {p.description}")
    lines.append("")
    lines.append("【下跌操盘 · 资金流出/陷阱】")
    for p in FLOW_CATALOG:
        if p.side == "down":
            lines.append(f"  {p.id} [{p.tier}] {p.name} → {p.action}（{p.hold}）")
            lines.append(f"      {p.description}")
    return lines


STATS_JSON = Path(__file__).resolve().parents[1] / "research" / "flow_pattern_stats.json"


def build_flow_history(
    df: pd.DataFrame,
    spy_close: pd.Series | None = None,
    *,
    min_idx: int = 55,
) -> pd.DataFrame:
    """逐日构建资金流向特征（回测用，无未来函数）。"""
    if df is None or df.empty or len(df) < min_idx + 2:
        return pd.DataFrame()
    spy_ma50 = None
    if spy_close is not None and len(spy_close) >= 50:
        spy_ma50 = spy_close.rolling(50, min_periods=25).mean()
    rows: list[dict] = []
    for i in range(min_idx, len(df) - 1):
        sub = df.iloc[: i + 1]
        feat = enrich_flow_row(sub, spy_close.iloc[: i + 1] if spy_close is not None else None)
        if not feat:
            continue
        feat["日期"] = df.index[i]
        if spy_close is not None and spy_ma50 is not None:
            try:
                feat["spy_bull"] = float(spy_close.iloc[i]) > float(spy_ma50.iloc[i])
            except Exception:  # noqa: BLE001
                feat["spy_bull"] = True
        else:
            feat["spy_bull"] = True
        # 次日收益（标签）
        c0 = float(df["Close"].astype(float).iloc[i])
        c1 = float(df["Close"].astype(float).iloc[i + 1])
        feat["fwd_1d"] = c1 / c0 - 1 if c0 > 0 else 0.0
        rows.append(feat)
    return pd.DataFrame(rows)


def match_pattern_by_id(pattern_id: str, r: dict[str, Any], *, spy_bull: bool = True) -> bool:
    """单条规律是否命中（回测/统计用）。"""
    up = _match_up_patterns(r, spy_bull=spy_bull)
    down = _match_down_patterns(r, spy_bull=spy_bull)
    ids = [h["规律ID"] for h in up + down]
    return pattern_id in ids


def load_flow_stats(path: Path | None = None) -> dict[str, Any]:
    p = path or STATS_JSON
    if not p.exists():
        return {}
    import json
    return json.loads(p.read_text(encoding="utf-8-sig"))


def stats_hint_for_pattern(pattern_id: str, stats: dict[str, Any] | None = None) -> str:
    doc = stats or load_flow_stats()
    row = (doc.get("patterns") or {}).get(pattern_id)
    if not row:
        return ""
    wr = row.get("win_rate_1d")
    n = row.get("sample_n")
    mean = row.get("mean_ret_1d_pct")
    if wr is None:
        return ""
    return f"回测1日胜率{wr:.0%} n={n} 均收益{mean:+.2f}%"


def build_daily_picks(
    scan_df: pd.DataFrame,
    *,
    long_top_n: int = 3,
    short_top_n: int = 3,
    avoid_top_n: int = 5,
) -> dict[str, pd.DataFrame]:
    """把扫描结果分为做多/做空/回避三池。"""
    if scan_df.empty:
        return {"long": pd.DataFrame(), "short": pd.DataFrame(), "avoid": pd.DataFrame()}
    long_df = scan_df[scan_df["信号"] == "做多"].head(long_top_n)
    short_df = scan_df[scan_df["信号"] == "做空"].head(short_top_n)
    avoid_df = scan_df[scan_df["信号"] == "回避"].head(avoid_top_n)
    return {"long": long_df, "short": short_df, "avoid": avoid_df}
