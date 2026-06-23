"""单标的规律策略 · 信号检测 + 治理过滤 + 每日扫描。

策略族（回测见 research/ticker_pattern_backtest.py）：
  S1 动量顺势 · S7 趋势中继 · S8 纯多头（S1|S7）
  S8U Ultra80 · 精英入场 + 路径止盈+2%（胜率≥80%）
  S2~S4 空头/回避信号（每日扫描中用于拦截做多）
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant.move_pattern import extract_trajectory_features_5d
from research.liquid_tier_a_scan import _avg_dollar_vol
from research.medallion_short import fetch_shares

ROOT = Path(__file__).resolve().parents[1]
SPECULATIVE_POOL_JSON = ROOT / "research" / "speculative_pool.json"
BACKTEST_JSON = ROOT / "research" / "ticker_pattern_backtest.json"
APPROVED_JSON = ROOT / "research" / "s8u_approved_tickers.json"

DEFAULT_MEME_LONG_TICKERS = ["MSTR", "SMCI", "COIN"]
HOLD_DAYS_DEFAULT = 5
PARAMS_JSON = ROOT / "research" / "ticker_pattern_params.json"
OOS_APPROVED_DEFAULT_MIN_TRADES = 5
OOS_APPROVED_DEFAULT_MIN_WIN = 0.80


@dataclass
class GovernanceRule:
    enabled: bool = True
    pause_until: str | None = None  # YYYY-MM-DD
    block_if_5d_drop_pct: float = 15.0
    block_if_shrink_top: bool = True
    block_if_overheat: bool = True


FEE_BPS_RT = 10  # 往返 10bp


@dataclass
class MemeLongConfig:
    enabled: bool = True
    bull_only: bool = True
    high_win_mode: bool = True
    tickers: list[str] = field(default_factory=lambda: list(DEFAULT_MEME_LONG_TICKERS))
    hold_days: int = HOLD_DAYS_DEFAULT
    alloc_pct: float = 0.25
    path_take_profit_pct: float = 0.02
    path_stop_loss_pct: float = 0.05
    exit_mode: str = "path_tp"  # path_tp | close_hold
    take_profit_pct: float = 0.08
    stop_loss_pct: float = 0.05
    use_bracket_exit: bool = False
    min_win_target: float = 0.80
    ticker_source: str = "manual"  # manual | oos_approved
    approved_file: str | None = None
    extra_tickers: list[str] = field(default_factory=list)
    position_weights: dict[str, float] = field(default_factory=dict)
    governance: dict[str, GovernanceRule] = field(default_factory=dict)


def load_oos_approved_doc(path: Path | None = None) -> dict:
    p = path or APPROVED_JSON
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def load_oos_approved_tickers(path: Path | None = None) -> list[str]:
    doc = load_oos_approved_doc(path)
    out: list[str] = []
    seen: set[str] = set()
    for tk in doc.get("tickers") or []:
        u = str(tk).upper()
        if u and u not in seen:
            seen.add(u)
            out.append(u)
    return out


def resolve_meme_long_tickers(raw: dict) -> list[str]:
    """解析 meme_long 扫描池：manual / oos_approved / speculative_pool + extra_tickers。"""
    from quant.speculative_pool import load_pool_tickers

    src = str(raw.get("ticker_source", "manual")).lower()
    extra = [str(t).upper() for t in (raw.get("extra_tickers") or [])]
    if src == "oos_approved":
        ap_path = raw.get("approved_file")
        path = ROOT / ap_path if ap_path else APPROVED_JSON
        tickers = load_oos_approved_tickers(path)
        if not tickers:
            tickers = list(DEFAULT_MEME_LONG_TICKERS)
    elif src == "speculative_pool":
        pool_path = raw.get("speculative_pool_file")
        p = ROOT / pool_path if pool_path else SPECULATIVE_POOL_JSON
        tier = str(raw.get("speculative_pool_tier", "core"))
        tickers = load_pool_tickers(p, tier=tier)
        if not tickers:
            tickers = list(DEFAULT_MEME_LONG_TICKERS)
    else:
        tickers = [str(t).upper() for t in (raw.get("tickers") or DEFAULT_MEME_LONG_TICKERS)]
    if raw.get("include_speculative_pool") and src != "speculative_pool":
        pool_path = raw.get("speculative_pool_file")
        p = ROOT / pool_path if pool_path else SPECULATIVE_POOL_JSON
        tier = str(raw.get("speculative_pool_tier", "core"))
        extra = list(extra) + load_pool_tickers(p, tier=tier)
    seen = set(tickers)
    for tk in extra:
        if tk not in seen:
            tickers.append(tk)
            seen.add(tk)
    return tickers


def parse_meme_long(cfg: dict | None) -> MemeLongConfig:
    raw = (cfg or {}).get("meme_long") or {}
    gov_raw = raw.get("governance") or {}
    gov: dict[str, GovernanceRule] = {}
    for tk, g in gov_raw.items():
        if not isinstance(g, dict):
            continue
        gov[str(tk).upper()] = GovernanceRule(
            enabled=bool(g.get("enabled", True)),
            pause_until=g.get("pause_until"),
            block_if_5d_drop_pct=float(g.get("block_if_5d_drop_pct", 15.0)),
            block_if_shrink_top=bool(g.get("block_if_shrink_top", True)),
            block_if_overheat=bool(g.get("block_if_overheat", True)),
        )
    tickers = resolve_meme_long_tickers(raw)
    return MemeLongConfig(
        enabled=bool(raw.get("enabled", True)),
        bull_only=bool(raw.get("bull_only", True)),
        high_win_mode=bool(raw.get("high_win_mode", True)),
        tickers=tickers,
        hold_days=int(raw.get("hold_days", HOLD_DAYS_DEFAULT)),
        alloc_pct=float(raw.get("alloc_pct", 0.25)),
        path_take_profit_pct=float(raw.get("path_take_profit_pct", raw.get("path_tp_pct", 0.02))),
        path_stop_loss_pct=float(raw.get("path_stop_loss_pct", raw.get("path_sl_pct", 0.05))),
        exit_mode=str(raw.get("exit_mode", "path_tp")),
        take_profit_pct=float(raw.get("take_profit_pct", 0.08)),
        stop_loss_pct=float(raw.get("stop_loss_pct", 0.05)),
        use_bracket_exit=bool(raw.get("use_bracket_exit", False)),
        min_win_target=float(raw.get("min_win_target", 0.80)),
        ticker_source=str(raw.get("ticker_source", "manual")),
        approved_file=raw.get("approved_file"),
        extra_tickers=[str(t).upper() for t in (raw.get("extra_tickers") or [])],
        position_weights={str(k).upper(): float(v) for k, v in (raw.get("position_weights") or {}).items()},
        governance=gov,
    )


def _as_dict(row: pd.Series | dict) -> dict:
    return dict(row) if isinstance(row, pd.Series) else row


def _base_long_ok(row: pd.Series | dict, spy_bull: bool) -> bool:
    """高胜率公共门槛：牛市 + MA50 + 无回避。"""
    if not spy_bull:
        return False
    r = _as_dict(row)
    if not r.get("above_ma50"):
        return False
    if detect_avoid_tags(r):
        return False
    return True


ULTRA_ELITE_TICKERS = frozenset({"MSTR", "SMCI", "COIN"})

# 非 meme 高流通票：通用 Ultra 门槛（与三票精英规则同量级）
ULTRA_GENERIC = {
    "vol_ratio_min": 2.0,
    "ret_5d_min": 0.05,
    "close_strength_min": 0.55,
    "turnover_pct_min": 1.0,  # 大盘换手低于 meme，阈值放宽
}


def long_signal_ultra(row: pd.Series | dict, spy_bull: bool) -> tuple[str | None, str]:
    """Ultra80 · 精英入场（回测 ALL 85.7% / OOS 90.9% 路径止盈胜率）。

    配合 exit_mode=path_tp：5 日内触达 +2% 即赢，-5% 止损。
    MSTR/SMCI/COIN 用分标的规则；其它高流通票走 ULTRA_GENERIC。
    """
    if not _base_long_ok(row, spy_bull):
        return None, ""
    r = _as_dict(row)
    tk = str(r.get("代码", "")).upper()
    r5 = float(r["ret_5d"])
    vr = float(r["vol_ratio"])
    cs = float(r.get("close_strength", 0.5))
    turn = float(r.get("换手率%") or 0)

    if tk == "MSTR":
        if vr >= 2.0 and r5 >= 0.05 and turn >= 3.0:
            return "U1", "精英动量·高换手"
    elif tk == "SMCI":
        if vr >= 1.5 and r5 >= 0.05 and cs >= 0.55 and turn >= 3.0:
            return "U1", "精英动量·高换手"
    elif tk == "COIN":
        if vr >= 2.5 and r5 >= 0.05 and cs >= 0.60:
            return "U1", "精英动量·强收"
    else:
        g = ULTRA_GENERIC
        turn_ok = (not np.isfinite(turn)) or turn >= g["turnover_pct_min"]
        if (
            vr >= g["vol_ratio_min"]
            and r5 >= g["ret_5d_min"]
            and cs >= g["close_strength_min"]
            and turn_ok
        ):
            return "U1", "通用精英动量"
    return None, ""


def long_signal_highwin(row: pd.Series | dict, spy_bull: bool) -> tuple[str | None, str]:
    """S8H · 分标的 S1/S7 + 无回避（收盘持有，胜率 ~55%）。"""
    if not _base_long_ok(row, spy_bull):
        return None, ""
    r = _as_dict(row)
    tk = str(r.get("代码", "")).upper()
    r5 = float(r["ret_5d"])
    r20 = float(r.get("ret_20d", 0) or 0)
    vr = float(r["vol_ratio"])
    cs = float(r.get("close_strength", 0.5))
    turn = float(r.get("换手率%") or 0)
    g1 = float(r.get("ret_1d", 0) or 0)
    if g1 > 0.12:
        return None, ""

    if tk == "MSTR":
        if r20 >= 0.30:
            return "S7", "趋势中继"
        if (
            vr >= 2.0 and r5 >= 0.08 and cs >= 0.55
            and (not np.isfinite(turn) or turn >= 2.0)
        ):
            return "S1", "动量·放量"
    elif tk == "SMCI":
        if r20 >= 0.30:
            return "S7", "趋势中继"
        if vr >= 1.5 and r5 >= 0.08 and cs >= 0.55:
            return "S1", "动量·放量"
    else:
        if r20 >= 0.30:
            return "S7", "趋势中继"
        if vr >= 1.5 and r5 >= 0.08 and cs >= 0.55:
            return "S1", "动量·放量"
    return None, ""


# ---------------------------------------------------------------------------
# 信号（与 ticker_pattern_backtest 一致）
# ---------------------------------------------------------------------------
def long_momentum(row: pd.Series | dict, spy_bull: bool) -> bool:
    if not spy_bull:
        return False
    r = dict(row) if isinstance(row, pd.Series) else row
    vr = float(r["vol_ratio"])
    r5 = float(r["ret_5d"])
    turn = float(r.get("换手率%") or 0)
    cs = float(r.get("close_strength", 0.5))
    tk = str(r.get("代码", ""))
    vr_min = 2.0 if tk == "MSTR" else 1.5
    turn_ok = (not np.isfinite(turn)) or turn >= 2.0
    return bool(r["above_ma50"]) and vr >= vr_min and r5 >= 0.05 and turn_ok and cs >= 0.50


def long_trend(row: pd.Series | dict, spy_bull: bool) -> bool:
    if not spy_bull:
        return False
    r = dict(row) if isinstance(row, pd.Series) else row
    return bool(r["above_ma50"]) and float(r.get("ret_20d", 0) or 0) >= 0.30


def short_deep_drop(row: pd.Series | dict) -> bool:
    r = dict(row) if isinstance(row, pd.Series) else row
    return float(r["ret_5d"]) <= -0.10


def short_shrink_top(row: pd.Series | dict) -> bool:
    r = dict(row) if isinstance(row, pd.Series) else row
    return float(r["ret_5d"]) >= 0.15 and float(r["vol_ratio"]) < 1.0


def short_overheat(row: pd.Series | dict) -> bool:
    r = dict(row) if isinstance(row, pd.Series) else row
    return float(r.get("ret_20d", 0) or 0) >= 0.40


def short_fade(row: pd.Series | dict, spy_bull: bool) -> bool:
    """弱市超涨回吐。"""
    if spy_bull:
        return False
    r = dict(row) if isinstance(row, pd.Series) else row
    g1 = float(r.get("ret_1d", 0))
    vr = float(r["vol_ratio"])
    cs = float(r.get("close_strength", 0.5))
    g20 = float(r.get("ret_20d", 0) or 0)
    return 0.07 <= g1 <= 0.14 and 1.5 <= vr <= 6.0 and cs <= 0.50 and g20 >= 0.08


def detect_long_signal(
    row: pd.Series | dict,
    spy_bull: bool,
    *,
    high_win: bool = True,
) -> tuple[str | None, str]:
    """返回 (策略ID, 描述) 或 (None, '')。"""
    if high_win:
        return long_signal_ultra(row, spy_bull)
    if long_momentum(row, spy_bull):
        return "S1", "动量顺势"
    if long_trend(row, spy_bull):
        return "S7", "趋势中继"
    return None, ""


def detect_avoid_tags(row: pd.Series | dict) -> list[str]:
    tags: list[str] = []
    if short_shrink_top(row):
        tags.append("S3缩量顶")
    if short_deep_drop(row):
        tags.append("S2深跌惯性")
    if short_overheat(row):
        tags.append("S4过热")
    return tags


def governance_blocked(
    ticker: str,
    row: pd.Series | dict,
    rule: GovernanceRule | None,
    *,
    as_of: date | None = None,
) -> tuple[bool, str]:
    """治理/事件过滤（默认 SMCI）。"""
    if rule is None or not rule.enabled:
        return False, ""
    tk = ticker.upper()
    as_of = as_of or date.today()
    if rule.pause_until:
        try:
            until = date.fromisoformat(str(rule.pause_until))
            if as_of <= until:
                return True, f"{tk} 治理暂停至 {rule.pause_until}"
        except ValueError:
            pass
    r = dict(row) if isinstance(row, pd.Series) else row
    r5_pct = float(r.get("ret_5d", 0) or 0) * 100
    if r5_pct <= -abs(rule.block_if_5d_drop_pct):
        return True, f"{tk} 5日跌{r5_pct:.1f}%≥{rule.block_if_5d_drop_pct}%（事件冲击）"
    if rule.block_if_shrink_top and short_shrink_top(row):
        return True, f"{tk} 缩量顶形态（治理期不做多）"
    if rule.block_if_overheat and short_overheat(row):
        return True, f"{tk} 20日过热≥40%（不追）"
    return False, ""


def build_latest_row(ticker: str, df: pd.DataFrame, *, shares_out: float | None = None) -> dict[str, Any]:
    """从 OHLCV 构建最新特征行。"""
    tk = ticker.upper()
    sh = shares_out if shares_out else fetch_shares([tk]).get(tk)
    feat = extract_trajectory_features_5d(
        df, shares_out=sh, horizon=5, up_threshold=0.02, down_threshold=0.02,
    )
    if feat.empty:
        return {}
    last = feat.iloc[-1].to_dict()
    last["代码"] = tk
    last["avg_dvol_m"] = _avg_dollar_vol(df["Close"], df["Volume"]) / 1e6
    if "日期" in last:
        last["日期"] = pd.Timestamp(last["日期"]).strftime("%Y-%m-%d")
    return last


def load_backtest_hint(ticker: str, strategy_id: str = "S8U") -> str:
    """从准入清单 / 回测 JSON 读取 OOS 参考指标。"""
    tk = ticker.upper()
    for row in load_oos_approved_doc().get("details") or []:
        if row.get("代码") == tk:
            return (
                f"OOS胜率{row.get('oos_win', row.get('胜率', 0)):.0%} "
                f"n={int(row.get('oos_n', row.get('笔数', 0)))} "
                f"年化{row.get('年化', 0):+.0%} "
                f"回撤{row.get('最大回撤', 0):.0%}"
            )
    if not BACKTEST_JSON.exists():
        return ""
    try:
        doc = json.loads(BACKTEST_JSON.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return ""
    prefer = ("S8U", "S8H", "S8") if strategy_id.startswith("S8") else (strategy_id,)
    for sid in prefer:
        for row in doc.get("summary_table") or []:
            if row.get("代码") == tk and row.get("策略") == sid:
                if row.get("区间") == "样本外2024+":
                    return (
                        f"OOS胜率{row.get('胜率', 0):.0%} "
                        f"年化{row.get('年化', 0):+.0%} "
                        f"回撤{row.get('最大回撤', 0):.0%}"
                    )
    return ""


def path_trade_return(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    entry_i: int,
    hold: int,
    *,
    take_profit: float = 0.02,
    stop_loss: float = 0.05,
    fee: float = FEE_BPS_RT / 10_000,
) -> tuple[float | None, int, str]:
    """路径止盈/止损；返回 (收益, 持有天数, 出场原因)。"""
    ret, held = trade_return_bracket(
        close, high, low, entry_i, hold,
        take_profit=take_profit, stop_loss=stop_loss, fee=fee,
    )
    if ret is None:
        return None, hold, ""
    entry_px = float(close.iloc[entry_i])
    tp_px = entry_px * (1 + take_profit)
    sl_px = entry_px * (1 - stop_loss)
    last_i = min(entry_i + hold, len(close) - 1)
    reason = "到期"
    for j in range(entry_i + 1, min(entry_i + held, last_i) + 1):
        if float(low.iloc[j]) <= sl_px:
            reason = "止损"
            break
        if float(high.iloc[j]) >= tp_px:
            reason = "止盈"
            break
    return ret, held, reason


def trade_return_bracket(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    entry_i: int,
    hold: int,
    *,
    take_profit: float = 0.08,
    stop_loss: float = 0.05,
    fee: float = 10 / 10_000,
) -> tuple[float | None, int]:
    """持有期内止盈/止损；返回 (收益, 实际持有天数)。"""
    if entry_i + 1 >= len(close):
        return None, hold
    entry_px = float(close.iloc[entry_i])
    if entry_px <= 0:
        return None, hold
    tp_px = entry_px * (1 + take_profit)
    sl_px = entry_px * (1 - stop_loss)
    last_i = min(entry_i + hold, len(close) - 1)
    for j in range(entry_i + 1, last_i + 1):
        hi = float(high.iloc[j])
        lo = float(low.iloc[j])
        if lo <= sl_px:
            return -stop_loss - fee, j - entry_i
        if hi >= tp_px:
            return take_profit - fee, j - entry_i
    exit_px = float(close.iloc[last_i])
    return exit_px / entry_px - 1.0 - fee, last_i - entry_i


def evaluate_ticker(
    ticker: str,
    df: pd.DataFrame,
    *,
    spy_bull: bool,
    mlc: MemeLongConfig,
    shares_out: float | None = None,
) -> dict[str, Any]:
    """单票评估：信号 / 回避 / 治理 / 建议仓位。"""
    tk = ticker.upper()
    row = build_latest_row(tk, df, shares_out=shares_out)
    if not row:
        return {"代码": tk, "状态": "无数据", "方向": "—", "选股理由": "行情不足"}

    avoid = detect_avoid_tags(row)
    gov_rule = mlc.governance.get(tk)
    blocked, block_reason = governance_blocked(tk, row, gov_rule)

    sid, sig_name = detect_long_signal(row, spy_bull, high_win=mlc.high_win_mode)
    hint = load_backtest_hint(tk, "S8U" if mlc.high_win_mode else "S8")

    feat_txt = (
        f"量比{float(row.get('vol_ratio', 0)):.1f} "
        f"5日{float(row.get('ret_5d', 0))*100:+.1f}% "
        f"20日{float(row.get('ret_20d', 0))*100:+.1f}% "
        f"MA50{'上' if row.get('above_ma50') else '下'}"
    )

    weight = mlc.position_weights.get(tk, mlc.alloc_pct)
    use_path = mlc.exit_mode == "path_tp" and mlc.high_win_mode

    if mlc.bull_only and not spy_bull:
        return {
            "代码": tk,
            "状态": "观望",
            "方向": "—",
            "子策略": "",
            "选股理由": f"弱市关闭 meme 规律做多 · {feat_txt}",
            "回避信号": "、".join(avoid) if avoid else "—",
            "建议仓位": "",
            **row,
        }

    if blocked:
        return {
            "代码": tk,
            "状态": "观望",
            "方向": "—",
            "子策略": "",
            "选股理由": f"治理过滤：{block_reason} · {feat_txt}",
            "回避信号": "、".join(avoid) if avoid else "—",
            "建议仓位": "",
            **row,
        }

    if avoid and sid and not mlc.high_win_mode:
        return {
            "代码": tk,
            "状态": "观望",
            "方向": "—",
            "子策略": sid,
            "选股理由": f"有多头信号但命中回避：{'、'.join(avoid)} · {feat_txt}",
            "回避信号": "、".join(avoid),
            "建议仓位": "",
            **row,
        }

    if sid:
        hold = mlc.hold_days
        exit_txt = (
            f"5日内触达+{mlc.path_take_profit_pct:.0%}止盈/"
            f"-{mlc.path_stop_loss_pct:.0%}止损（路径价）"
            if use_path
            else (
                f"持有{hold}日或触达止盈{mlc.take_profit_pct:.0%}/"
                f"止损{mlc.stop_loss_pct:.0%}"
                if mlc.use_bracket_exit
                else f"持有{hold}个交易日后收盘卖出"
            )
        )
        reason = (
            f"{tk} {sig_name}({sid}) · {feat_txt} · "
            f"{exit_txt} · 仓位{weight:.0%}"
        )
        if hint:
            reason += f" · 回测{hint}"
        return {
            "代码": tk,
            "状态": "可开仓",
            "方向": "做多",
            "子策略": sid,
            "选股理由": reason,
            "买进时机": "今日收盘买入",
            "卖出时机": exit_txt,
            "建议仓位": f"{weight:.0%}",
            "回避信号": "—",
            **row,
        }

    reason = f"未触发 Ultra80 精英条件 · {feat_txt}"
    if avoid:
        reason += f" · 回避：{'、'.join(avoid)}"
    return {
        "代码": tk,
        "状态": "观望",
        "方向": "—",
        "子策略": "",
        "选股理由": reason,
        "回避信号": "、".join(avoid) if avoid else "—",
        "建议仓位": "",
        **row,
    }


def scan_meme_long(
    batch: dict[str, pd.DataFrame],
    *,
    spy_bull: bool,
    mlc: MemeLongConfig,
) -> list[dict[str, Any]]:
    """扫描 meme 规律做多池。"""
    if not mlc.enabled:
        return []
    mod = (
        "规律·Ultra80准入" if mlc.ticker_source == "oos_approved"
        else ("规律·Ultra80" if mlc.high_win_mode else "规律·纯多头")
    )
    acct = "Meme S8U" if mlc.high_win_mode else "Meme S8"
    rows: list[dict[str, Any]] = []
    shares = fetch_shares(mlc.tickers)
    for tk in mlc.tickers:
        df = batch.get(tk)
        if df is None or df.empty:
            rows.append({
                "模块": mod,
                "账户": acct,
                "代码": tk,
                "状态": "无数据",
                "方向": "—",
                "选股理由": f"{tk} 无行情",
            })
            continue
        ev = evaluate_ticker(tk, df, spy_bull=spy_bull, mlc=mlc, shares_out=shares.get(tk))
        rows.append({
            "模块": mod,
            "账户": acct,
            "代码": ev.get("代码", tk),
            "状态": ev.get("状态", "观望"),
            "方向": ev.get("方向", "—"),
            "子策略": ev.get("子策略", ""),
            "选股理由": ev.get("选股理由", ""),
            "买进时机": ev.get("买进时机", ""),
            "卖出时机": ev.get("卖出时机", ""),
            "建议仓位": ev.get("建议仓位", ""),
            "回避信号": ev.get("回避信号", ""),
            "量比": round(float(ev.get("vol_ratio", 0) or 0), 2),
        })
    return rows
