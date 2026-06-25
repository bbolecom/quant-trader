"""Gainer10+ 分板块多空策略 · 5年事件 + 板块网格优化。

v3 分板块：每板块独立多/空规则（research/gainer10_sector_rules.json）
组合回测（L≥60%+avg≥3 · S≥80%）：胜率 ~75%，空头 ~79%，CAGR ~12%
组合回测（L≥58%+avg≥2 · S≥80%）：胜率 ~70%，CAGR ~29%
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant.data import fetch_history
from quant.providers import DataConfig, get_provider, reset_provider_cache
from quant.screener import fetch_gainer_universe_live, sector_cn

ROOT = Path(__file__).resolve().parents[1]
SECTOR_CACHE = ROOT / "research" / "sector_map.json"
SECTOR_RULES_PATH = ROOT / "research" / "gainer10_sector_rules.json"

WEAK_SECTORS = frozenset({
    "Healthcare", "Communication Services", "Consumer Cyclical", "Consumer Defensive",
})

SECTOR_CN = {
    "Technology": "科技",
    "Financial Services": "金融",
    "Healthcare": "医疗",
    "Consumer Cyclical": "可选消费",
    "Consumer Defensive": "必需消费",
    "Communication Services": "通信",
    "Industrials": "工业",
    "Energy": "能源",
    "Basic Materials": "原材料",
    "Real Estate": "房地产",
    "Utilities": "公用事业",
    "Unknown": "未知",
}

MODE_PRESETS: dict[str, dict[str, Any]] = {
    "high_win": {
        "sector_mode": True,
        "sector_only": True,
        "sector_long_min_win": 60.0,
        "sector_long_min_avg": 3.0,
        "sector_short_min_win": 80.0,
    },
    "balanced": {
        "sector_mode": True,
        "sector_only": False,
        "sector_long_min_win": 58.0,
        "sector_long_min_avg": 2.0,
        "sector_short_min_win": 80.0,
    },
    "legacy": {
        "sector_mode": False,
        "sector_only": False,
    },
}


@dataclass
class Gainer10Config:
    min_gain_pct: float = 10.0
    min_dvol_m: float = 100.0
    gainer_count: int = 250
    # 规则 A（多头·进取）
    a_ext20_min: float = 0.40
    a_rsi_min: float = 75.0
    a_sectors: tuple[str, ...] = ("Technology",)
    # 规则 B（多头·均衡）
    b_gap_min: float = 0.05
    b_ext20_min: float = 0.20
    b_vol_x_min: float = 2.0
    b_entry_dip_pct: float = 5.0
    # 规则 S（空头·衰竭）
    short_enabled: bool = True
    s_weak_only: bool = True
    s_gap_max: float = 0.0
    s_ext20_max: float = 0.0
    s_hold_days: int = 10
    s_take_profit_pct: float | None = None
    s_stop_loss_pct: float | None = None
    sector_mode: bool = True
    sector_long_min_win: float = 60.0
    sector_long_min_avg: float = 3.0
    sector_short_min_win: float = 80.0
    sector_only: bool = True
    mode: str = "high_win"
    require_bull: bool = True
    take_profit_pct: float = 20.0
    stop_loss_pct: float = 10.0
    hold_days: int = 20
    max_picks: int = 12


@dataclass
class Gainer10Signal:
    代码: str
    信号: str          # 续涨A / 续涨B / 做空S
    动作: str
    现价: float
    涨幅_pct: float
    成交额M: float
    板块: str
    跳空_pct: float
    乖离20_pct: float
    RSI: float
    量比: float
    收盘强度: float
    方向: str = "long"   # long / short
    限价入场: float | None = None
    止盈_pct: float | None = None
    止损_pct: float | None = None
    持有天: int = 20
    规则说明: str = ""
    历史胜率: str = ""
    历史均收益: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def config_from_dict(raw: dict[str, Any]) -> Gainer10Config:
    """从 JSON 配置构建；支持 mode 预设（high_win / balanced / legacy）。"""
    merged = dict(raw)
    preset = MODE_PRESETS.get(str(merged.get("mode", "high_win")), MODE_PRESETS["high_win"])
    for k, v in preset.items():
        merged.setdefault(k, v)
    return Gainer10Config(
        min_gain_pct=float(merged.get("min_gain_pct", 10.0)),
        min_dvol_m=float(merged.get("min_dvol_m", 100.0)),
        gainer_count=int(merged.get("gainer_count", 250)),
        a_ext20_min=float(merged.get("a_ext20_min", 0.40)),
        a_rsi_min=float(merged.get("a_rsi_min", 75.0)),
        a_sectors=tuple(merged.get("a_sectors", ("Technology",))),
        b_gap_min=float(merged.get("b_gap_min", 0.05)),
        b_ext20_min=float(merged.get("b_ext20_min", 0.20)),
        b_vol_x_min=float(merged.get("b_vol_x_min", 2.0)),
        b_entry_dip_pct=float(merged.get("b_entry_dip_pct", 5.0)),
        short_enabled=bool(merged.get("short_enabled", True)),
        s_weak_only=bool(merged.get("s_weak_only", True)),
        s_gap_max=float(merged.get("s_gap_max", 0.0)),
        s_ext20_max=float(merged.get("s_ext20_max", 0.0)),
        s_hold_days=int(merged.get("s_hold_days", 10)),
        s_take_profit_pct=merged.get("s_take_profit_pct"),
        s_stop_loss_pct=merged.get("s_stop_loss_pct"),
        sector_mode=bool(merged.get("sector_mode", True)),
        sector_long_min_win=float(merged.get("sector_long_min_win", 60.0)),
        sector_long_min_avg=float(merged.get("sector_long_min_avg", 3.0)),
        sector_short_min_win=float(merged.get("sector_short_min_win", 80.0)),
        sector_only=bool(merged.get("sector_only", True)),
        mode=str(merged.get("mode", "high_win")),
        require_bull=bool(merged.get("require_bull", True)),
        take_profit_pct=float(merged.get("take_profit_pct", 20.0)),
        stop_loss_pct=float(merged.get("stop_loss_pct", 10.0)),
        hold_days=int(merged.get("hold_days", 20)),
        max_picks=int(merged.get("max_picks", 12)),
    )


def _rsi(c: pd.Series, n: int = 14) -> float:
    d = c.diff()
    g = d.clip(lower=0).rolling(n).mean()
    l = (-d.clip(upper=0)).rolling(n).mean()
    rs = g / l.replace(0, np.nan)
    v = float((100 - 100 / (1 + rs)).iloc[-1])
    return v if np.isfinite(v) else 50.0


def _features(df: pd.DataFrame) -> dict[str, float] | None:
    if df is None or len(df) < 25:
        return None
    o, h, l, c, v = (df[k].astype(float) for k in ["Open", "High", "Low", "Close", "Volume"])
    prev = float(c.iloc[-2])
    last = float(c.iloc[-1])
    chg = last / prev - 1
    rng = float(h.iloc[-1] - l.iloc[-1])
    clv = 0.5 if rng <= 0 else ((last - l.iloc[-1]) - (h.iloc[-1] - last)) / rng
    gap = float(o.iloc[-1]) / prev - 1
    vma = float(v.rolling(20).mean().iloc[-1])
    vol_x = float(v.iloc[-1]) / vma if vma > 0 else 1.0
    ext20 = last / float(c.iloc[-21]) - 1 if len(c) > 21 else 0.0
    return {
        "close": last, "chg": chg, "clv": clv, "gap": gap,
        "vol_x": vol_x, "ext20": ext20, "rsi": _rsi(c),
        "dvol_m": last * float(v.iloc[-1]) / 1e6,
    }


def market_regime() -> dict[str, Any]:
    try:
        end = date.today().isoformat()
        start = (date.today() - timedelta(days=60)).isoformat()
        spy = fetch_history("SPY", start=start, end=end)
        px = float(spy["Close"].iloc[-1])
        ma20 = float(spy["Close"].tail(20).mean())
        return {"SPY": round(px, 2), "MA20": round(ma20, 2), "站上MA20": px >= ma20}
    except Exception:  # noqa: BLE001
        return {"SPY": None, "MA20": None, "站上MA20": True}


def load_sector_map() -> dict[str, str]:
    if SECTOR_CACHE.exists():
        return json.loads(SECTOR_CACHE.read_text(encoding="utf-8"))
    return {}


def load_active_sector_rules(cfg: Gainer10Config) -> tuple[list[dict], list[dict]]:
    """加载分板块规则，按回测胜率降序。"""
    if not cfg.sector_mode or not SECTOR_RULES_PATH.exists():
        return [], []
    raw = json.loads(SECTOR_RULES_PATH.read_text(encoding="utf-8"))
    long_r = [
        r for r in raw.get("long_rules", [])
        if r.get("win_pct", 0) >= cfg.sector_long_min_win
        and r.get("avg_pct", 0) >= cfg.sector_long_min_avg
    ]
    short_r = [
        r for r in raw.get("short_rules", [])
        if r.get("win_pct", 0) >= cfg.sector_short_min_win
    ]
    long_r.sort(key=lambda x: -x.get("win_pct", 0))
    short_r.sort(key=lambda x: -x.get("win_pct", 0))
    return long_r, short_r


def _match_sector_rule(f: dict[str, float], sec: str, bull: bool, rule: dict) -> bool:
    if rule.get("sector") != sec:
        return False
    if rule.get("require_bull") and not bull:
        return False
    if rule.get("gap_min") is not None and f["gap"] < rule["gap_min"]:
        return False
    if rule.get("gap_max") is not None and f["gap"] > rule["gap_max"]:
        return False
    if rule.get("ext20_min") is not None and f["ext20"] < rule["ext20_min"]:
        return False
    if rule.get("ext20_max") is not None and f["ext20"] > rule["ext20_max"]:
        return False
    if rule.get("rsi_min") is not None and f["rsi"] < rule["rsi_min"]:
        return False
    if rule.get("volx_min") is not None and f["vol_x"] < rule["volx_min"]:
        return False
    if rule.get("clv_max") is not None and f["clv"] > rule["clv_max"]:
        return False
    return True


def _signal_from_sector_rule(f: dict[str, float], sec: str, rule: dict) -> Gainer10Signal:
    cn = rule.get("sector_cn") or SECTOR_CN.get(sec, sec)
    side = rule.get("side", "long")
    hold = int(rule.get("hold", 20))
    entry_dip = float(rule.get("entry_dip") or 0)
    tp, sl = rule.get("tp"), rule.get("sl")
    if side == "long":
        tag = f"续涨·{cn}"
        entry = round(f["close"] * (1 - entry_dip), 2) if entry_dip > 0 else round(f["close"] * (1 - 0.03), 2)
        if entry_dip > 0:
            action = f"挂回踩{entry_dip*100:.0f}%限价·hold{hold}天"
        else:
            action = f"追收盘·hold{hold}天"
        if tp and sl:
            action = f"{action}·TP{float(tp)*100:.0f}%/SL{float(sl)*100:.0f}%"
        return Gainer10Signal(
            代码="", 信号=tag, 方向="long", 动作=action,
            现价=f["close"], 涨幅_pct=round(f["chg"] * 100, 1),
            成交额M=round(f["dvol_m"], 0), 板块=cn,
            跳空_pct=round(f["gap"] * 100, 1), 乖离20_pct=round(f["ext20"] * 100, 1),
            RSI=round(f["rsi"], 0), 量比=round(f["vol_x"], 2), 收盘强度=round(f["clv"], 2),
            限价入场=entry,
            止盈_pct=float(tp) * 100 if tp else 25.0,
            止损_pct=float(sl) * 100 if sl else 12.0,
            持有天=hold,
            规则说明=rule.get("filter", ""),
            历史胜率=f"~{rule.get('win_pct', 0):.0f}%",
            历史均收益=f"~+{rule.get('avg_pct', 0):.1f}%/笔",
        )
    tag = f"做空·{cn}"
    exit_note = f"hold{hold}"
    if tp and sl:
        exit_note = f"TP{float(tp)*100:.0f}%/SL{float(sl)*100:.0f}% · hold{hold}"
    return Gainer10Signal(
        代码="", 信号=tag, 方向="short",
        动作=f"爆涨日收盘做空·{exit_note}",
        现价=f["close"], 涨幅_pct=round(f["chg"] * 100, 1),
        成交额M=round(f["dvol_m"], 0), 板块=cn,
        跳空_pct=round(f["gap"] * 100, 1), 乖离20_pct=round(f["ext20"] * 100, 1),
        RSI=round(f["rsi"], 0), 量比=round(f["vol_x"], 2), 收盘强度=round(f["clv"], 2),
        限价入场=f["close"],
        止盈_pct=float(tp) * 100 if tp else None,
        止损_pct=float(sl) * 100 if sl else None,
        持有天=hold,
        规则说明=rule.get("filter", ""),
        历史胜率=f"~{rule.get('win_pct', 0):.0f}%",
        历史均收益=f"~+{rule.get('avg_pct', 0):.1f}%/笔",
    )


def classify_sector(
    f: dict[str, float], sec: str, bull: bool, cfg: Gainer10Config,
    long_rules: list[dict], short_rules: list[dict],
) -> Gainer10Signal | None:
    for rule in long_rules:
        if _match_sector_rule(f, sec, bull, rule):
            return _signal_from_sector_rule(f, sec, rule)
    for rule in short_rules:
        if _match_sector_rule(f, sec, bull, rule):
            return _signal_from_sector_rule(f, sec, rule)
    return None


def _match_short(f: dict[str, float], sec: str, cfg: Gainer10Config) -> bool:
    if not cfg.short_enabled:
        return False
    if f["gap"] > cfg.s_gap_max or f["ext20"] > cfg.s_ext20_max:
        return False
    if cfg.s_weak_only and sec not in WEAK_SECTORS:
        return False
    return True


def classify(f: dict[str, float], sec: str, bull: bool, cfg: Gainer10Config) -> Gainer10Signal | None:
    long_rules, short_rules = load_active_sector_rules(cfg)
    if cfg.sector_mode and (long_rules or short_rules):
        sig = classify_sector(f, sec, bull, cfg, long_rules, short_rules)
        if sig is not None:
            return sig
        if cfg.sector_only:
            return None
    return _classify_legacy(f, sec, bull, cfg)


def _classify_legacy(f: dict[str, float], sec: str, bull: bool, cfg: Gainer10Config) -> Gainer10Signal | None:
    dip = cfg.b_entry_dip_pct / 100.0
    tp = cfg.take_profit_pct
    sl = cfg.stop_loss_pct
    hold = cfg.hold_days
    allow_long = (not cfg.require_bull) or bull

    if allow_long and sec in cfg.a_sectors and f["ext20"] >= cfg.a_ext20_min and f["rsi"] >= cfg.a_rsi_min:
        return Gainer10Signal(
            代码="", 信号="续涨A", 方向="long",
            动作="追收盘·持有20天(宽止损12%更优)",
            现价=f["close"], 涨幅_pct=round(f["chg"] * 100, 1),
            成交额M=round(f["dvol_m"], 0), 板块=sec,
            跳空_pct=round(f["gap"] * 100, 1), 乖离20_pct=round(f["ext20"] * 100, 1),
            RSI=round(f["rsi"], 0), 量比=round(f["vol_x"], 2),
            收盘强度=round(f["clv"], 2),
            限价入场=round(f["close"] * (1 - 0.03), 2),
            止盈_pct=25.0, 止损_pct=12.0, 持有天=hold,
            规则说明="科技+乖离≥40%+RSI≥75+大盘MA20上",
            历史胜率="~62%", 历史均收益="~+18%/笔(hold20)",
        )
    if allow_long and (sec not in WEAK_SECTORS and f["gap"] >= cfg.b_gap_min
                       and f["ext20"] >= cfg.b_ext20_min and f["vol_x"] >= cfg.b_vol_x_min):
        entry = round(f["close"] * (1 - dip), 2)
        return Gainer10Signal(
            代码="", 信号="续涨B", 方向="long",
            动作=f"挂回踩{cfg.b_entry_dip_pct:.0f}%限价·TP{tp:.0f}%/SL{sl:.0f}%",
            现价=f["close"], 涨幅_pct=round(f["chg"] * 100, 1),
            成交额M=round(f["dvol_m"], 0), 板块=sec,
            跳空_pct=round(f["gap"] * 100, 1), 乖离20_pct=round(f["ext20"] * 100, 1),
            RSI=round(f["rsi"], 0), 量比=round(f["vol_x"], 2),
            收盘强度=round(f["clv"], 2),
            限价入场=entry, 止盈_pct=tp, 止损_pct=sl, 持有天=hold,
            规则说明="非弱板块+跳空≥5%+乖离≥20%+天量+大盘MA20上",
            历史胜率="~54%", 历史均收益="~+9%/笔(hold20)",
        )
    if _match_short(f, sec, cfg):
        s_tp = cfg.s_take_profit_pct
        s_sl = cfg.s_stop_loss_pct
        exit_note = f"hold{cfg.s_hold_days}"
        if s_tp is not None and s_sl is not None:
            exit_note = f"TP{s_tp:.0f}%/SL{s_sl:.0f}% · hold{cfg.s_hold_days}"
        elif s_tp is not None:
            exit_note = f"TP{s_tp:.0f}% · hold{cfg.s_hold_days}"
        return Gainer10Signal(
            代码="", 信号="做空S", 方向="short",
            动作=f"爆涨日收盘做空·{exit_note}",
            现价=f["close"], 涨幅_pct=round(f["chg"] * 100, 1),
            成交额M=round(f["dvol_m"], 0), 板块=sec,
            跳空_pct=round(f["gap"] * 100, 1), 乖离20_pct=round(f["ext20"] * 100, 1),
            RSI=round(f["rsi"], 0), 量比=round(f["vol_x"], 2),
            收盘强度=round(f["clv"], 2),
            限价入场=f["close"], 止盈_pct=s_tp, 止损_pct=s_sl, 持有天=cfg.s_hold_days,
            规则说明="弱板块+平低开+低位首爆·衰竭回落",
            历史胜率="~69%", 历史均收益="~+3.6%/笔(hold10)",
        )
    return None


def run_gainer10_scan(cfg: Gainer10Config | None = None) -> dict[str, Any]:
    cfg = cfg or Gainer10Config()
    reg = market_regime()
    bull = bool(reg.get("站上MA20", True))
    secmap = load_sector_map()

    snap = fetch_gainer_universe_live(count=cfg.gainer_count)
    if snap.empty:
        return {"date": date.today().isoformat(), "market": reg, "picks": [], "scan_stats": {}}

    snap = snap.copy()
    snap["涨幅%"] = pd.to_numeric(snap["涨幅%"], errors="coerce")
    snap = snap[snap["涨幅%"] >= cfg.min_gain_pct].sort_values("涨幅%", ascending=False)
    cands = [str(r["代码"]).upper() for _, r in snap.head(80).iterrows()]
    start = (date.today() - timedelta(days=90)).isoformat()
    end = date.today().isoformat()
    reset_provider_cache()
    yahoo = get_provider(DataConfig(provider="yahoo"))
    hist = yahoo.fetch_batch(cands, start, end)

    buy_a: list[dict] = []
    buy_b: list[dict] = []
    buy_sector: list[dict] = []
    short_s: list[dict] = []
    short_sector: list[dict] = []

    long_rules, short_rules = load_active_sector_rules(cfg)

    for _, row in snap.head(80).iterrows():
        tk = str(row["代码"]).upper()
        df = hist.get(tk)
        if df is None or df.empty:
            try:
                df = fetch_history(tk, start=start, end=end)
            except Exception:  # noqa: BLE001
                continue
        feat = _features(df)
        if feat is None or feat["dvol_m"] < cfg.min_dvol_m:
            continue
        if feat["chg"] * 100 < cfg.min_gain_pct:
            continue
        sec_en = secmap.get(tk) or str(row.get("_行业EN") or row.get("行业") or "Unknown")
        if sec_en in ("Unknown", "", "nan") and row.get("行业"):
            sec_en = str(row.get("行业"))
        sig = classify(feat, sec_en, bull, cfg)
        if sig is None:
            continue
        sig.代码 = tk
        sig.板块 = sector_cn(sec_en) if sec_en not in ("Unknown",) else sec_en
        d = sig.to_dict()
        if sig.方向 == "long":
            if sig.信号.startswith("续涨·"):
                buy_sector.append(d)
            elif sig.信号 == "续涨A":
                buy_a.append(d)
            else:
                buy_b.append(d)
        elif sig.信号.startswith("做空·"):
            short_sector.append(d)
        else:
            short_s.append(d)
        n = len(buy_a) + len(buy_b) + len(buy_sector) + len(short_s) + len(short_sector)
        if n >= cfg.max_picks:
            break

    picks = buy_sector + buy_a + buy_b + short_sector + short_s
    n_long = len(buy_sector) + len(buy_a) + len(buy_b)
    n_short = len(short_sector) + len(short_s)
    note = ""
    if cfg.require_bull and not bull:
        note = "大盘MA20下：分板块多头规则暂停；空头规则仍可用"
    bt = (json.loads(SECTOR_RULES_PATH.read_text(encoding="utf-8")).get("portfolio_high_win")
          if cfg.sector_mode and SECTOR_RULES_PATH.exists() else None)
    return {
        "date": date.today().isoformat(),
        "title": "Gainer10+ 分板块高胜率",
        "strategy_id": "gainer10",
        "market": reg,
        "note": note,
        "config": {
            "mode": cfg.mode,
            "sector_mode": cfg.sector_mode,
            "sector_only": cfg.sector_only,
            "min_gain_pct": cfg.min_gain_pct,
            "min_dvol_m": cfg.min_dvol_m,
        },
        "strategy": {
            "name": "Gainer10+ 分板块高胜率",
            "version": "3.0_sector",
            "sector_mode": cfg.sector_mode,
            "active_long_sectors": [r.get("sector_cn") for r in long_rules],
            "active_short_sectors": [r.get("sector_cn") for r in short_rules],
            "filters": {
                "long_min_win": cfg.sector_long_min_win,
                "long_min_avg": cfg.sector_long_min_avg,
                "short_min_win": cfg.sector_short_min_win,
            },
            "backtest_5y_high_win": bt or {
                "组合胜率": "74.9%",
                "多头胜率": "66.0%",
                "空头胜率": "78.5%",
                "CAGR": "11.8%",
                "条件": "L≥60%+avg≥3 · S≥80%",
            },
            "backtest_5y_balanced": {
                "组合胜率": "69.7%",
                "CAGR": "28.6%",
                "条件": "L≥58%+avg≥2 · S≥80%",
            },
        },
        "scan_stats": {
            "扫描": int(len(snap)),
            "续涨A": len(buy_a),
            "续涨B": len(buy_b),
            "分板块多": len(buy_sector),
            "做空S": len(short_s),
            "分板块空": len(short_sector),
            "可开仓": n_long + n_short,
        },
        "buy_a": buy_a,
        "buy_b": buy_b,
        "buy_sector": buy_sector,
        "short_s": short_s,
        "short_sector": short_sector,
        "avoid_c": short_sector + short_s,
        "picks": picks,
        "signals": picks,
    }
