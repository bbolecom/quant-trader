"""高胜率(≥80%)选股 · 回测统计挂载与过滤。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class BacktestStats:
    win_rate: float
    ann_return: float | None = None
    max_dd: float | None = None
    sample_n: int | None = None
    source: str = ""
    label: str = ""

    @property
    def passes(self) -> bool:
        return self.win_rate >= 0.80

    def as_dict(self) -> dict[str, Any]:
        return {
            "历史胜率": round(self.win_rate, 4),
            "历史年化": round(self.ann_return, 4) if self.ann_return is not None else None,
            "最大回撤": round(self.max_dd, 4) if self.max_dd is not None else None,
            "回测样本": self.sample_n,
            "回测来源": self.source,
        }

    def fmt_line(self) -> str:
        ann = f"年化{self.ann_return * 100:.1f}%" if self.ann_return is not None else "年化—"
        dd = f"回撤{self.max_dd * 100:.1f}%" if self.max_dd is not None else "回撤—"
        return f"胜率{self.win_rate:.0%} · {ann} · {dd}"


class StatsStore:
    """加载各策略历史回测锚点。"""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or ROOT
        self._fleet: dict[str, BacktestStats] = {}
        self._s8u: dict[str, BacktestStats] = {}
        self._gain15_surge: dict[str, BacktestStats] = {}
        self._gain15_drop: dict[str, BacktestStats] = {}
        self._flow_patterns: dict[str, BacktestStats] = {}
        self._module_defaults: dict[str, BacktestStats] = {}
        self._load_all()

    def _read_json(self, rel: str) -> dict | None:
        p = self.root / rel
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def _load_all(self) -> None:
        from research.strategy_ranker import CATALOG

        for m in CATALOG:
            if m.win_rate >= 0.80 and m.category != "avoid":
                self._module_defaults[m.id] = BacktestStats(
                    m.win_rate, m.ann_return, m.max_dd, source="strategy_ranker", label=m.name,
                )

        self._module_defaults["call_spread"] = BacktestStats(
            0.88, 0.28, -0.20, source="strategy_ranker", label="卖看涨价差",
        )
        self._module_defaults["tier_a_csp"] = BacktestStats(
            0.966, 0.567, -0.052, source="strategy_ranker", label="Tier A CSP",
        )
        self._module_defaults["weekly_soup"] = BacktestStats(
            0.87, 0.22, -0.25, source="strategy_ranker", label="周铁鹰",
        )

        mr = self._read_json("research/mean_reversion_dip_best.json")
        if mr and mr.get("oos"):
            o = mr["oos"]
            self._module_defaults["mean_reversion_dip"] = BacktestStats(
                float(o.get("win_rate", 0)),
                float(o.get("cagr", 0)),
                float(o.get("max_dd", 0)),
                int(o.get("n_trades", 0) or 0),
                source="mean_reversion_dip_best.json(OOS)",
                label="均值回归买跌",
            )

        fs = self._read_json("research/flow_strategy_backtest.json")
        if fs:
            wr = float(fs.get("笔胜率") or fs.get("日胜率") or 0)
            self._module_defaults["flow_strategy"] = BacktestStats(
                wr,
                float(fs.get("年化收益率", 0)),
                float(fs.get("最大回撤", 0)),
                int(fs.get("总笔数", 0) or 0),
                source="flow_strategy_backtest.json",
                label=str(fs.get("strategy", "flow")),
            )

        fleet = self._read_json("research/screen_fleet_stats.json")
        for acct in (fleet or {}).get("accounts") or []:
            tk = self._ticker_from_role(acct.get("role", ""), acct.get("description", ""))
            st = acct.get("stats") or {}
            if not tk:
                continue
            self._fleet[tk.upper()] = BacktestStats(
                float(st.get("trade_win_rate") or st.get("period_win_rate") or 0),
                float(st.get("ann_return", 0)),
                float(st.get("max_dd", 0)),
                int(st.get("rebalance_count", 0) or 0),
                source="screen_fleet_stats.json",
                label=str(acct.get("role", tk)),
            )

        s8u = self._read_json("research/s8u_approved_tickers.json")
        for row in (s8u or {}).get("details") or []:
            tk = str(row.get("代码", "")).upper()
            if not tk:
                continue
            self._s8u[tk] = BacktestStats(
                float(row.get("胜率") or row.get("oos_win") or 0),
                float(row.get("年化", 0)),
                float(row.get("最大回撤", 0)),
                int(row.get("oos_n") or row.get("笔数") or 0),
                source="s8u_approved_tickers.json",
                label="S8U OOS",
            )
        if s8u and s8u.get("pool_stats"):
            ps = s8u["pool_stats"]
            self._module_defaults["meme_ultra80_pool"] = BacktestStats(
                float(ps.get("weighted_win_rate", 0)),
                None,
                None,
                int(ps.get("total_oos_trades", 0) or 0),
                source="s8u_approved_tickers.json",
                label="Ultra80池",
            )

        g15 = self._read_json("research/gain15_rules_80pct.json")
        self._gain15_by_name: dict[str, BacktestStats] = {}
        for row in (g15 or {}).get("surge_rules_80plus") or []:
            name = str(row.get("rule", ""))
            st = BacktestStats(
                float(row.get("surge_rate", 0)),
                self._ann_proxy_from_avg(row.get("avg_fwd_5d_pct")),
                None,
                int(row.get("n", 0)),
                source="gain15_rules_80pct.json",
                label=name,
            )
            self._gain15_by_name[name] = st
            self._gain15_surge[name] = st
        for row in (g15 or {}).get("drop_rules_80plus") or []:
            name = str(row.get("rule", ""))
            st = BacktestStats(
                float(row.get("drop_rate", 0)),
                self._ann_proxy_from_avg(row.get("avg_fwd_5d_pct")),
                None,
                int(row.get("n", 0)),
                source="gain15_rules_80pct.json",
                label=name,
            )
            self._gain15_by_name[name] = st
            self._gain15_drop[name] = st

        fp = self._read_json("research/flow_pattern_stats.json")
        for pid, row in ((fp or {}).get("patterns") or {}).items():
            wr = float(row.get("win_rate_1d", 0))
            if wr >= 0.80:
                self._flow_patterns[str(pid)] = BacktestStats(
                    wr,
                    None,
                    None,
                    int(row.get("sample_n", 0) or 0),
                    source="flow_pattern_stats.json",
                    label=str(row.get("name", pid)),
                )

        mp5 = self._read_json("research/move_pattern_5d_rules.json")
        buckets = ((mp5 or {}).get("meta") or {}).get("top_buckets") or []
        if buckets:
            b = buckets[0]
            oos = float(b.get("oos_hit_rate") or 0)
            if oos >= 0.80:
                self._module_defaults["move_pattern_5d_up"] = BacktestStats(
                    oos, None, None, int(b.get("n", 0) or 0),
                    source="move_pattern_5d_rules.json",
                    label=str(b.get("desc", "5日路径"))[:40],
                )

    @staticmethod
    def _ticker_from_role(role: str, desc: str) -> str:
        for text in (role, desc):
            m = re.match(r"^([A-Z]{1,5})[·\.]", text or "")
            if m:
                return m.group(1).upper()
        return ""

    @staticmethod
    def _ann_proxy_from_avg(avg_pct: Any) -> float | None:
        if avg_pct is None:
            return None
        try:
            v = float(avg_pct)
        except (TypeError, ValueError):
            return None
        if abs(v) > 3:
            return v / 100.0 * 50
        return v / 100.0 * 12

    def resolve(self, pick: dict) -> BacktestStats | None:
        mod = str(pick.get("模块", ""))
        tk = str(pick.get("代码", "")).upper()
        direction = str(pick.get("方向", ""))

        if mod.startswith("暴涨80%"):
            rule_name = str(pick.get("规则", ""))
            if rule_name in self._gain15_by_name:
                return self._gain15_by_name[rule_name]
            rid = str(pick.get("规则ID", ""))
            hit = pick.get("历史命中率")
            if hit:
                try:
                    wr = float(str(hit).strip().replace("%", "")) / 100.0
                    return BacktestStats(wr, source="pick.历史命中率", label=rule_name or mod)
                except ValueError:
                    pass

        if mod.startswith("5×舰队") and tk in self._fleet:
            return self._fleet[tk]

        if "Ultra80" in mod or mod.startswith("规律·"):
            if tk in self._s8u:
                return self._s8u[tk]
            return self._module_defaults.get("meme_ultra80_pool")

        if mod == "资金流向组合" or mod.startswith("资金流向组合"):
            return self._module_defaults.get("flow_strategy")

        if mod.startswith("资金流向"):
            for key in ("上涨规律", "下跌规律"):
                raw = str(pick.get(key, ""))
                for pid in re.findall(r"[UD]_[SA][12]|D_OFFERING|D_B[123]", raw):
                    if pid in self._flow_patterns:
                        return self._flow_patterns[pid]
            return None

        if "卖Call" in mod or "卖Call" in str(pick.get("策略动作", "")):
            return self._module_defaults.get("call_spread")

        if mod.startswith("均值回归"):
            return self._module_defaults.get("mean_reversion_dip")

        if mod.startswith("SNDK") or "铁鹰" in mod:
            return self._module_defaults.get("weekly_soup")

        if mod.startswith("Universal") or mod.startswith("5×舰队"):
            return self._module_defaults.get("tier_a_csp")

        if mod.startswith("三腿策略·5日路径"):
            return self._module_defaults.get("move_pattern_5d_up")

        if mod.startswith("策略排名"):
            act = str(pick.get("策略动作", ""))
            for m in self._module_defaults.values():
                if m.label and m.label in act:
                    return m
            return None

        if mod.startswith("VRP"):
            return None

        if mod.startswith("日历"):
            return None

        return None


def is_placeholder_pick(pick: dict) -> bool:
    """非机会占位行：quick 跳过、无代码、扫描失败等，不挂高胜率标签。"""
    status = str(pick.get("状态", "") or "")
    ticker = str(pick.get("代码", "") or "").strip()
    direction = str(pick.get("方向", "") or "").strip()
    reason = str(pick.get("选股理由", "") or "")
    if status in {"扫描失败", "无数据"}:
        return True
    if "quick 模式跳过" in reason or "quick" in reason.lower() and "跳过" in reason:
        return True
    if ticker in {"", "—", "-", "None", "nan"}:
        return True
    if direction in {"", "—", "-"} and status != "可开仓":
        return True
    return False


# 通用合成锚点来源（非逐票/逐规则真实回测），不足以为「无真实报价」期权背书
GENERIC_ANCHOR_SOURCES = {"strategy_ranker"}


def option_lacks_real_chain(pick: dict) -> bool:
    """期权类信号但没有真实期权链报价（模型估算 / 真实链不可用）。"""
    from quant.daily_pick_push import infer_data_source, is_option_pick

    if not is_option_pick(pick):
        return False
    return infer_data_source(pick) != "真实链"


def enrich_pick(pick: dict, store: StatsStore | None = None) -> dict:
    store = store or StatsStore()
    out = dict(pick)
    if is_placeholder_pick(out):
        out["历史胜率"] = None
        out["历史年化"] = None
        out["最大回撤"] = None
        out["回测摘要"] = "占位/跳过行，不计入机会"
        out["高胜率达标"] = False
        return out
    stats = store.resolve(out)
    # 无真实期权链的期权信号，若只能挂「通用合成锚点」(如 call_spread 88%)，
    # 则不背书为高胜率——避免 MBC/MMYT 这类无流动期权的小盘票被错误贴标。
    # 逐票/逐规则的真实回测锚点（舰队 / S8U / gain15 等）仍然保留。
    if (
        stats is not None
        and stats.source in GENERIC_ANCHOR_SOURCES
        and option_lacks_real_chain(out)
    ):
        out["历史胜率"] = None
        out["历史年化"] = None
        out["最大回撤"] = None
        out["回测摘要"] = "无真实期权链，不计入高胜率"
        out["高胜率达标"] = False
        return out
    if stats:
        out.update(stats.as_dict())
        out["回测摘要"] = stats.fmt_line()
        out["高胜率达标"] = stats.passes
    else:
        out["历史胜率"] = None
        out["历史年化"] = None
        out["最大回撤"] = None
        out["回测摘要"] = "无≥80%回测锚点"
        out["高胜率达标"] = False
    return out


def enrich_picks(picks: list[dict], store: StatsStore | None = None) -> list[dict]:
    store = store or StatsStore()
    return [enrich_pick(p, store) for p in picks]


def filter_high_win_picks(
    picks: list[dict],
    *,
    min_win_rate: float = 0.80,
    actionable_only: bool = True,
) -> list[dict]:
    out: list[dict] = []
    for p in picks:
        if is_placeholder_pick(p):
            continue
        wr = p.get("历史胜率")
        if wr is None:
            continue
        try:
            wr_f = float(wr)
        except (TypeError, ValueError):
            continue
        if wr_f < min_win_rate:
            continue
        if actionable_only and p.get("状态") != "可开仓":
            continue
        out.append(p)
    return out


def build_high_win_doc(
    picks: list[dict],
    *,
    min_win_rate: float = 0.80,
    regime: dict | None = None,
) -> dict:
    store = StatsStore()
    enriched = enrich_picks(picks, store)
    # 高胜率池只收「能真正成交」的信号：期权类必须有真实期权链报价。
    # 无真实链的期权（含有真实历史回测的舰队 CSP）今天无法下单、会显示
    # 「无真实报价」，不应进入可开仓/观察池——但其回测锚点仍保留在 picks 行上。
    high_win = [
        p for p in filter_high_win_picks(enriched, min_win_rate=min_win_rate, actionable_only=True)
        if not option_lacks_real_chain(p)
    ]
    watch_high = [
        p for p in enriched
        if p.get("高胜率达标")
        and p.get("状态") != "可开仓"
        and p.get("状态") not in ("扫描失败",)
        and not is_placeholder_pick(p)
        and not option_lacks_real_chain(p)
    ]
    return {
        "min_win_rate": min_win_rate,
        "regime": regime or {},
        "total_scanned": len(picks),
        "high_win_actionable": high_win,
        "high_win_watch": watch_high[:30],
        "summary": {
            "可开仓高胜率": len(high_win),
            "观察高胜率": len(watch_high),
            "模块": sorted({str(p.get("模块", "")) for p in high_win}),
        },
        "all_enriched": enriched,
    }
