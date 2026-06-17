"""短线策略搜索脚本：拉取并缓存候选池数据，跑稳健性搜索，输出排行榜 CSV。

用法：
    python search_short_term.py            # 默认 1.5 年、缓存到 _short_term_cache.pkl
    python search_short_term.py --no-short # 仅做多
    python search_short_term.py --refresh  # 忽略缓存重新拉取

数据走当前配置的数据源（如 Polygon，免费档限速会较慢，已带自动退避重试）。
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import warnings

import pandas as pd

warnings.filterwarnings("ignore")

from quant.strategy_search import DEFAULT_SHORT_TERM_POOL as CANDIDATE_POOL

CACHE = "_short_term_cache.pkl"


def load_data(start: str, end: str, refresh: bool) -> dict[str, pd.DataFrame]:
    from quant.providers import reset_provider_cache
    from quant.providers.config import _load_secrets_file
    _load_secrets_file.cache_clear()
    reset_provider_cache()
    from quant.data import get_data_source_info, fetch_history_batch

    print("数据源:", get_data_source_info()["label"])
    if os.path.exists(CACHE) and not refresh:
        data = pickle.load(open(CACHE, "rb"))
        print(f"用缓存数据 {CACHE}，{len(data)} 只票")
        return data
    print(f"拉取 {len(CANDIDATE_POOL)} 只票 {start}~{end}（免费档限速会较慢）...")
    data = fetch_history_batch(CANDIDATE_POOL, start, end)
    pickle.dump(data, open(CACHE, "wb"))
    print(f"成功拉取 {len(data)} 只：{sorted(data.keys())}")
    return data


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=(pd.Timestamp.today() - pd.DateOffset(months=18)).strftime("%Y-%m-%d"))
    ap.add_argument("--end", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    ap.add_argument("--no-short", action="store_true", help="仅做多")
    ap.add_argument("--refresh", action="store_true", help="忽略缓存重拉数据")
    ap.add_argument("--forward", type=int, default=20, help="评估窗口(交易日)")
    ap.add_argument("--out", default="short_term_ranking.csv")
    args = ap.parse_args()

    data = load_data(args.start, args.end, args.refresh)
    if not data:
        print("无可用数据，退出。")
        return 1

    from quant import strategy_search as ss

    def progress(k, total, combo):
        if k % 10 == 0 or k == total:
            print(f"  [{k}/{total}] {combo.label}")

    print("开始搜索...")
    table, results = ss.search_short_term(
        data, forward_days=args.forward,
        include_short=not args.no_short, progress=progress,
    )
    if table.empty:
        print("没有产生有效组合（数据不足？）。")
        return 1

    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 40)
    show_cols = [
        "稳健通过", "样本内评分", "思路", "交易策略", "参数", "看盘日", "调仓日", "方向",
        "内-胜率", "内-平均收益%", "内-信息比", "外-笔数", "外-胜率", "外-平均收益%", "外-信息比", "外-盈亏比", "外-最差单笔%",
    ]
    show_cols = [c for c in show_cols if c in table.columns]
    print("\n===== 短线策略排行榜（前 12，按稳健通过 + 样本内评分）=====")
    print(table[show_cols].head(12).to_string(index=False))

    table.drop(columns=["_id"], errors="ignore").to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"\n完整排行榜已写入 {args.out}（共 {len(table)} 个组合）")

    robust = table[table["稳健通过"] == "✅"]
    if not robust.empty:
        best_id = robust.iloc[0]["_id"]
        best = next(r["combo"] for r in results if r["combo"].id == best_id)
        print("\n===== 最优稳健组合 =====")
        print("组合:", best.label)
        print("过滤:", best.filters)
    else:
        print("\n⚠️ 没有组合通过样本外验证——说明该池+窗口下短线难有稳定 alpha，建议换池或拉长持有窗口。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
