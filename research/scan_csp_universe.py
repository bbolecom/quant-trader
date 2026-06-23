"""把 CSP 稳定收租策略（Delta 0.20 + 50日均线过滤 + 50%止盈）批量回测在
「日均成交额 > 阈值」的所有票上，按稳定性/年化排序。

候选池覆盖美股全部大盘 + 热门高波动股——日成交额能到数十亿美元的票必然在其中。
权利金为 BS+VRP 近似；高波动/小盘股的回测对实盘高估更严重，仅供横向比较。
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from quant import decline_income as di  # noqa: E402

# 覆盖美股大盘 + 热门高波动 + 用户关注标的（去重）
UNIVERSE = sorted(set([
    # 大盘科技
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "GOOG", "META", "AVGO", "TSLA", "NFLX",
    "ORCL", "CRM", "ADBE", "AMD", "INTC", "QCOM", "CSCO", "TXN", "AMAT", "MU",
    "INTU", "NOW", "ARM", "LRCX", "KLAC", "TSM", "ASML", "DELL", "HPQ",
    # 热门高波动 / meme / 加密
    "MSTR", "PLTR", "COIN", "SMCI", "MARA", "RIOT", "RGTI", "SOFI", "HOOD", "NIO",
    "RIVN", "LCID", "PLUG", "AFRM", "UPST", "DKNG", "SNAP", "UBER", "ABNB", "SHOP",
    "RBLX", "DDOG", "NET", "CRWD", "SNOW", "PANW", "BABA", "PYPL", "QBTS", "IONQ",
    # 存储/半导体（用户线）
    "SNDK", "WDC", "STX",
    # 金融
    "JPM", "BAC", "WFC", "C", "GS", "MS", "V", "MA", "AXP",
    # 消费/工业/能源/医药
    "DIS", "F", "GM", "T", "VZ", "KO", "PEP", "WMT", "COST", "HD", "NKE", "MCD",
    "SBUX", "BA", "CAT", "XOM", "CVX", "OXY", "LLY", "UNH", "JNJ", "PFE", "MRNA",
    # 大成交额 ETF（参考）
    "SPY", "QQQ", "IWM", "TQQQ", "SOXL",
]))

THRESHOLD_USD = 5e9   # 日均成交额阈值（美元）
START = "2023-01-01"
END = "2026-06-17"


def main(threshold: float = THRESHOLD_USD):
    print(f"下载 {len(UNIVERSE)} 只候选股 ({START}~{END})…")
    data = yf.download(UNIVERSE, start=START, end=END, auto_adjust=True,
                       progress=False, group_by="column", threads=True)
    closes = data["Close"]
    vols = data["Volume"]

    rows = []
    for tk in UNIVERSE:
        try:
            c = closes[tk].dropna()
            v = vols[tk].dropna()
        except Exception:
            continue
        if len(c) < 120:
            continue
        dvol = float((c * v).tail(20).mean())   # 近20日均成交额（美元）
        if not np.isfinite(dvol) or dvol < threshold:
            continue
        rv = float(di.realized_vol(c).iloc[-1]) * 100
        ma50 = float(c.rolling(50).mean().iloc[-1])
        px = float(c.iloc[-1])
        bt = di.backtest_csp_income(c, delta=0.20)
        if not bt:
            continue
        rows.append({
            "代码": tk,
            "现价": round(px, 2),
            "成交额(亿$)": round(dvol / 1e8, 0),
            "RV%": round(rv, 0),
            "可开仓": "✅" if px > ma50 else "❌",
            "胜率": bt.get("胜率"),
            "年化": bt.get("年化"),
            "信息比": bt.get("信息比"),
            "最差单笔": bt.get("最差单笔"),
            "合成回撤": bt.get("合成回撤"),
            "交易数": bt.get("交易数"),
        })

    if not rows:
        print(f"没有日均成交额 > {threshold/1e8:.0f} 亿美元的标的。")
        return
    df = pd.DataFrame(rows)
    print(f"\n共 {len(df)} 只满足『日均成交额 > {threshold/1e8:.0f} 亿美元』\n")

    disp = df.sort_values("年化", ascending=False).copy()
    for col in ["胜率", "年化", "最差单笔", "合成回撤"]:
        disp[col] = disp[col].map(lambda x: f"{x:+.0%}" if pd.notna(x) else "-")
    disp["信息比"] = disp["信息比"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
    disp["成交额(亿$)"] = disp["成交额(亿$)"].map(lambda x: f"{x:,.0f}")
    print("=== 按回测年化降序 ===")
    print(disp.to_string(index=False))

    out = "research/csp_universe_scan.csv"
    df.sort_values("信息比", ascending=False).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n完整结果已存 {out}")
    return df


if __name__ == "__main__":
    th = float(sys.argv[1]) if len(sys.argv) > 1 else THRESHOLD_USD
    main(th)
