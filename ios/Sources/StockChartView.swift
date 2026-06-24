import SwiftUI

struct CandlestickChartView: View {
    let bars: [OHLCVBar]
    let ma20: [Double?]
    let ma50: [Double?]

    var body: some View {
        GeometryReader { geo in
            let w = geo.size.width
            let h = geo.size.height
            let plotH = h * 0.78
            let volH = h - plotH - 8
            let count = max(bars.count, 1)
            let barW = max(2.0, (w - 8) / CGFloat(count) * 0.72)
            let (minP, maxP) = priceBounds
            let maxVol = bars.map(\.volume).max() ?? 1

            ZStack(alignment: .topLeading) {
                // price grid
                ForEach(0..<4, id: \.self) { i in
                    let y = plotH * CGFloat(i) / 3
                    Path { p in
                        p.move(to: CGPoint(x: 0, y: y))
                        p.addLine(to: CGPoint(x: w, y: y))
                    }
                    .stroke(ThsTheme.border.opacity(0.35), lineWidth: 0.5)
                }

                // MA lines
                maLine(values: ma20, color: .yellow, width: w, height: plotH, minP: minP, maxP: maxP)
                maLine(values: ma50, color: .cyan, width: w, height: plotH, minP: minP, maxP: maxP)

                // candles
                HStack(alignment: .bottom, spacing: max(0.5, (w - CGFloat(count) * barW) / CGFloat(max(count - 1, 1)))) {
                    ForEach(bars) { bar in
                        candle(bar, barW: barW, plotH: plotH, minP: minP, maxP: maxP)
                    }
                }
                .padding(.horizontal, 4)
                .frame(height: plotH, alignment: .bottom)

                // volume
                HStack(alignment: .bottom, spacing: max(0.5, (w - CGFloat(count) * barW) / CGFloat(max(count - 1, 1)))) {
                    ForEach(bars) { bar in
                        let vh = max(1, plotH * 0.0 + volH * CGFloat(bar.volume / maxVol))
                        Rectangle()
                            .fill(bar.isUp ? ThsTheme.up.opacity(0.35) : ThsTheme.down.opacity(0.35))
                            .frame(width: barW, height: vh)
                    }
                }
                .padding(.horizontal, 4)
                .frame(maxHeight: .infinity, alignment: .bottom)
            }
        }
    }

    private var priceBounds: (Double, Double) {
        guard !bars.isEmpty else { return (0, 1) }
        var lo = bars.map(\.low).min() ?? 0
        var hi = bars.map(\.high).max() ?? 1
        for v in ma20.compactMap({ $0 }) + ma50.compactMap({ $0 }) {
            lo = min(lo, v)
            hi = max(hi, v)
        }
        let pad = (hi - lo) * 0.06
        return (lo - pad, hi + pad)
    }

    private func yPos(_ price: Double, height: CGFloat, minP: Double, maxP: Double) -> CGFloat {
        guard maxP > minP else { return height / 2 }
        let t = (price - minP) / (maxP - minP)
        return height * CGFloat(1 - t)
    }

    @ViewBuilder
    private func candle(_ bar: OHLCVBar, barW: CGFloat, plotH: CGFloat, minP: Double, maxP: Double) -> some View {
        let color = bar.isUp ? ThsTheme.up : ThsTheme.down
        let top = yPos(max(bar.open, bar.close), height: plotH, minP: minP, maxP: maxP)
        let bottom = yPos(min(bar.open, bar.close), height: plotH, minP: minP, maxP: maxP)
        let wickTop = yPos(bar.high, height: plotH, minP: minP, maxP: maxP)
        let wickBottom = yPos(bar.low, height: plotH, minP: minP, maxP: maxP)
        ZStack {
            Path { p in
                let x = barW / 2
                p.move(to: CGPoint(x: x, y: wickTop))
                p.addLine(to: CGPoint(x: x, y: wickBottom))
            }
            .stroke(color, lineWidth: 1)
            Rectangle()
                .fill(color)
                .frame(width: max(1, barW * 0.85), height: max(1, bottom - top))
                .offset(y: (top + bottom) / 2 - plotH / 2)
        }
        .frame(width: barW, height: plotH)
    }

    @ViewBuilder
    private func maLine(values: [Double?], color: Color, width: CGFloat, height: CGFloat, minP: Double, maxP: Double) -> some View {
        Path { path in
            var started = false
            let step = width / CGFloat(max(values.count - 1, 1))
            for (i, v) in values.enumerated() {
                guard let v else { continue }
                let x = CGFloat(i) * step
                let y = yPos(v, height: height, minP: minP, maxP: maxP)
                if started { path.addLine(to: CGPoint(x: x, y: y)) }
                else { path.move(to: CGPoint(x: x, y: y)); started = true }
            }
        }
        .stroke(color.opacity(0.85), lineWidth: 1.2)
    }
}

struct StockChartPanel: View {
    let ticker: String
    @StateObject private var loader = StockChartLoader()

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .firstTextBaseline) {
                Text(ticker.uppercased())
                    .font(.title3.weight(.bold))
                    .foregroundStyle(ThsTheme.textPrimary)
                if let px = loader.lastPrice {
                    Text("$\(JsonHelper.formatNum(px))")
                        .font(.headline)
                        .foregroundStyle(ThsTheme.textPrimary)
                }
                if let ch = loader.changePct {
                    Text(String(format: "%+.2f%%", ch))
                        .font(.subheadline.weight(.semibold))
                        .foregroundStyle(ch >= 0 ? ThsTheme.up : ThsTheme.down)
                }
                Spacer()
                HStack(spacing: 8) {
                    legendDot("MA20", .yellow)
                    legendDot("MA50", .cyan)
                }
            }
            if loader.isLoading && loader.bars.isEmpty {
                ProgressView()
                    .frame(maxWidth: .infinity)
                    .frame(height: 220)
            } else if let err = loader.errorMessage, loader.bars.isEmpty {
                Text("K线加载失败：\(err)")
                    .font(.caption)
                    .foregroundStyle(.orange)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.vertical, 24)
            } else if !loader.bars.isEmpty {
                CandlestickChartView(bars: loader.bars, ma20: loader.ma20, ma50: loader.ma50)
                    .frame(height: 240)
            }
        }
        .padding(14)
        .thsCard(border: ThsTheme.accent.opacity(0.2))
        .task(id: ticker) { await loader.load(ticker: ticker) }
    }

    private func legendDot(_ label: String, _ color: Color) -> some View {
        HStack(spacing: 4) {
            Circle().fill(color).frame(width: 6, height: 6)
            Text(label).font(.caption2).foregroundStyle(ThsTheme.textTertiary)
        }
    }
}
