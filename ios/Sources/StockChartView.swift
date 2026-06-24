import SwiftUI

// MARK: - 同花顺风格 K 线

struct ThsKLineChartView: View {
    let bars: [OHLCVBar]
    let ma20: [Double?]
    let ma50: [Double?]
    @Binding var selectedIndex: Int?
    @Binding var scrollOffset: Int
    let maxScrollOffset: Int

    @State private var panAnchorOffset: Int = 0
    @State private var isPanning = false

    private let priceAxisWidth: CGFloat = 44

    var body: some View {
        GeometryReader { geo in
            let totalH = geo.size.height
            let priceH = totalH * 0.72
            let volH = totalH * 0.22
            let gapH = totalH - priceH - volH
            let plotW = geo.size.width - priceAxisWidth - 4
            let (minP, maxP) = priceBounds
            let maxVol = bars.map(\.volume).max() ?? 1
            let barW = barWidth(plotW: plotW)

            HStack(spacing: 0) {
                ZStack(alignment: .topLeading) {
                    ForEach(0..<5, id: \.self) { i in
                        let y = priceH * CGFloat(i) / 4
                        Path { p in
                            p.move(to: CGPoint(x: 0, y: y))
                            p.addLine(to: CGPoint(x: plotW, y: y))
                        }
                        .stroke(ThsTheme.border.opacity(0.45), lineWidth: 0.5)
                    }

                    maPath(values: ma20, color: Color(red: 1.0, green: 0.84, blue: 0.0), plotW: plotW, plotH: priceH, minP: minP, maxP: maxP)
                    maPath(values: ma50, color: Color(red: 0.45, green: 0.72, blue: 1.0), plotW: plotW, plotH: priceH, minP: minP, maxP: maxP)

                    HStack(alignment: .bottom, spacing: barSpacing(plotW: plotW, barW: barW)) {
                        ForEach(Array(bars.enumerated()), id: \.element.id) { idx, bar in
                            candleView(bar, idx: idx, barW: barW, plotH: priceH, minP: minP, maxP: maxP)
                        }
                    }
                    .padding(.leading, 2)
                    .frame(width: plotW, height: priceH, alignment: .bottomLeading)

                    if let sel = selectedIndex, bars.indices.contains(sel), !isPanning {
                        let x = xPos(index: sel, plotW: plotW, barW: barW)
                        Path { p in
                            p.move(to: CGPoint(x: x, y: 0))
                            p.addLine(to: CGPoint(x: x, y: priceH + gapH + volH))
                        }
                        .stroke(ThsTheme.textTertiary.opacity(0.8), style: StrokeStyle(lineWidth: 0.8, dash: [4, 3]))
                    }
                }
                .frame(width: plotW, height: totalH)
                .contentShape(Rectangle())
                .highPriorityGesture(chartGesture(plotW: plotW, barW: barW))

                VStack(spacing: 0) {
                    ForEach(0..<5, id: \.self) { i in
                        let price = maxP - (maxP - minP) * Double(i) / 4
                        Text(formatPrice(price))
                            .font(.system(size: 9, weight: .medium, design: .monospaced))
                            .foregroundStyle(ThsTheme.textTertiary)
                            .frame(height: priceH / 4, alignment: .trailing)
                    }
                    Spacer(minLength: gapH + volH)
                }
                .frame(width: priceAxisWidth)
            }
            .overlay(alignment: .bottomLeading) {
                HStack(alignment: .bottom, spacing: barSpacing(plotW: plotW, barW: barW)) {
                    ForEach(Array(bars.enumerated()), id: \.element.id) { idx, bar in
                        let vh = max(1, volH * CGFloat(bar.volume / maxVol))
                        let color = bar.isUp ? ThsTheme.up : ThsTheme.down
                        Rectangle()
                            .fill(color.opacity(selectedIndex == idx && !isPanning ? 0.85 : 0.55))
                            .frame(width: max(1, barW * 0.82), height: vh)
                    }
                }
                .padding(.leading, 2)
                .frame(width: plotW, height: volH, alignment: .bottomLeading)
            }
        }
        .onChange(of: scrollOffset) { newValue in panAnchorOffset = newValue }
        .onAppear { panAnchorOffset = scrollOffset }
    }

    private func chartGesture(plotW: CGFloat, barW: CGFloat) -> some Gesture {
        DragGesture(minimumDistance: 4)
            .onChanged { value in
                let dx = value.translation.width
                let dy = value.translation.height
                if abs(dx) > abs(dy) * 0.8 || isPanning {
                    isPanning = true
                    selectedIndex = nil
                    let step = max(barW + barSpacing(plotW: plotW, barW: barW), 1)
                    let delta = Int((-dx / step).rounded())
                    scrollOffset = min(max(panAnchorOffset + delta, 0), maxScrollOffset)
                } else {
                    selectedIndex = index(at: value.location.x, plotW: plotW, barW: barW)
                }
            }
            .onEnded { value in
                if isPanning {
                    panAnchorOffset = scrollOffset
                } else if hypot(value.translation.width, value.translation.height) < 8 {
                    selectedIndex = index(at: value.location.x, plotW: plotW, barW: barW)
                }
                isPanning = false
            }
    }

    @ViewBuilder
    private func candleView(_ bar: OHLCVBar, idx: Int, barW: CGFloat, plotH: CGFloat, minP: Double, maxP: Double) -> some View {
        let up = bar.isUp
        let color = up ? ThsTheme.up : ThsTheme.down
        let bodyTop = yPos(max(bar.open, bar.close), plotH: plotH, minP: minP, maxP: maxP)
        let bodyBot = yPos(min(bar.open, bar.close), plotH: plotH, minP: minP, maxP: maxP)
        let wickTop = yPos(bar.high, plotH: plotH, minP: minP, maxP: maxP)
        let wickBot = yPos(bar.low, plotH: plotH, minP: minP, maxP: maxP)
        let bodyH = max(1, bodyBot - bodyTop)
        let highlighted = selectedIndex == idx

        ZStack {
            Path { p in
                let cx = barW / 2
                p.move(to: CGPoint(x: cx, y: wickTop))
                p.addLine(to: CGPoint(x: cx, y: wickBot))
            }
            .stroke(color, lineWidth: highlighted ? 1.4 : 1)

            if up {
                // 阳线：空心红框（同花顺经典）
                RoundedRectangle(cornerRadius: 0.5)
                    .stroke(color, lineWidth: highlighted ? 1.4 : 1)
                    .background(
                        RoundedRectangle(cornerRadius: 0.5)
                            .fill(color.opacity(highlighted ? 0.25 : 0.08))
                    )
                    .frame(width: max(2, barW * 0.78), height: bodyH)
                    .position(x: barW / 2, y: (bodyTop + bodyBot) / 2)
            } else {
                // 阴线：实心绿
                Rectangle()
                    .fill(color)
                    .frame(width: max(2, barW * 0.78), height: bodyH)
                    .position(x: barW / 2, y: (bodyTop + bodyBot) / 2)
            }
        }
        .frame(width: barW, height: plotH)
    }

    private func maPath(values: [Double?], color: Color, plotW: CGFloat, plotH: CGFloat, minP: Double, maxP: Double) -> some View {
        Path { path in
            guard bars.count > 1 else { return }
            var started = false
            let step = plotW / CGFloat(max(bars.count - 1, 1))
            for i in 0..<min(values.count, bars.count) {
                guard let v = values[i] else { continue }
                let x = CGFloat(i) * step + barWidth(plotW: plotW) / 2
                let y = yPos(v, plotH: plotH, minP: minP, maxP: maxP)
                if started { path.addLine(to: CGPoint(x: x, y: y)) }
                else { path.move(to: CGPoint(x: x, y: y)); started = true }
            }
        }
        .stroke(color, lineWidth: 1)
    }

    private var priceBounds: (Double, Double) {
        guard !bars.isEmpty else { return (0, 1) }
        var lo = bars.map(\.low).min() ?? 0
        var hi = bars.map(\.high).max() ?? 1
        for v in ma20.compactMap({ $0 }) + ma50.compactMap({ $0 }) {
            lo = min(lo, v)
            hi = max(hi, v)
        }
        let pad = max((hi - lo) * 0.05, 0.01)
        return (lo - pad, hi + pad)
    }

    private func barWidth(plotW: CGFloat) -> CGFloat {
        max(3, min(10, plotW / CGFloat(max(bars.count, 1)) * 0.68))
    }

    private func barSpacing(plotW: CGFloat, barW: CGFloat) -> CGFloat {
        guard bars.count > 1 else { return 0 }
        return max(0.5, (plotW - barW * CGFloat(bars.count)) / CGFloat(bars.count - 1))
    }

    private func yPos(_ price: Double, plotH: CGFloat, minP: Double, maxP: Double) -> CGFloat {
        guard maxP > minP else { return plotH / 2 }
        return plotH * CGFloat(1 - (price - minP) / (maxP - minP))
    }

    private func xPos(index: Int, plotW: CGFloat, barW: CGFloat) -> CGFloat {
        CGFloat(index) * (barW + barSpacing(plotW: plotW, barW: barW)) + barW / 2
    }

    private func index(at x: CGFloat, plotW: CGFloat, barW: CGFloat) -> Int {
        let spacing = barSpacing(plotW: plotW, barW: barW)
        let step = barW + spacing
        guard step > 0 else { return 0 }
        let idx = Int((x / step).rounded())
        return min(max(idx, 0), bars.count - 1)
    }

    private func formatPrice(_ v: Double) -> String {
        if v >= 100 { return String(format: "%.0f", v) }
        if v >= 10 { return String(format: "%.1f", v) }
        return String(format: "%.2f", v)
    }
}

// MARK: - 面板

struct StockChartPanel: View {
    let ticker: String
    @StateObject private var loader = StockChartLoader()
    @State private var refreshTimer: Timer?

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            periodTabs
            quoteHeader
            ohlcvStrip
            chartArea
        }
        .padding(12)
        .background(ThsTheme.elevated)
        .clipShape(RoundedRectangle(cornerRadius: 12))
        .overlay(
            RoundedRectangle(cornerRadius: 12)
                .stroke(ThsTheme.border.opacity(0.6), lineWidth: 0.5)
        )
        .task(id: "\(ticker)-\(loader.period.rawValue)") {
            await loader.load(ticker: ticker)
            startAutoRefresh()
        }
        .onDisappear {
            refreshTimer?.invalidate()
            refreshTimer = nil
        }
        .refreshable {
            await loader.load(ticker: ticker)
        }
    }

    private func startAutoRefresh() {
        refreshTimer?.invalidate()
        refreshTimer = Timer.scheduledTimer(withTimeInterval: 60, repeats: true) { _ in
            Task { @MainActor in
                await loader.load(ticker: ticker)
            }
        }
    }

    private var periodTabs: some View {
        HStack(spacing: 0) {
            ForEach(ChartPeriod.allCases) { p in
                Button {
                    loader.setPeriod(p, ticker: ticker)
                } label: {
                    Text(p.title)
                        .font(.caption.weight(loader.period == p ? .bold : .medium))
                        .foregroundStyle(loader.period == p ? ThsTheme.accent : ThsTheme.textTertiary)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 8)
                        .background(
                            loader.period == p
                                ? ThsTheme.card
                                : Color.clear
                        )
                }
                .buttonStyle(.plain)
            }
        }
        .background(ThsTheme.background.opacity(0.5))
        .clipShape(RoundedRectangle(cornerRadius: 8))
        .padding(.bottom, 10)
        .zIndex(1)
    }

    private var quoteHeader: some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            Text(ticker.uppercased())
                .font(.headline.weight(.bold))
                .foregroundStyle(ThsTheme.textPrimary)
            if let px = activePrice {
                Text("$\(JsonHelper.formatNum(px))")
                    .font(.title3.weight(.bold))
                    .foregroundStyle(priceColor)
            }
            if let ch = activeChangePct {
                Text(String(format: "%+.2f%%", ch))
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(ch >= 0 ? ThsTheme.up : ThsTheme.down)
            }
            Spacer()
            VStack(alignment: .trailing, spacing: 2) {
                if let src = loader.dataSource {
                    Text("来源 \(src.rawValue)")
                        .font(.system(size: 9))
                        .foregroundStyle(ThsTheme.textTertiary)
                }
                if let stamp = loader.cloudUpdatedAt {
                    Text("更新 \(stamp)")
                        .font(.system(size: 8))
                        .foregroundStyle(ThsTheme.textTertiary)
                }
                maLegend("MA20", Color(red: 1.0, green: 0.84, blue: 0.0), loader.displayMA20.compactMap { $0 }.last)
                maLegend("MA50", Color(red: 0.45, green: 0.72, blue: 1.0), loader.displayMA50.compactMap { $0 }.last)
            }
        }
        .padding(.bottom, 6)
    }

    private var ohlcvStrip: some View {
        Group {
            if let bar = loader.selectedBar ?? loader.displayBars.last {
                HStack(spacing: 10) {
                    Text(formatDate(bar.date))
                        .foregroundStyle(ThsTheme.textTertiary)
                    ohlcvChip("开", bar.open, bar.isUp)
                    ohlcvChip("高", bar.high, true)
                    ohlcvChip("低", bar.low, false)
                    ohlcvChip("收", bar.close, bar.isUp)
                    Text("量 \(formatVolume(bar.volume))")
                        .foregroundStyle(ThsTheme.textSecondary)
                }
                .font(.system(size: 10, weight: .medium, design: .monospaced))
                .lineLimit(1)
                .minimumScaleFactor(0.8)
                .padding(.vertical, 6)
                .padding(.horizontal, 8)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(ThsTheme.background.opacity(0.45))
                .clipShape(RoundedRectangle(cornerRadius: 6))
                .padding(.bottom, 8)
            }
        }
    }

    @ViewBuilder
    private var chartArea: some View {
        if loader.isLoading && loader.bars.isEmpty {
            ProgressView()
                .tint(ThsTheme.accent)
                .frame(maxWidth: .infinity)
                .frame(height: 260)
        } else if let err = loader.errorMessage, loader.bars.isEmpty {
            VStack(spacing: 8) {
                Image(systemName: "chart.xyaxis.line")
                    .foregroundStyle(ThsTheme.textTertiary)
                Text("K线加载失败")
                    .font(.footnote.weight(.semibold))
                Text(err)
                    .font(.caption2)
                    .foregroundStyle(ThsTheme.textTertiary)
                    .multilineTextAlignment(.center)
                Button("重试") {
                    Task { await loader.load(ticker: ticker) }
                }
                .font(.caption.weight(.semibold))
                .foregroundStyle(ThsTheme.accent)
            }
            .frame(maxWidth: .infinity)
            .frame(height: 260)
        } else if !loader.displayBars.isEmpty {
            VStack(spacing: 4) {
                if loader.maxScrollOffset > 0 {
                    Text(scrollHint)
                        .font(.system(size: 9))
                        .foregroundStyle(ThsTheme.textTertiary)
                        .frame(maxWidth: .infinity, alignment: .trailing)
                }
                ThsKLineChartView(
                    bars: loader.displayBars,
                    ma20: loader.displayMA20,
                    ma50: loader.displayMA50,
                    selectedIndex: $loader.selectedIndex,
                    scrollOffset: $loader.scrollOffset,
                    maxScrollOffset: loader.maxScrollOffset
                )
                .frame(height: 260)
            }
        }
    }

    private var scrollHint: String {
        if loader.scrollOffset == 0 {
            return "← 左滑查看更早"
        }
        if loader.scrollOffset >= loader.maxScrollOffset {
            return "已至最早 · 右滑回最新 →"
        }
        return "← 左滑更早 · 右滑更新 →"
    }

    private var activePrice: Double? {
        loader.selectedBar?.close ?? loader.lastPrice
    }

    private var activeChangePct: Double? {
        if let bar = loader.selectedBar, let idx = loader.selectedIndex, idx > 0 {
            let prev = loader.displayBars[idx - 1].close
            guard prev != 0 else { return nil }
            return (bar.close / prev - 1) * 100
        }
        return loader.changePct
    }

    private var priceColor: Color {
        guard let bar = loader.selectedBar ?? loader.displayBars.last else { return ThsTheme.textPrimary }
        return bar.isUp ? ThsTheme.up : ThsTheme.down
    }

    private func maLegend(_ label: String, _ color: Color, _ value: Double?) -> some View {
        HStack(spacing: 4) {
            Circle().fill(color).frame(width: 5, height: 5)
            Text(label)
                .font(.system(size: 9))
                .foregroundStyle(ThsTheme.textTertiary)
            if let v = value {
                Text(formatPrice(v))
                    .font(.system(size: 9, weight: .medium, design: .monospaced))
                    .foregroundStyle(color)
            }
        }
    }

    private func ohlcvChip(_ key: String, _ val: Double, _ up: Bool) -> some View {
        HStack(spacing: 2) {
            Text(key)
                .foregroundStyle(ThsTheme.textTertiary)
            Text(formatPrice(val))
                .foregroundStyle(up ? ThsTheme.up : ThsTheme.down)
        }
    }

    private func formatDate(_ d: Date) -> String {
        let f = DateFormatter()
        switch loader.period {
        case .daily:
            f.dateFormat = "MM-dd"
        case .weekly:
            f.dateFormat = "yyyy-MM-dd"
        case .monthly:
            f.dateFormat = "yyyy-MM"
        }
        return f.string(from: d)
    }

    private func formatPrice(_ v: Double) -> String {
        if v >= 100 { return String(format: "%.1f", v) }
        return String(format: "%.2f", v)
    }

    private func formatVolume(_ v: Double) -> String {
        if v >= 1_000_000 { return String(format: "%.1fM", v / 1_000_000) }
        if v >= 1_000 { return String(format: "%.0fK", v / 1_000) }
        return String(format: "%.0f", v)
    }
}