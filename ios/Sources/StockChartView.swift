import SwiftUI

// MARK: - 同花顺 K 线色板

private enum ThsChartStyle {
    static let canvas = Color(red: 0.02, green: 0.024, blue: 0.035)
    static let grid = Color(red: 0.16, green: 0.17, blue: 0.20)
    static let divider = Color(red: 0.22, green: 0.24, blue: 0.28)
    static let crosshair = Color.white.opacity(0.45)
    static let ma5 = Color(red: 0.90, green: 0.90, blue: 0.92)
    static let ma10 = Color(red: 1.0, green: 0.76, blue: 0.05)
    static let ma20 = Color(red: 0.88, green: 0.38, blue: 0.95)
    static let ma60 = Color(red: 0.30, green: 0.82, blue: 0.55)
    static let volMA5 = Color(red: 0.90, green: 0.90, blue: 0.92)   // 同花顺：量MA5 白
    static let volMA10 = Color(red: 1.0, green: 0.76, blue: 0.05)   // 同花顺：量MA10 黄
    static let axisLabel = Color(red: 0.55, green: 0.58, blue: 0.64)
    static let crosshairBox = Color(red: 0.20, green: 0.22, blue: 0.28)  // 光标轴标签底框
}

// MARK: - 同花顺风格 K 线

struct ThsKLineChartView: View {
    let bars: [OHLCVBar]
    let ma5: [Double?]
    let ma10: [Double?]
    let ma20: [Double?]
    let ma60: [Double?]
    var period: ChartPeriod = .daily
    @Binding var selectedIndex: Int?
    @Binding var scrollOffset: Int
    let maxScrollOffset: Int

    @State private var panAnchorOffset: Int = 0
    @State private var isPanning = false

    private let priceAxisWidth: CGFloat = 52
    private let volLabelH: CGFloat = 14
    private let dateAxisH: CGFloat = 16

    private var activeIndex: Int? {
        if isPanning { return nil }
        if let sel = selectedIndex, bars.indices.contains(sel) { return sel }
        return bars.isEmpty ? nil : bars.count - 1
    }

    var body: some View {
        GeometryReader { geo in
            let totalH = geo.size.height
            let bodyH = totalH - dateAxisH
            let priceH = bodyH * 0.74
            let volH = max(1, bodyH - priceH - volLabelH - 0.5)
            let volBlockH = volLabelH + volH + 0.5
            let plotW = geo.size.width - priceAxisWidth
            let (minP, maxP) = priceBounds
            let maxVol = bars.map(\.volume).max() ?? 1
            let barW = barWidth(plotW: plotW)
            let volMA5 = volumeMA(period: 5)
            let volMA10 = volumeMA(period: 10)

            ZStack(alignment: .topLeading) {
                ThsChartStyle.canvas

                HStack(spacing: 0) {
                    VStack(spacing: 0) {
                        pricePlot(
                            plotW: plotW,
                            priceH: priceH,
                            volBlockH: volBlockH,
                            minP: minP,
                            maxP: maxP,
                            barW: barW
                        )
                        Rectangle()
                            .fill(ThsChartStyle.divider)
                            .frame(height: 0.5)
                        volLabelRow(volMA5: volMA5, volMA10: volMA10)
                            .frame(height: volLabelH)
                        volPlot(
                            plotW: plotW,
                            volH: volH,
                            maxVol: maxVol,
                            barW: barW,
                            volMA5: volMA5,
                            volMA10: volMA10
                        )
                        dateAxis(plotW: plotW, barW: barW)
                            .frame(width: plotW, height: dateAxisH)
                    }
                    .frame(width: plotW)

                    priceAxis(
                        priceH: priceH,
                        volBlockH: volLabelH + volH + 0.5 + dateAxisH,
                        minP: minP,
                        maxP: maxP
                    )
                    .frame(width: priceAxisWidth)
                }
            }
        }
        .background(ThsChartStyle.canvas)
        .onChange(of: scrollOffset) { newValue in panAnchorOffset = newValue }
        .onAppear { panAnchorOffset = scrollOffset }
    }

    // MARK: - 主图

    @ViewBuilder
    private func pricePlot(plotW: CGFloat, priceH: CGFloat, volBlockH: CGFloat, minP: Double, maxP: Double, barW: CGFloat) -> some View {
        ZStack(alignment: .topLeading) {
            ForEach(0..<5, id: \.self) { i in
                let y = priceH * CGFloat(i) / 4
                Path { p in
                    p.move(to: CGPoint(x: 0, y: y))
                    p.addLine(to: CGPoint(x: plotW, y: y))
                }
                .stroke(ThsChartStyle.grid.opacity(i == 0 || i == 4 ? 0.9 : 0.55), lineWidth: 0.5)
            }

            maPath(values: ma5, color: ThsChartStyle.ma5, plotW: plotW, plotH: priceH, minP: minP, maxP: maxP, barW: barW)
            maPath(values: ma10, color: ThsChartStyle.ma10, plotW: plotW, plotH: priceH, minP: minP, maxP: maxP, barW: barW)
            maPath(values: ma20, color: ThsChartStyle.ma20, plotW: plotW, plotH: priceH, minP: minP, maxP: maxP, barW: barW)
            maPath(values: ma60, color: ThsChartStyle.ma60, plotW: plotW, plotH: priceH, minP: minP, maxP: maxP, barW: barW)

            HStack(alignment: .bottom, spacing: barSpacing(plotW: plotW, barW: barW)) {
                ForEach(Array(bars.enumerated()), id: \.element.id) { idx, bar in
                    candleView(bar, idx: idx, barW: barW, plotH: priceH, minP: minP, maxP: maxP)
                }
            }
            .padding(.leading, 1)
            .frame(width: plotW, height: priceH, alignment: .bottomLeading)

            if let last = bars.last, selectedIndex == nil || isPanning {
                latestPriceLine(close: last.close, isUp: last.isUp, plotW: plotW, plotH: priceH, minP: minP, maxP: maxP)
            }

            if let idx = activeIndex, bars.indices.contains(idx) {
                crosshair(index: idx, plotW: plotW, priceH: priceH, volBlockH: volBlockH, minP: minP, maxP: maxP, barW: barW)
            }
        }
        .frame(width: plotW, height: priceH)
        .contentShape(Rectangle())
        .highPriorityGesture(chartGesture(plotW: plotW, barW: barW))
    }

    private func volLabelRow(volMA5: [Double?], volMA10: [Double?]) -> some View {
        let idx = activeIndex
        let v = idx.flatMap { bars.indices.contains($0) ? bars[$0].volume : nil }
        let m5 = idx.flatMap { volMA5.indices.contains($0) ? volMA5[$0] : nil } ?? volMA5.compactMap { $0 }.last
        let m10 = idx.flatMap { volMA10.indices.contains($0) ? volMA10[$0] : nil } ?? volMA10.compactMap { $0 }.last
        return HStack(spacing: 8) {
            Text("VOL\(v.map { " " + formatVol($0) } ?? "")")
                .foregroundStyle(ThsTheme.textTertiary)
            if let m5 = m5 {
                Text("MA5 \(formatVol(m5))").foregroundStyle(ThsChartStyle.volMA5)
            }
            if let m10 = m10 {
                Text("MA10 \(formatVol(m10))").foregroundStyle(ThsChartStyle.volMA10)
            }
            Spacer()
        }
        .font(.system(size: 9, weight: .bold, design: .monospaced))
        .lineLimit(1)
        .padding(.leading, 4)
        .background(ThsChartStyle.canvas)
    }

    @ViewBuilder
    private func volPlot(plotW: CGFloat, volH: CGFloat, maxVol: Double, barW: CGFloat, volMA5: [Double?], volMA10: [Double?]) -> some View {
        ZStack(alignment: .bottomLeading) {
            volMAPath(values: volMA5, maxVol: maxVol, plotW: plotW, plotH: volH, barW: barW, color: ThsChartStyle.volMA5)
            volMAPath(values: volMA10, maxVol: maxVol, plotW: plotW, plotH: volH, barW: barW, color: ThsChartStyle.volMA10)
            HStack(alignment: .bottom, spacing: barSpacing(plotW: plotW, barW: barW)) {
                ForEach(Array(bars.enumerated()), id: \.element.id) { idx, bar in
                    let vh = max(1, volH * CGFloat(bar.volume / maxVol))
                    let color = bar.isUp ? ThsTheme.up : ThsTheme.down
                    Rectangle()
                        .fill(color.opacity(activeIndex == idx ? 1.0 : 0.72))
                        .frame(width: max(1, barW * 0.76), height: vh)
                }
            }
            .padding(.leading, 1)
            .frame(width: plotW, height: volH, alignment: .bottomLeading)
        }
        .frame(width: plotW, height: volH)
    }

    @ViewBuilder
    private func priceAxis(priceH: CGFloat, volBlockH: CGFloat, minP: Double, maxP: Double) -> some View {
        ZStack(alignment: .topTrailing) {
            VStack(spacing: 0) {
                ForEach(0..<5, id: \.self) { i in
                    let price = maxP - (maxP - minP) * Double(i) / 4
                    Text(formatAxisPrice(price))
                        .font(.system(size: 9, weight: .medium, design: .monospaced))
                        .foregroundStyle(ThsTheme.textTertiary)
                        .frame(height: priceH / 4, alignment: .trailing)
                }
                Spacer(minLength: volBlockH)
            }

            if let idx = activeIndex, bars.indices.contains(idx) {
                let bar = bars[idx]
                let y = yPos(bar.close, plotH: priceH, minP: minP, maxP: maxP)
                axisPriceBadge(price: bar.close, isUp: bar.isUp)
                    .offset(y: y - 8)
            }
        }
        .padding(.trailing, 2)
    }

    @ViewBuilder
    private func axisPriceBadge(price: Double, isUp: Bool) -> some View {
        Text(formatAxisPrice(price))
            .font(.system(size: 9, weight: .bold, design: .monospaced))
            .foregroundStyle(.white)
            .padding(.horizontal, 3)
            .padding(.vertical, 2)
            .background(
                RoundedRectangle(cornerRadius: 2)
                    .fill(isUp ? ThsTheme.up : ThsTheme.down)
            )
    }

    @ViewBuilder
    private func latestPriceLine(close: Double, isUp: Bool, plotW: CGFloat, plotH: CGFloat, minP: Double, maxP: Double) -> some View {
        let y = yPos(close, plotH: plotH, minP: minP, maxP: maxP)
        Path { p in
            p.move(to: CGPoint(x: 0, y: y))
            p.addLine(to: CGPoint(x: plotW, y: y))
        }
        .stroke(isUp ? ThsTheme.up.opacity(0.55) : ThsTheme.down.opacity(0.55), style: StrokeStyle(lineWidth: 0.6, dash: [3, 3]))
    }

    @ViewBuilder
    private func crosshair(index: Int, plotW: CGFloat, priceH: CGFloat, volBlockH: CGFloat, minP: Double, maxP: Double, barW: CGFloat) -> some View {
        let bar = bars[index]
        let x = xPos(index: index, plotW: plotW, barW: barW)
        let y = yPos(bar.close, plotH: priceH, minP: minP, maxP: maxP)
        let totalH = priceH + volBlockH

        Path { p in
            p.move(to: CGPoint(x: x, y: 0))
            p.addLine(to: CGPoint(x: x, y: totalH))
        }
        .stroke(ThsChartStyle.crosshair, style: StrokeStyle(lineWidth: 0.6, dash: [3, 2]))

        Path { p in
            p.move(to: CGPoint(x: 0, y: y))
            p.addLine(to: CGPoint(x: plotW, y: y))
        }
        .stroke(ThsChartStyle.crosshair, style: StrokeStyle(lineWidth: 0.6, dash: [3, 2]))
    }

    private func chartGesture(plotW: CGFloat, barW: CGFloat) -> some Gesture {
        DragGesture(minimumDistance: 4)
            .onChanged { value in
                let dx = value.translation.width
                let dy = value.translation.height
                if abs(dx) > abs(dy) * 0.75 || isPanning {
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
        let highlighted = activeIndex == idx
        let bodyW = max(2, barW * 0.72)

        ZStack {
            Path { p in
                let cx = barW / 2
                p.move(to: CGPoint(x: cx, y: wickTop))
                p.addLine(to: CGPoint(x: cx, y: wickBot))
            }
            .stroke(color, lineWidth: highlighted ? 1.2 : 0.8)

            if up {
                Rectangle()
                    .fill(ThsChartStyle.canvas)
                    .frame(width: bodyW, height: bodyH)
                    .overlay(
                        Rectangle()
                            .stroke(color, lineWidth: highlighted ? 1.3 : 1.0)
                    )
                    .position(x: barW / 2, y: (bodyTop + bodyBot) / 2)
            } else {
                Rectangle()
                    .fill(color)
                    .frame(width: bodyW, height: bodyH)
                    .position(x: barW / 2, y: (bodyTop + bodyBot) / 2)
            }
        }
        .frame(width: barW, height: plotH)
    }

    private func maPath(values: [Double?], color: Color, plotW: CGFloat, plotH: CGFloat, minP: Double, maxP: Double, barW: CGFloat) -> some View {
        Path { path in
            guard bars.count > 1 else { return }
            var started = false
            for i in 0..<min(values.count, bars.count) {
                guard let v = values[i] else { continue }
                let x = xPos(index: i, plotW: plotW, barW: barW)
                let y = yPos(v, plotH: plotH, minP: minP, maxP: maxP)
                if started { path.addLine(to: CGPoint(x: x, y: y)) }
                else { path.move(to: CGPoint(x: x, y: y)); started = true }
            }
        }
        .stroke(color, lineWidth: 0.9)
    }

    private func volMAPath(values: [Double?], maxVol: Double, plotW: CGFloat, plotH: CGFloat, barW: CGFloat, color: Color) -> some View {
        Path { path in
            guard bars.count > 1, maxVol > 0 else { return }
            var started = false
            for i in 0..<min(values.count, bars.count) {
                guard let v = values[i] else { continue }
                let x = xPos(index: i, plotW: plotW, barW: barW)
                let y = plotH * (1 - CGFloat(v / maxVol))
                if started { path.addLine(to: CGPoint(x: x, y: y)) }
                else { path.move(to: CGPoint(x: x, y: y)); started = true }
            }
        }
        .stroke(color.opacity(0.9), lineWidth: 0.8)
    }

    private func volumeMA(period: Int) -> [Double?] {
        StockChartService.movingAverage(bars.map(\.volume), period: period)
    }

    // MARK: - 底部日期轴（同花顺标准）

    @ViewBuilder
    private func dateAxis(plotW: CGFloat, barW: CGFloat) -> some View {
        ZStack(alignment: .topLeading) {
            ThsChartStyle.canvas
            ForEach(dateTickIndices, id: \.self) { idx in
                Text(formatAxisDate(bars[idx].date))
                    .font(.system(size: 8, design: .monospaced))
                    .foregroundStyle(ThsChartStyle.axisLabel)
                    .fixedSize()
                    .position(x: clampLabelX(xPos(index: idx, plotW: plotW, barW: barW), plotW: plotW), y: dateAxisH / 2)
            }
            if let idx = activeIndex, bars.indices.contains(idx) {
                Text(formatAxisDate(bars[idx].date))
                    .font(.system(size: 8, weight: .bold, design: .monospaced))
                    .foregroundStyle(.white)
                    .fixedSize()
                    .padding(.horizontal, 3)
                    .padding(.vertical, 1)
                    .background(RoundedRectangle(cornerRadius: 2).fill(ThsChartStyle.crosshairBox))
                    .position(x: clampLabelX(xPos(index: idx, plotW: plotW, barW: barW), plotW: plotW), y: dateAxisH / 2)
            }
        }
    }

    /// 均匀取 4 个刻度（首/三分位/三分之二/尾），数据少时全取。
    private var dateTickIndices: [Int] {
        let n = bars.count
        guard n > 1 else { return n == 1 ? [0] : [] }
        if n <= 3 { return Array(0..<n) }
        let count = 4
        var idxs: [Int] = []
        for i in 0..<count {
            let p = Int((Double(i) * Double(n - 1) / Double(count - 1)).rounded())
            if !idxs.contains(p) { idxs.append(p) }
        }
        return idxs
    }

    private func formatAxisDate(_ d: Date) -> String {
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        switch period {
        case .daily, .weekly: f.dateFormat = "MM-dd"
        case .monthly: f.dateFormat = "yy-MM"
        }
        return f.string(from: d)
    }

    private func clampLabelX(_ x: CGFloat, plotW: CGFloat) -> CGFloat {
        min(max(x, 16), max(16, plotW - 16))
    }

    private func formatVol(_ v: Double) -> String {
        if v >= 1e8 { return String(format: "%.1f亿", v / 1e8) }
        if v >= 1e4 { return String(format: "%.0f万", v / 1e4) }
        return String(format: "%.0f", v)
    }

    private var priceBounds: (Double, Double) {
        guard !bars.isEmpty else { return (0, 1) }
        var lo = bars.map(\.low).min() ?? 0
        var hi = bars.map(\.high).max() ?? 1
        for v in (ma5 + ma10 + ma20 + ma60).compactMap({ $0 }) {
            lo = min(lo, v)
            hi = max(hi, v)
        }
        let pad = max((hi - lo) * 0.04, 0.01)
        return (lo - pad, hi + pad)
    }

    private func barWidth(plotW: CGFloat) -> CGFloat {
        max(2.5, min(11, plotW / CGFloat(max(bars.count, 1)) * 0.72))
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
        CGFloat(index) * (barW + barSpacing(plotW: plotW, barW: barW)) + barW / 2 + 1
    }

    private func index(at x: CGFloat, plotW: CGFloat, barW: CGFloat) -> Int {
        let spacing = barSpacing(plotW: plotW, barW: barW)
        let step = barW + spacing
        guard step > 0 else { return 0 }
        let idx = Int(((x - 1) / step).rounded())
        return min(max(idx, 0), bars.count - 1)
    }

    private func formatAxisPrice(_ v: Double) -> String {
        if v >= 1000 { return String(format: "%.0f", v) }
        if v >= 100 { return String(format: "%.1f", v) }
        if v >= 10 { return String(format: "%.2f", v) }
        return String(format: "%.3f", v)
    }
}

// MARK: - 面板

struct StockChartPanel: View {
    let ticker: String
    @StateObject private var loader = StockChartLoader()
    @State private var refreshTimer: Timer?

    var body: some View {
        VStack(spacing: 0) {
            periodTabs
            maIndicatorRow
            ohlcvStrip
            chartArea
            chartFooter
        }
        .background(ThsChartStyle.canvas)
        .clipShape(RoundedRectangle(cornerRadius: 10))
        .overlay(
            RoundedRectangle(cornerRadius: 10)
                .stroke(ThsTheme.border.opacity(0.5), lineWidth: 0.5)
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

    // 同花顺 Tab：选中项红色 + 底部红线
    private var periodTabs: some View {
        HStack(spacing: 0) {
            ForEach(ChartPeriod.allCases) { p in
                Button {
                    loader.setPeriod(p, ticker: ticker)
                } label: {
                    VStack(spacing: 5) {
                        Text(p.title)
                            .font(.system(size: 13, weight: loader.period == p ? .bold : .medium))
                            .foregroundStyle(loader.period == p ? ThsTheme.accent : ThsTheme.textSecondary)
                        Rectangle()
                            .fill(loader.period == p ? ThsTheme.accent : Color.clear)
                            .frame(height: 2)
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.top, 8)
                    .padding(.bottom, 4)
                }
                .buttonStyle(.plain)
            }
        }
        .background(ThsChartStyle.canvas)
        .zIndex(1)
    }

    private var maIndicatorRow: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 10) {
                maChip("MA5", ThsChartStyle.ma5, maAt(loader.displayMA5))
                maChip("MA10", ThsChartStyle.ma10, maAt(loader.displayMA10))
                maChip("MA20", ThsChartStyle.ma20, maAt(loader.displayMA20))
                maChip("MA60", ThsChartStyle.ma60, maAt(loader.displayMA60))
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
        }
        .background(ThsChartStyle.canvas)
    }

    /// 选中某根 K 线时取该位置 MA 值，否则取最新值（同花顺联动）。
    private func maAt(_ arr: [Double?]) -> Double? {
        if let i = loader.selectedIndex, arr.indices.contains(i) { return arr[i] }
        return arr.compactMap { $0 }.last
    }

    private func maChip(_ label: String, _ color: Color, _ value: Double?) -> some View {
        HStack(spacing: 3) {
            Text(label)
                .font(.system(size: 10, weight: .medium))
                .foregroundStyle(color)
            if let v = value {
                Text(formatPrice(v))
                    .font(.system(size: 10, weight: .semibold, design: .monospaced))
                    .foregroundStyle(color)
            }
        }
    }

    private var ohlcvStrip: some View {
        Group {
            if let bar = loader.selectedBar ?? loader.displayBars.last {
                let prevClose = previousClose(for: bar)
                let change = bar.close - prevClose
                let changePct = prevClose != 0 ? change / prevClose * 100 : 0
                VStack(alignment: .leading, spacing: 4) {
                    HStack(spacing: 0) {
                        Text(formatDate(bar.date))
                            .foregroundStyle(ThsTheme.textSecondary)
                        Spacer(minLength: 8)
                        Text(String(format: "%+.2f  %+.2f%%", change, changePct))
                            .font(.system(size: 11, weight: .bold, design: .monospaced))
                            .foregroundStyle(change >= 0 ? ThsTheme.up : ThsTheme.down)
                    }
                    HStack(spacing: 8) {
                        ohlcvItem("开", bar.open, prevClose)
                        ohlcvItem("高", bar.high, prevClose)
                        ohlcvItem("低", bar.low, prevClose)
                        ohlcvItem("收", bar.close, prevClose)
                        Text("量:\(formatVolume(bar.volume))")
                            .font(.system(size: 10, design: .monospaced))
                            .foregroundStyle(ThsTheme.textSecondary)
                    }
                }
                .font(.system(size: 10, weight: .medium, design: .monospaced))
                .lineLimit(1)
                .minimumScaleFactor(0.75)
                .padding(.horizontal, 10)
                .padding(.vertical, 6)
                .background(ThsChartStyle.canvas)
            }
        }
    }

    @ViewBuilder
    private var chartArea: some View {
        if loader.isLoading && loader.bars.isEmpty {
            ProgressView()
                .tint(ThsTheme.accent)
                .frame(maxWidth: .infinity)
                .frame(height: 300)
                .background(ThsChartStyle.canvas)
        } else if let err = loader.errorMessage, loader.bars.isEmpty {
            VStack(spacing: 8) {
                Image(systemName: "chart.xyaxis.line")
                    .foregroundStyle(ThsTheme.textTertiary)
                Text("K线加载失败")
                    .font(.footnote.weight(.semibold))
                    .foregroundStyle(ThsTheme.textPrimary)
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
            .frame(height: 300)
            .background(ThsChartStyle.canvas)
        } else if !loader.displayBars.isEmpty {
            ThsKLineChartView(
                bars: loader.displayBars,
                ma5: loader.displayMA5,
                ma10: loader.displayMA10,
                ma20: loader.displayMA20,
                ma60: loader.displayMA60,
                period: loader.period,
                selectedIndex: $loader.selectedIndex,
                scrollOffset: $loader.scrollOffset,
                maxScrollOffset: loader.maxScrollOffset
            )
            .frame(height: 300)
        }
    }

    private var chartFooter: some View {
        HStack {
            if loader.maxScrollOffset > 0 {
                Text(scrollHint)
                    .font(.system(size: 9))
                    .foregroundStyle(ThsTheme.textTertiary)
            }
            Spacer()
            if let src = loader.dataSource {
                Text("\(src.rawValue) · \(loader.cloudUpdatedAt ?? "—")")
                    .font(.system(size: 8))
                    .foregroundStyle(ThsTheme.textTertiary)
            }
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 5)
        .background(ThsChartStyle.canvas)
    }

    private var scrollHint: String {
        if loader.scrollOffset == 0 { return "← 左滑看更早K线" }
        if loader.scrollOffset >= loader.maxScrollOffset { return "右滑回最新 →" }
        return "← 左滑更早 · 右滑更新 →"
    }

    private func previousClose(for bar: OHLCVBar) -> Double {
        guard let idx = loader.displayBars.firstIndex(of: bar), idx > 0 else {
            if let last = loader.displayBars.last, bar.id == last.id, loader.displayBars.count >= 2 {
                return loader.displayBars[loader.displayBars.count - 2].close
            }
            return bar.open
        }
        return loader.displayBars[idx - 1].close
    }

    private func ohlcvItem(_ key: String, _ val: Double, _ prevClose: Double) -> some View {
        HStack(spacing: 2) {
            Text(key)
                .foregroundStyle(ThsTheme.textTertiary)
            Text(formatPrice(val))
                .foregroundStyle(val > prevClose ? ThsTheme.up : (val < prevClose ? ThsTheme.down : ThsTheme.textSecondary))
        }
    }

    private func formatDate(_ d: Date) -> String {
        let f = DateFormatter()
        switch loader.period {
        case .daily: f.dateFormat = "yyyy-MM-dd"
        case .weekly: f.dateFormat = "yyyy-MM-dd"
        case .monthly: f.dateFormat = "yyyy-MM"
        }
        return f.string(from: d)
    }

    private func formatPrice(_ v: Double) -> String {
        if v >= 100 { return String(format: "%.2f", v) }
        if v >= 10 { return String(format: "%.3f", v) }
        return String(format: "%.4f", v)
    }

    private func formatVolume(_ v: Double) -> String {
        if v >= 1_000_000_000 { return String(format: "%.2fB", v / 1_000_000_000) }
        if v >= 1_000_000 { return String(format: "%.1fM", v / 1_000_000) }
        if v >= 1_000 { return String(format: "%.0fK", v / 1_000) }
        return String(format: "%.0f", v)
    }
}
