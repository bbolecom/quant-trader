import SwiftUI

/// 各模块 JSON 详情（暴涨80% / 暴涨扫描 / 投机池 / 通用）
struct ModuleDetailView: View {
    @EnvironmentObject private var nav: AppNavigation
    let feature: ManifestFeature
    @StateObject private var loader = JsonDataLoader()
    @State private var chartSelection: ChartSelection?

    private struct ChartSelection: Identifiable {
        let id = UUID()
        let ticker: String
        let title: String?
        let metadata: [String: Any]
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                headerCard
                if loader.isLoading && loader.root == nil {
                    ProgressView()
                        .frame(maxWidth: .infinity)
                        .padding(.top, 40)
                } else if let root = loader.root {
                    if let note = loader.errorMessage {
                        softNotice(note)
                    }
                    content(for: root)
                } else if let err = loader.errorMessage {
                    errorCard(err)
                } else if feature.isTerminalOnly {
                    terminalCard
                } else {
                    noDataCard
                }
                metaCard
            }
            .padding(16)
        }
        .background(ThsTheme.background.ignoresSafeArea())
        .navigationTitle(feature.name)
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .topBarTrailing) {
                Button {
                    Task { await reload() }
                } label: {
                    Image(systemName: "arrow.clockwise")
                }
            }
        }
        .task { await reload() }
        .refreshable { await reload() }
        .preferredColorScheme(.dark)
        .sheet(item: $chartSelection) { sel in
            TickerDetailView(ticker: sel.ticker, title: sel.title, metadata: sel.metadata)
        }
    }

    private func reload() async {
        guard let path = feature.todayJson, !path.isEmpty else { return }
        await loader.load(path: path)
    }

    private var headerCard: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Image(systemName: feature.icon)
                    .font(.title2)
                    .foregroundStyle(ThsTheme.accent)
                VStack(alignment: .leading) {
                    Text(feature.category)
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(ThsTheme.accent)
                    if !feature.detail.isEmpty {
                        Text(feature.detail)
                            .font(.footnote)
                            .foregroundStyle(ThsTheme.textSecondary)
                    }
                }
            }
            HStack(spacing: 16) {
                statPill("可开仓", feature.actionable ?? 0, ThsTheme.up)
                statPill("观望", feature.watching ?? 0, .orange)
                statPill("合计", feature.total ?? 0, ThsTheme.textSecondary)
            }
        }
        .padding(14)
        .thsCard()
    }

    @ViewBuilder
    private func content(for root: [String: Any]) -> some View {
        switch feature.viewType ?? "json_generic" {
        case "gain15":
            gain15Content(root)
        case "surge":
            surgeContent(root)
        case "speculative_pool":
            speculativePoolContent(root)
        case "pattern":
            patternContent(root)
        case "path5d":
            path5dContent(root)
        case "json_stats":
            jsonStatsContent(root)
        case "meme":
            memeContent(root)
        case "playbook":
            playbookContent(root)
        case "daily_pick":
            EmptyView()
        default:
            genericContent(root)
        }
    }

    // MARK: - Gain15

    private func gain15Content(_ root: [String: Any]) -> some View {
        VStack(alignment: .leading, spacing: 16) {
            marketBanner(root["market"] as? [String: Any])
            jsonSection("今日新暴涨", rows: JsonHelper.array(root, "new_spikes"), emoji: "👁")
            jsonSection("追多确认", rows: JsonHelper.array(root, "buy_confirmed"), emoji: "✅", accent: ThsTheme.up)
            jsonSection("回避确认", rows: JsonHelper.array(root, "avoid_confirmed"), emoji: "⛔", accent: ThsTheme.down)
            jsonSection("观察中", rows: JsonHelper.array(root, "watching"), emoji: "⏳", accent: .orange)
        }
    }

    // MARK: - Surge

    private func surgeContent(_ root: [String: Any]) -> some View {
        VStack(alignment: .leading, spacing: 16) {
            marketBanner(root["market"] as? [String: Any])
            if let stats = root["scan_stats"] as? [String: Any] {
                HStack(spacing: 10) {
                    ThsMetricTile(title: "扫描", value: JsonHelper.string(stats, "universe") ?? "0", accent: ThsTheme.textPrimary, icon: "magnifyingglass")
                    ThsMetricTile(title: "A突破", value: JsonHelper.string(stats, "breakout") ?? "0", accent: ThsTheme.up, icon: "arrow.up.right")
                    ThsMetricTile(title: "B延续", value: JsonHelper.string(stats, "continuation") ?? "0", accent: .orange, icon: "bolt")
                    ThsMetricTile(title: "C前兆", value: JsonHelper.string(stats, "precursor") ?? "0", accent: .cyan, icon: "eye")
                }
            }
            jsonSection("A 突破型", rows: JsonHelper.array(root, "breakout"), emoji: "🚀", accent: ThsTheme.up)
            jsonSection("B 延续/高潮", rows: JsonHelper.array(root, "continuation"), emoji: "⚡", accent: .orange)
            jsonSection("C 前兆蓄势", rows: JsonHelper.array(root, "precursor"), emoji: "👁", accent: .cyan)
        }
    }

    // MARK: - Speculative pool

    private func speculativePoolContent(_ root: [String: Any]) -> some View {
        VStack(alignment: .leading, spacing: 16) {
            jsonSection("C类前兆 · 优先盯盘", rows: JsonHelper.array(root, "today_precursors"), emoji: "👁", accent: .cyan)
            jsonSection("A类突破", rows: JsonHelper.array(root, "today_breakouts"), emoji: "🚀", accent: ThsTheme.up)
            jsonSection("核心池", rows: JsonHelper.array(root, "core"), emoji: "⭐", accent: ThsTheme.accent)
            jsonSection("扩展池", rows: JsonHelper.array(root, "extended"), emoji: "📋")
        }
    }

    // MARK: - Pattern / path5d / stats

    private func patternContent(_ root: [String: Any]) -> some View {
        VStack(alignment: .leading, spacing: 16) {
            if let regime = root["regime"] as? [String: Any] {
                marketBanner(regime)
            }
            jsonSection("腿① 做多", rows: JsonHelper.array(root, "long"), emoji: "✅", accent: ThsTheme.up)
            jsonSection("腿①b 5日路径", rows: JsonHelper.array(root, "path5d"), emoji: "📈", accent: .orange)
            jsonSection("腿② 回避", rows: JsonHelper.array(root, "avoid"), emoji: "⛔", accent: ThsTheme.down)
            if let income = root["income"] as? [String: Any], let lines = income["lines"] as? [String] {
                jsonSection("腿③ 收租", rows: lines.map { ["说明": $0] }, emoji: "💰", accent: ThsTheme.accent)
            }
        }
    }

    private func path5dContent(_ root: [String: Any]) -> some View {
        VStack(alignment: .leading, spacing: 16) {
            jsonSection("5日路径命中", rows: JsonHelper.array(root, "rows", "picks"), emoji: "🛤", accent: .orange)
        }
    }

    private func jsonStatsContent(_ root: [String: Any]) -> some View {
        VStack(alignment: .leading, spacing: 16) {
            jsonKeyValueSummary(root)
            if let rows = root["results"] as? [[String: Any]], !rows.isEmpty {
                jsonSection("回测明细", rows: rows, emoji: "📊")
            } else if let patterns = root["patterns"] as? [String: Any] {
                let rows = patterns.map { key, val -> [String: Any] in
                    var r: [String: Any] = ["规律ID": key]
                    if let d = val as? [String: Any] { r.merge(d) { _, n in n } }
                    return r
                }
                jsonSection("规律统计", rows: rows, emoji: "📈")
            }
        }
    }

    // MARK: - Meme / generic picks

    private func memeContent(_ root: [String: Any]) -> some View {
        VStack(alignment: .leading, spacing: 16) {
            if let regime = root["regime"] as? [String: Any] {
                marketBanner(regime)
            }
            jsonSection("扫描结果", rows: JsonHelper.array(root, "picks"), emoji: "📊")
        }
    }

    private func playbookContent(_ root: [String: Any]) -> some View {
        VStack(alignment: .leading, spacing: 16) {
            jsonSection("舰队计划", rows: JsonHelper.array(root, "plans", "fleet"), emoji: "🚢")
            jsonSection("信号", rows: JsonHelper.array(root, "signals"), emoji: "📡")
            genericContent(root)
        }
    }

    private func genericContent(_ root: [String: Any]) -> some View {
        VStack(alignment: .leading, spacing: 16) {
            if let picks = root["picks"] as? [[String: Any]], !picks.isEmpty {
                jsonSection("信号列表", rows: picks, emoji: "📋")
            }
            let keys = ["buy_confirmed", "avoid_confirmed", "watching", "new_spikes", "members", "core", "signals", "rows", "long", "avoid", "path5d"]
            ForEach(keys, id: \.self) { key in
                let rows = JsonHelper.array(root, key)
                if !rows.isEmpty && key != "picks" {
                    jsonSection(key, rows: rows, emoji: "•")
                }
            }
            if root["picks"] == nil {
                jsonKeyValueSummary(root)
            }
        }
    }

    // MARK: - Components

    private func marketBanner(_ mkt: [String: Any]?) -> some View {
        Group {
            if let mkt {
                let above = mkt["站上MA20"] as? Bool ?? false
                HStack {
                    Text(above ? "🟢 大盘 MA20 上" : "🔴 大盘 MA20 下")
                        .font(.subheadline.weight(.semibold))
                    Spacer()
                    if let spy = JsonHelper.double(mkt, "SPY") {
                        Text("SPY \(JsonHelper.formatNum(spy))")
                            .font(.caption)
                            .foregroundStyle(ThsTheme.textSecondary)
                    }
                }
                .padding(12)
                .thsCard(border: above ? ThsTheme.up.opacity(0.3) : ThsTheme.down.opacity(0.3))
            }
        }
    }

    private func jsonSection(_ title: String, rows: [[String: Any]], emoji: String, accent: Color = ThsTheme.textPrimary) -> some View {
        Group {
            if !rows.isEmpty {
                VStack(alignment: .leading, spacing: 10) {
                    ThsSectionHeader(title: title, count: rows.count, accent: accent, icon: "list.bullet")
                    ForEach(Array(rows.enumerated()), id: \.offset) { _, row in
                        Button {
                            openChart(for: row, section: title)
                        } label: {
                            JsonRecordCard(row: row)
                        }
                        .buttonStyle(.plain)
                    }
                }
            }
        }
    }

    private func jsonKeyValueSummary(_ root: [String: Any]) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            ThsSectionHeader(title: "数据摘要", accent: ThsTheme.textSecondary, icon: "doc.text")
            ForEach(sortedKeys(root).prefix(20), id: \.self) { key in
                if let val = root[key], !(val is [Any] || val is [String: Any]) {
                    HStack {
                        Text(key).font(.caption).foregroundStyle(ThsTheme.textSecondary)
                        Spacer()
                        Text("\(val)").font(.caption).foregroundStyle(ThsTheme.textPrimary)
                    }
                }
            }
            .padding(12)
            .thsCard()
        }
    }

    private func sortedKeys(_ dict: [String: Any]) -> [String] {
        dict.keys.sorted()
    }

    private func openChart(for row: [String: Any], section: String) {
        guard let tk = JsonHelper.ticker(from: row), JsonHelper.isValidTicker(tk) else { return }
        chartSelection = ChartSelection(ticker: tk, title: section, metadata: row)
    }

    private func statPill(_ title: String, _ n: Int, _ color: Color) -> some View {
        VStack(spacing: 2) {
            Text("\(n)").font(.headline.weight(.bold)).foregroundStyle(n > 0 ? color : ThsTheme.textTertiary)
            Text(title).font(.caption2).foregroundStyle(ThsTheme.textSecondary)
        }
    }

    private var terminalCard: some View {
        VStack(spacing: 16) {
            Image(systemName: "chart.xyaxis.line")
                .font(.largeTitle)
                .foregroundStyle(ThsTheme.accent)
            Text("此功能在 Streamlit 量化终端中运行")
                .font(.subheadline)
                .multilineTextAlignment(.center)
            if let tab = feature.terminalTab {
                Text("Tab：\(tab)")
                    .font(.caption)
                    .foregroundStyle(ThsTheme.textSecondary)
            }
            Button {
                nav.openTerminal()
            } label: {
                Label("打开量化终端", systemImage: "arrow.up.forward.app")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .tint(ThsTheme.accent)
        }
        .padding(20)
        .thsCard()
    }

    private func softNotice(_ msg: String) -> some View {
        HStack(spacing: 8) {
            Image(systemName: "icloud.and.arrow.down")
                .foregroundStyle(.orange)
            Text(msg)
                .font(.caption)
                .foregroundStyle(ThsTheme.textSecondary)
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.orange.opacity(0.08), in: RoundedRectangle(cornerRadius: 10))
    }

    private func errorCard(_ msg: String) -> some View {
        VStack(spacing: 12) {
            Image(systemName: "exclamationmark.triangle")
                .foregroundStyle(.orange)
            Text(msg)
                .font(.footnote)
                .multilineTextAlignment(.center)
            if let path = feature.todayJson {
                Text(path)
                    .font(.caption2)
                    .foregroundStyle(ThsTheme.textTertiary)
            }
            Text(AppSettings.shared.jsonURLHint)
                .font(.caption2)
                .foregroundStyle(ThsTheme.textTertiary)
        }
        .padding(16)
        .thsCard()
    }

    private var noDataCard: some View {
        VStack(spacing: 12) {
            if feature.isTerminalOnly {
                terminalCard
            } else {
                Text("暂无 JSON 数据源")
                    .foregroundStyle(ThsTheme.textSecondary)
                if let script = feature.script {
                    Text("Mac 运行：python \(script)")
                        .font(.caption)
                        .foregroundStyle(ThsTheme.textTertiary)
                }
            }
        }
        .padding(16)
    }

    private var metaCard: some View {
        VStack(alignment: .leading, spacing: 6) {
            if let script = feature.script, !script.isEmpty {
                metaRow("脚本", script)
            }
            if let cfg = feature.config, !cfg.isEmpty {
                metaRow("配置", cfg)
            }
            if let path = feature.todayJson, !path.isEmpty {
                metaRow("JSON", path)
            }
            if let date = feature.dataDate, date != "—" {
                metaRow("数据日期", date)
            }
            if let host = loader.loadedFrom {
                metaRow("来源", host)
            }
        }
        .font(.caption2)
        .foregroundStyle(ThsTheme.textTertiary)
        .padding(.top, 8)
    }

    private func metaRow(_ k: String, _ v: String) -> some View {
        HStack(alignment: .top) {
            Text(k + "：").foregroundStyle(ThsTheme.textSecondary)
            Text(v).textSelection(.enabled)
        }
    }
}

struct JsonRecordCard: View {
    let row: [String: Any]

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                if let tk = JsonHelper.string(row, "代码", "ticker") {
                    Text(tk)
                        .font(.headline.weight(.bold))
                        .foregroundStyle(ThsTheme.textPrimary)
                }
                Spacer()
                if let status = JsonHelper.string(row, "状态", "信号", "阶段", "类型名") {
                    Text(status)
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(statusColor(status))
                }
            }
            if let note = JsonHelper.string(row, "说明", "选股理由", "规则") {
                Text(note)
                    .font(.footnote)
                    .foregroundStyle(ThsTheme.textSecondary)
                    .lineLimit(3)
            }
            HStack(spacing: 12) {
                if let gain = JsonHelper.double(row, "涨幅_pct") ?? JsonHelper.double(row, "涨幅%") {
                    chip("涨", JsonHelper.formatPct(gain / (abs(gain) > 1.5 ? 100 : 1)), ThsTheme.up)
                }
                if let vr = JsonHelper.double(row, "量比") {
                    chip("量比", JsonHelper.formatNum(vr), ThsTheme.accent)
                }
                if let score = JsonHelper.double(row, "相似分") {
                    chip("相似", JsonHelper.formatNum(score), .cyan)
                }
                if let px = JsonHelper.double(row, "现价") ?? JsonHelper.double(row, "暴涨收盘价") {
                    chip("价", "$\(JsonHelper.formatNum(px))", ThsTheme.textSecondary)
                }
            }
        }
        .padding(12)
        .thsCard()
    }

    private func chip(_ title: String, _ value: String, _ color: Color) -> some View {
        HStack(spacing: 2) {
            Text(title).font(.caption2)
            Text(value).font(.caption.weight(.bold))
        }
        .foregroundStyle(color)
    }

    private func statusColor(_ s: String) -> Color {
        if s.contains("可开") || s.contains("追多") || s.contains("突破") { return ThsTheme.up }
        if s.contains("回避") || s.contains("出清") { return ThsTheme.down }
        if s.contains("观察") || s.contains("前兆") { return .orange }
        return ThsTheme.textSecondary
    }
}

#Preview {
    NavigationStack {
        ModuleDetailView(feature: ManifestFeature(
            id: "gain15", name: "暴涨80%", category: "动量", thsCategory: "momentum",
            icon: "flame", script: nil, config: nil, todayJson: "research/gain15_daily_today.json",
            todayCsv: nil, historyCsv: nil, description: "test", integratedInDailyPick: true,
            dailyPickModule: nil, launcher: nil, viewType: "gain15", terminalTab: nil,
            actionable: 0, watching: 1, total: 1, hasData: true, dataDate: "2026-06-23"
        ))
    }
    .environmentObject(AppNavigation())
}
