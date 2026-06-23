import SwiftUI

struct StrategyCatalogView: View {
    let rows: [StrategyCatalogRow]
    @State private var query = ""
    @State private var filter: CatalogFilter = .all

    enum CatalogFilter: String, CaseIterable {
        case all = "全部"
        case integrated = "已接入"
        case actionable = "有信号"
    }

    private var filtered: [StrategyCatalogRow] {
        rows.filter { row in
            let matchFilter: Bool
            switch filter {
            case .all: matchFilter = true
            case .integrated: matchFilter = row.integrated
            case .actionable: matchFilter = row.actionable > 0 || row.watching > 0
            }
            let q = query.trimmingCharacters(in: .whitespaces)
            let matchQuery = q.isEmpty
                || row.name.localizedCaseInsensitiveContains(q)
                || row.category.localizedCaseInsensitiveContains(q)
                || row.detail.localizedCaseInsensitiveContains(q)
            return matchFilter && matchQuery
        }
    }

    var body: some View {
        VStack(spacing: 0) {
            filterBar
            List(filtered) { row in
                NavigationLink {
                    if row.strategyID == "daily_pick" {
                        DailyPickView(embedded: true)
                    } else {
                        ModuleDetailView(feature: row.asManifestFeature())
                    }
                } label: {
                    StrategyCatalogCard(row: row)
                }
                .listRowInsets(EdgeInsets(top: 6, leading: 16, bottom: 6, trailing: 16))
                .listRowSeparator(.hidden)
                .listRowBackground(Color.clear)
            }
            .listStyle(.plain)
        }
        .scrollContentBackground(.hidden)
        .background(ThsTheme.background)
        .navigationTitle("策略目录")
        .searchable(text: $query, prompt: "搜索策略")
        .preferredColorScheme(.dark)
    }

    private var filterBar: some View {
        Picker("筛选", selection: $filter) {
            ForEach(CatalogFilter.allCases, id: \.self) { f in
                Text(f.rawValue).tag(f)
            }
        }
        .pickerStyle(.segmented)
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
        .background(ThsTheme.card)
    }
}

struct StrategyCatalogCard: View {
    let row: StrategyCatalogRow

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 4) {
                    Text(row.name)
                        .font(.headline)
                        .foregroundStyle(ThsTheme.textPrimary)
                    Text(row.category)
                        .font(.caption)
                        .foregroundStyle(ThsTheme.textSecondary)
                }
                Spacer()
                VStack(alignment: .trailing, spacing: 6) {
                    if row.integrated {
                        tag("已接入", color: ThsTheme.up)
                    }
                    if row.hasData {
                        tag("有数据", color: ThsTheme.accent)
                    }
                }
            }

            HStack(spacing: 16) {
                signalMetric("可开仓", row.actionable, icon: "checkmark.circle.fill", tint: ThsTheme.up)
                signalMetric("观望", row.watching, icon: "eye", tint: .orange)
                signalMetric("合计", row.total, icon: "list.bullet", tint: ThsTheme.textSecondary)
            }

            if !row.detail.isEmpty {
                Text(row.detail)
                    .font(.footnote)
                    .foregroundStyle(ThsTheme.textSecondary)
                    .lineLimit(3)
            }

            if !row.dataDate.isEmpty {
                Text("数据日期 \(row.dataDate)")
                    .font(.caption2)
                    .foregroundStyle(ThsTheme.textTertiary)
            }
        }
        .padding(14)
        .thsCard(
            border: row.actionable > 0 ? ThsTheme.up.opacity(0.3) : ThsTheme.border,
            radius: 14
        )
    }

    private func tag(_ text: String, color: Color) -> some View {
        Text(text)
            .font(.caption2.weight(.bold))
            .foregroundStyle(color)
            .padding(.horizontal, 8)
            .padding(.vertical, 3)
            .background(color.opacity(0.15), in: Capsule())
    }

    private func signalMetric(_ title: String, _ value: Int, icon: String, tint: Color) -> some View {
        HStack(spacing: 4) {
            Image(systemName: icon)
                .font(.caption2)
            Text("\(value)")
                .font(.caption.weight(.bold))
            Text(title)
                .font(.caption2)
        }
        .foregroundStyle(value > 0 ? tint : ThsTheme.textTertiary)
    }
}

/// 离线功能说明（内置 manifest）
struct StrategyCatalogStaticView: View {
    @StateObject private var loader = ManifestLoader()
    @State private var query = ""

    private var filtered: [ManifestFeature] {
        let all = loader.manifest?.features ?? []
        let q = query.trimmingCharacters(in: .whitespaces)
        guard !q.isEmpty else { return all }
        return all.filter {
            $0.name.localizedCaseInsensitiveContains(q)
                || $0.detail.localizedCaseInsensitiveContains(q)
                || $0.category.localizedCaseInsensitiveContains(q)
        }
    }

    var body: some View {
        Group {
            if filtered.isEmpty && loader.isLoading {
                ProgressView()
            } else if filtered.isEmpty {
                Text("无功能清单，请连接 Mac JSON 服务")
                    .foregroundStyle(ThsTheme.textSecondary)
            } else {
                List(filtered) { feat in
                    NavigationLink {
                        if feat.id == "daily_pick" {
                            DailyPickView(embedded: true)
                        } else {
                            ModuleDetailView(feature: feat)
                        }
                    } label: {
                        HStack(alignment: .top, spacing: 14) {
                            Image(systemName: feat.icon)
                                .font(.title3)
                                .foregroundStyle(ThsTheme.accent)
                                .frame(width: 32)
                            VStack(alignment: .leading, spacing: 4) {
                                Text(feat.name).font(.headline)
                                Text(feat.category)
                                    .font(.caption.weight(.semibold))
                                    .foregroundStyle(ThsTheme.accent.opacity(0.85))
                                if !feat.detail.isEmpty {
                                    Text(feat.detail)
                                        .font(.footnote)
                                        .foregroundStyle(ThsTheme.textSecondary)
                                        .lineLimit(2)
                                }
                            }
                        }
                        .padding(.vertical, 6)
                    }
                    .listRowBackground(ThsTheme.card)
                }
            }
        }
        .scrollContentBackground(.hidden)
        .background(ThsTheme.background)
        .navigationTitle("策略说明")
        .searchable(text: $query, prompt: "搜索策略")
        .task { await loader.reload() }
        .preferredColorScheme(.dark)
    }
}

extension StrategyCatalogRow {
    func asManifestFeature() -> ManifestFeature {
        ManifestFeature(
            id: strategyID,
            name: name,
            category: category,
            thsCategory: thsCategoryId(for: category),
            icon: iconForId(strategyID),
            script: nil,
            config: nil,
            todayJson: jsonPathForId(strategyID),
            todayCsv: nil,
            historyCsv: nil,
            description: detail,
            integratedInDailyPick: integrated,
            dailyPickModule: moduleLabel,
            launcher: nil,
            viewType: viewTypeForId(strategyID),
            terminalTab: terminalTabForId(strategyID),
            actionable: actionable,
            watching: watching,
            total: total,
            hasData: hasData,
            dataDate: dataDate
        )
    }

    private func thsCategoryId(for cat: String) -> String {
        switch cat {
        case "聚合": return "hub"
        case "动量": return "momentum"
        case "量价": return "flow"
        case "规律": return "pattern"
        case "期权收入", "期权": return "options"
        case "综合": return "composite"
        case "筛选", "监控": return "screen"
        default: return "composite"
        }
    }

    private func iconForId(_ id: String) -> String {
        switch id {
        case "gain15": return "flame.fill"
        case "surge_scan": return "bolt.fill"
        case "speculative_pool": return "airplane"
        case "capital_flow": return "arrow.left.arrow.right"
        case "meme_pattern": return "sparkles"
        default: return "square.grid.2x2"
        }
    }

    private func jsonPathForId(_ id: String) -> String? {
        switch id {
        case "gain15": return "research/gain15_daily_today.json"
        case "surge_scan": return "research/surge_scan_today.json"
        case "speculative_pool": return "research/speculative_pool.json"
        case "capital_flow": return "research/flow_daily_today.json"
        case "flow_strategy": return "research/flow_strategy_today.json"
        case "meme_pattern", "ticker_pattern": return "research/ticker_pattern_today.json"
        case "universal_playbook": return "research/universal_playbook_today.json"
        case "daily_pick": return "research/daily_pick_today.json"
        default: return nil
        }
    }

    private func viewTypeForId(_ id: String) -> String {
        switch id {
        case "gain15": return "gain15"
        case "surge_scan": return "surge"
        case "speculative_pool": return "speculative_pool"
        case "meme_pattern", "ticker_pattern": return "meme"
        case "universal_playbook": return "playbook"
        default: return "json_generic"
        }
    }

    private func terminalTabForId(_ id: String) -> String? {
        switch id {
        case "precursor": return "前兆"
        case "backtest_single": return "回测"
        default: return nil
        }
    }
}

#Preview {
    NavigationStack {
        StrategyCatalogStaticView()
    }
}
