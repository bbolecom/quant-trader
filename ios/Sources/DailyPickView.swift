import SwiftUI

struct DailyPickView: View {
    var embedded: Bool = false
    @StateObject private var loader = DailyPickLoader.shared
    @State private var selectedPick: PickRow?
    @State private var showSettings = false

    var body: some View {
        Group {
            if embedded {
                pickBody
            } else {
                NavigationStack {
                    pickBody
                }
            }
        }
        .tint(ThsTheme.accent)
        .preferredColorScheme(.dark)
    }

    private var pickBody: some View {
        Group {
            if loader.isLoading && loader.document == nil {
                DailyPickSkeleton()
            } else if let doc = loader.document {
                pickContent(doc)
            } else {
                emptyState
            }
        }
        .background(ThsTheme.heroGradient.ignoresSafeArea())
        .navigationTitle("今日选股")
        .navigationBarTitleDisplayMode(embedded ? .inline : .large)
        .toolbar {
            if !embedded {
                ToolbarItem(placement: .topBarLeading) {
                    if let updated = loader.lastUpdated {
                        Text(updated, style: .time)
                            .font(.caption2)
                            .foregroundStyle(ThsTheme.textTertiary)
                    }
                }
                ToolbarItemGroup(placement: .topBarTrailing) {
                    Button { showSettings = true } label: {
                        Image(systemName: "gearshape")
                    }
                    Button {
                        Task { await loader.reload() }
                    } label: {
                        Image(systemName: "arrow.clockwise")
                    }
                    .disabled(loader.isLoading)
                }
            }
        }
        .sheet(item: $selectedPick) { row in
            PickDetailView(row: row)
        }
        .sheet(isPresented: $showSettings) {
            SettingsView(loader: loader)
        }
        .task { await loader.reload() }
        .refreshable { await loader.reload() }
    }

    private var emptyState: some View {
        VStack(spacing: 20) {
            Image(systemName: "antenna.radiowaves.left.and.right.slash")
                .font(.system(size: 48))
                .foregroundStyle(ThsTheme.accent.opacity(0.8))
            Text("暂无选股数据")
                .font(.title3.weight(.semibold))
                .foregroundStyle(ThsTheme.textPrimary)
            Text(loader.errorMessage ?? "请先在 Mac 运行 daily_pick.py，并配置 JSON 地址。")
                .font(.footnote)
                .multilineTextAlignment(.center)
                .foregroundStyle(ThsTheme.textSecondary)
                .padding(.horizontal, 24)
            Button {
                Task { await loader.reload() }
            } label: {
                Label("重新加载", systemImage: "arrow.clockwise")
                    .font(.subheadline.weight(.semibold))
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .tint(ThsTheme.accent)
            .padding(.horizontal, 40)
            Text(AppConfig.dailyPickURLHint)
                .font(.caption2)
                .foregroundStyle(ThsTheme.textTertiary)
                .multilineTextAlignment(.center)
                .padding(.horizontal)
        }
        .padding()
    }

    @ViewBuilder
    private func pickContent(_ doc: DailyPickDocument) -> some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                RegimeBanner(regime: doc.regime, pickDate: doc.pickDate, pickTime: doc.pickTime)

                metricsRow(doc)

                if let philosophy = doc.philosophy, !philosophy.isEmpty {
                    Text(philosophy)
                        .font(.caption)
                        .foregroundStyle(ThsTheme.textSecondary)
                        .padding(.horizontal, 4)
                }

                highWinSection(doc)

                if !doc.highWinWatch.isEmpty {
                    watchHighWinSection(doc)
                }

                if let mods = doc.modulesSummary, !mods.isEmpty {
                    modulesSection(doc)
                }

                strategyLinkSection(doc)

                if let watching = doc.picks?.filter({ !$0.isActionable && !$0.isHighWinQualified }), !watching.isEmpty {
                    ThsSectionHeader(
                        title: "观望 / 观察",
                        subtitle: "待确认或条件未满足",
                        count: watching.count,
                        accent: .orange,
                        icon: "eye"
                    )
                    ForEach(watching.prefix(25)) { row in
                        PickCardView(row: row, highlight: false) {
                            selectedPick = row
                        }
                    }
                }
            }
            .padding(.horizontal, 16)
            .padding(.bottom, 24)
        }
    }

    private func metricsRow(_ doc: DailyPickDocument) -> some View {
        HStack(spacing: 10) {
            ThsMetricTile(
                title: "高胜率",
                value: "\(doc.summary?.highWinActionable ?? doc.highWinPicks.count)",
                accent: ThsTheme.up,
                icon: "star.fill"
            )
            ThsMetricTile(
                title: "可开仓",
                value: "\(doc.summary?.actionable ?? 0)",
                accent: ThsTheme.textPrimary,
                icon: "checkmark.seal"
            )
            ThsMetricTile(
                title: "观望",
                value: "\(doc.summary?.watching ?? 0)",
                accent: .orange,
                icon: "hourglass"
            )
        }
    }

    @ViewBuilder
    private func highWinSection(_ doc: DailyPickDocument) -> some View {
        let picks = doc.highWinPicks
        let minWR = doc.highWin?.minWinRate ?? 0.80
        ThsSectionHeader(
            title: "高胜率可执行",
            subtitle: String(format: "历史胜率 ≥ %.0f%% · 今日可开仓", minWR * 100),
            count: picks.count,
            accent: ThsTheme.up,
            icon: "star.circle.fill"
        )
        if picks.isEmpty {
            emptyHighWinCard(doc)
        } else {
            ForEach(picks) { row in
                PickCardView(row: row, highlight: true) {
                    selectedPick = row
                }
            }
        }
    }

    private func emptyHighWinCard(_ doc: DailyPickDocument) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            if doc.summary?.emptyDay == true {
                Label("今日空仓", systemImage: "moon.zzz")
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(.orange)
                Text("无符合高胜率规则的标的，正常等待下一信号。")
            } else {
                Label("暂无高胜率可开仓", systemImage: "line.3.horizontal.decrease.circle")
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(ThsTheme.textSecondary)
                Text("部分模块有观望信号，可在下方查看；或运行完整 daily_pick 刷新。")
            }
        }
        .font(.footnote)
        .foregroundStyle(ThsTheme.textSecondary)
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .thsCard(border: ThsTheme.border.opacity(0.5))
    }

    private func watchHighWinSection(_ doc: DailyPickDocument) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            ThsSectionHeader(
                title: "高胜率观察池",
                subtitle: "规则达标 · 待 T+1/T+3 确认",
                count: doc.highWinWatch.count,
                accent: .yellow,
                icon: "binoculars"
            )
            ForEach(doc.highWinWatch.prefix(15)) { row in
                PickCardView(row: row, highlight: false) {
                    selectedPick = row
                }
            }
        }
    }

    private func modulesSection(_ doc: DailyPickDocument) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            ThsSectionHeader(
                title: "模块信号",
                subtitle: "今日各策略产出概览",
                count: doc.summary?.moduleCount,
                accent: ThsTheme.accent,
                icon: "square.grid.3x3"
            )
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 10) {
            ForEach(doc.sortedModules, id: \.0) { name, stats in
                ModuleChip(
                    name: name,
                    actionable: stats.actionable ?? 0,
                    watching: stats.watching ?? 0
                )
            }
                }
                .padding(.vertical, 2)
            }
        }
    }

    @ViewBuilder
    private func strategyLinkSection(_ doc: DailyPickDocument) -> some View {
        if let catalog = doc.strategySummary?.catalog, !catalog.isEmpty {
            VStack(alignment: .leading, spacing: 10) {
                ThsSectionHeader(
                    title: "全系统策略",
                    subtitle: "接入 \(doc.strategySummary?.integratedCount ?? catalog.count) 个 · 今日有数据 \(doc.strategySummary?.integratedWithData ?? 0)",
                    count: catalog.count,
                    accent: ThsTheme.accent,
                    icon: "books.vertical"
                )
                NavigationLink {
                    StrategyCatalogView(rows: catalog)
                } label: {
                    HStack {
                        VStack(alignment: .leading, spacing: 4) {
                            Text("查看策略目录")
                                .font(.subheadline.weight(.semibold))
                            Text("分类 · 今日信号 · 接入状态")
                                .font(.caption)
                                .foregroundStyle(ThsTheme.textSecondary)
                        }
                        Spacer()
                        Image(systemName: "chevron.right")
                            .font(.caption.weight(.bold))
                    }
                    .foregroundStyle(ThsTheme.textPrimary)
                    .padding(16)
                    .thsCard(border: ThsTheme.accent.opacity(0.25))
                }
            }
        }
    }
}

#Preview {
    DailyPickView()
}
