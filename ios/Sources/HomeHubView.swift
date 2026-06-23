import SwiftUI

/// 同花顺式首页：大盘 + 金刚区 + 分类入口
struct HomeHubView: View {
    @EnvironmentObject private var nav: AppNavigation
    @StateObject private var manifestLoader = ManifestLoader()
    @ObservedObject private var pickLoader = DailyPickLoader.shared

    private let gridColumns = Array(repeating: GridItem(.flexible(), spacing: 12), count: 4)

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    regimeSection
                    metricsSection
                    quickGridSection
                    categorySection
                    moduleRunsSection
                }
                .padding(.horizontal, 16)
                .padding(.bottom, 28)
            }
            .background(ThsTheme.heroGradient.ignoresSafeArea())
            .navigationTitle("首页")
            .navigationBarTitleDisplayMode(.large)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button {
                        Task {
                            await pickLoader.reload()
                            await manifestLoader.reload()
                        }
                    } label: {
                        Image(systemName: "arrow.clockwise")
                    }
                }
            }
            .task {
                await manifestLoader.reload()
                if pickLoader.document == nil {
                    await pickLoader.reload()
                }
            }
            .refreshable {
                await pickLoader.reload()
                await manifestLoader.reload()
            }
        }
        .tint(ThsTheme.accent)
    }

    @ViewBuilder
    private var regimeSection: some View {
        if let doc = pickLoader.document {
            RegimeBanner(
                regime: doc.regime,
                pickDate: doc.pickDate,
                pickTime: doc.pickTime
            )
        } else {
            HStack {
                Image(systemName: "chart.line.uptrend.xyaxis")
                    .foregroundStyle(ThsTheme.accent)
                Text("运行 daily_pick.py 获取大盘开关")
                    .font(.footnote)
                    .foregroundStyle(ThsTheme.textSecondary)
            }
            .padding(14)
            .frame(maxWidth: .infinity, alignment: .leading)
            .thsCard()
        }
    }

    private var metricsSection: some View {
        HStack(spacing: 10) {
            ThsMetricTile(
                title: "可开仓",
                value: "\(pickLoader.document?.summary?.actionable ?? 0)",
                accent: ThsTheme.up,
                icon: "checkmark.seal.fill"
            )
            .onTapGesture { nav.openPicks() }

            ThsMetricTile(
                title: "高胜率",
                value: "\(pickLoader.document?.summary?.highWinActionable ?? pickLoader.document?.highWinPicks.count ?? 0)",
                accent: ThsTheme.accent,
                icon: "star.fill"
            )
            .onTapGesture { nav.openPicks() }

            ThsMetricTile(
                title: "功能",
                value: "\(manifestLoader.manifest?.totalFeatures ?? manifestLoader.manifest?.features.count ?? 0)",
                accent: ThsTheme.textPrimary,
                icon: "square.grid.2x2"
            )

            ThsMetricTile(
                title: "模块",
                value: "\(pickLoader.document?.summary?.moduleCount ?? 0)",
                accent: .orange,
                icon: "puzzlepiece.extension"
            )
        }
    }

    @ViewBuilder
    private var quickGridSection: some View {
        let entries = manifestLoader.manifest?.quickEntries ?? []
        if !entries.isEmpty {
            VStack(alignment: .leading, spacing: 12) {
                ThsSectionHeader(
                    title: "快捷入口",
                    subtitle: "同花顺金刚区 · 一键直达",
                    count: entries.count,
                    accent: ThsTheme.accent,
                    icon: "square.grid.3x3.fill"
                )
                LazyVGrid(columns: gridColumns, spacing: 16) {
                    ForEach(entries) { feat in
                        NavigationLink {
                            destination(for: feat)
                        } label: {
                            ThsQuickIcon(feature: feat)
                        }
                        .buttonStyle(.plain)
                    }
                }
                .padding(.vertical, 8)
                .padding(.horizontal, 4)
                .thsCard()
            }
        }
    }

    private var categorySection: some View {
        VStack(alignment: .leading, spacing: 12) {
            ThsSectionHeader(
                title: "功能分类",
                subtitle: "全系统 \(manifestLoader.manifest?.features.count ?? 0) 项",
                accent: ThsTheme.accent,
                icon: "folder.fill"
            )
            ForEach(manifestLoader.manifest?.categories ?? []) { cat in
                let items = manifestLoader.features(in: cat.id)
                if !items.isEmpty {
                    NavigationLink {
                        FeatureCategoryView(category: cat, features: items)
                    } label: {
                        ThsCategoryRow(
                            category: cat,
                            count: items.count,
                            actionable: items.reduce(0) { $0 + ($1.actionable ?? 0) }
                        )
                    }
                    .buttonStyle(.plain)
                }
            }
        }
    }

    @ViewBuilder
    private var moduleRunsSection: some View {
        if let runs = pickLoader.document?.moduleRuns, !runs.isEmpty {
            VStack(alignment: .leading, spacing: 10) {
                ThsSectionHeader(
                    title: "模块运行状态",
                    subtitle: "daily_pick 执行明细",
                    count: runs.count,
                    accent: ThsTheme.textSecondary,
                    icon: "gearshape.2"
                )
                ForEach(runs) { run in
                    HStack {
                        Image(systemName: run.ok ? "checkmark.circle.fill" : "xmark.circle.fill")
                            .foregroundStyle(run.ok ? ThsTheme.up : ThsTheme.down)
                        Text(run.moduleID)
                            .font(.subheadline)
                        Spacer()
                        if let c = run.count {
                            Text("\(c) 条")
                                .font(.caption)
                                .foregroundStyle(ThsTheme.textTertiary)
                        }
                        if let err = run.error, !err.isEmpty {
                            Text(err)
                                .font(.caption2)
                                .foregroundStyle(ThsTheme.down)
                                .lineLimit(1)
                        }
                    }
                    .padding(12)
                    .thsCard()
                }
            }
        }
    }

    @ViewBuilder
    func destination(for feat: ManifestFeature) -> some View {
        if feat.id == "daily_pick" {
            DailyPickView(embedded: true)
        } else {
            ModuleDetailView(feature: feat)
        }
    }
}

/// 分类下列表
struct FeatureCategoryView: View {
    let category: ManifestCategory
    let features: [ManifestFeature]
    @State private var query = ""

    private var filtered: [ManifestFeature] {
        let q = query.trimmingCharacters(in: .whitespaces)
        guard !q.isEmpty else { return features }
        return features.filter {
            $0.name.localizedCaseInsensitiveContains(q)
                || $0.detail.localizedCaseInsensitiveContains(q)
        }
    }

    var body: some View {
        List(filtered) { feat in
            NavigationLink {
                if feat.id == "daily_pick" {
                    DailyPickView(embedded: true)
                } else {
                    ModuleDetailView(feature: feat)
                }
            } label: {
                ManifestFeatureRow(feature: feat)
            }
            .listRowInsets(EdgeInsets(top: 6, leading: 16, bottom: 6, trailing: 16))
            .listRowSeparator(.hidden)
            .listRowBackground(Color.clear)
        }
        .listStyle(.plain)
        .scrollContentBackground(.hidden)
        .background(ThsTheme.background)
        .navigationTitle(category.name)
        .searchable(text: $query, prompt: "搜索\(category.name)")
        .preferredColorScheme(.dark)
    }
}

/// 功能 Tab：全部分类
struct FeatureHubView: View {
    @StateObject private var loader = ManifestLoader()

    var body: some View {
        NavigationStack {
            Group {
                if loader.isLoading && loader.manifest == nil {
                    ProgressView("加载功能清单…")
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else {
                    List(loader.manifest?.categories ?? []) { cat in
                        let items = loader.features(in: cat.id)
                        if !items.isEmpty {
                            NavigationLink {
                                FeatureCategoryView(category: cat, features: items)
                            } label: {
                                ThsCategoryRow(
                                    category: cat,
                                    count: items.count,
                                    actionable: items.reduce(0) { $0 + ($1.actionable ?? 0) }
                                )
                            }
                            .listRowBackground(ThsTheme.card)
                        }
                    }
                    .scrollContentBackground(.hidden)
                }
            }
            .background(ThsTheme.background)
            .navigationTitle("功能")
            .navigationBarTitleDisplayMode(.large)
            .task { await loader.reload() }
            .refreshable { await loader.reload() }
        }
        .preferredColorScheme(.dark)
    }
}

struct ThsQuickIcon: View {
    let feature: ManifestFeature

    var body: some View {
        VStack(spacing: 6) {
            ZStack(alignment: .topTrailing) {
                Image(systemName: feature.icon)
                    .font(.title2)
                    .foregroundStyle(ThsTheme.accent)
                    .frame(width: 44, height: 44)
                    .background(ThsTheme.accent.opacity(0.12), in: RoundedRectangle(cornerRadius: 12))
                if (feature.actionable ?? 0) > 0 {
                    Circle()
                        .fill(ThsTheme.up)
                        .frame(width: 8, height: 8)
                        .offset(x: 2, y: -2)
                }
            }
            Text(feature.name)
                .font(.system(size: 10))
                .foregroundStyle(ThsTheme.textPrimary)
                .lineLimit(2)
                .multilineTextAlignment(.center)
                .frame(height: 28)
        }
    }
}

struct ThsCategoryRow: View {
    let category: ManifestCategory
    let count: Int
    let actionable: Int

    var body: some View {
        HStack(spacing: 14) {
            Image(systemName: category.icon)
                .font(.title3)
                .foregroundStyle(ThsTheme.accent)
                .frame(width: 36)
            VStack(alignment: .leading, spacing: 4) {
                Text(category.name)
                    .font(.headline)
                    .foregroundStyle(ThsTheme.textPrimary)
                Text("\(count) 项功能")
                    .font(.caption)
                    .foregroundStyle(ThsTheme.textSecondary)
            }
            Spacer()
            if actionable > 0 {
                Text("\(actionable) 信号")
                    .font(.caption2.weight(.bold))
                    .foregroundStyle(ThsTheme.up)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 3)
                    .background(ThsTheme.up.opacity(0.15), in: Capsule())
            }
            Image(systemName: "chevron.right")
                .font(.caption.weight(.bold))
                .foregroundStyle(ThsTheme.textTertiary)
        }
        .padding(14)
        .thsCard()
    }
}

struct ManifestFeatureRow: View {
    let feature: ManifestFeature

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Image(systemName: feature.icon)
                    .foregroundStyle(ThsTheme.accent)
                Text(feature.name)
                    .font(.headline)
                Spacer()
                if feature.integrated {
                    Text("已接入")
                        .font(.caption2.weight(.bold))
                        .foregroundStyle(ThsTheme.up)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(ThsTheme.up.opacity(0.12), in: Capsule())
                }
            }
            if !feature.detail.isEmpty {
                Text(feature.detail)
                    .font(.footnote)
                    .foregroundStyle(ThsTheme.textSecondary)
                    .lineLimit(2)
            }
            HStack(spacing: 12) {
                label("可开", feature.actionable ?? 0, ThsTheme.up)
                label("观望", feature.watching ?? 0, .orange)
                if feature.hasJsonFeed {
                    Text(feature.hasData == true ? "有数据" : "无今日JSON")
                        .font(.caption2)
                        .foregroundStyle(feature.hasData == true ? ThsTheme.accent : ThsTheme.textTertiary)
                } else if feature.isTerminalOnly {
                    Text("量化终端")
                        .font(.caption2)
                        .foregroundStyle(Color.cyan)
                }
            }
        }
        .padding(14)
        .thsCard(border: (feature.actionable ?? 0) > 0 ? ThsTheme.up.opacity(0.25) : ThsTheme.border)
    }

    private func label(_ title: String, _ n: Int, _ color: Color) -> some View {
        HStack(spacing: 3) {
            Text("\(n)").font(.caption.weight(.bold))
            Text(title).font(.caption2)
        }
        .foregroundStyle(n > 0 ? color : ThsTheme.textTertiary)
    }
}

#Preview {
    HomeHubView()
        .environmentObject(AppNavigation())
}
