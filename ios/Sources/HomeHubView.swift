import SwiftUI

/// 同花顺式首页：红顶栏 · 纵向单列 · 金刚区 · 热点 Feed
struct HomeHubView: View {
    @EnvironmentObject private var nav: AppNavigation
    @EnvironmentObject private var manifestLoader: ManifestLoader
    @EnvironmentObject private var pickLoader: DailyPickLoader
    @State private var searchText = ""
    @State private var hotTab: HomeHotTab = .recommended
    @State private var selectedPick: PickRow?
    @State private var bannerIndex = 0

    private let iconColumns = Array(repeating: GridItem(.flexible(), spacing: 0), count: 5)

    var body: some View {
        NavigationStack {
            ScrollView(.vertical, showsIndicators: false) {
                VStack(spacing: 0) {
                    homeHeader
                    homeBody
                }
            }
            .background(ThsTheme.homeBackground.ignoresSafeArea())
            .toolbar(.hidden, for: .navigationBar)
            .navigationDestination(for: ManifestFeature.self) { feat in
                FeatureDestinationView(feature: feat)
            }
            .navigationDestination(for: ManifestCategory.self) { cat in
                FeatureCategoryView(
                    category: cat,
                    features: manifestLoader.features(in: cat.id)
                )
            }
            .sheet(item: $selectedPick) { row in
                PickDetailView(row: row)
            }
            .task {
                await manifestLoader.reload()
                if pickLoader.document == nil {
                    await pickLoader.reload()
                }
            }
            .refreshable {
                await AppServices.refreshAll()
            }
        }
        .tint(ThsTheme.accent)
        .preferredColorScheme(.light)
    }

    // MARK: - Header

    private var homeHeader: some View {
        VStack(spacing: 10) {
            HStack(spacing: 12) {
                Button { nav.selectedTab = 4 } label: {
                    Image(systemName: "person.circle.fill")
                        .font(.system(size: 28))
                        .foregroundStyle(.white.opacity(0.95))
                }
                .buttonStyle(.plain)

                HStack(spacing: 8) {
                    Image(systemName: "magnifyingglass")
                        .font(.subheadline)
                        .foregroundStyle(ThsTheme.homeTextSecondary)
                    TextField("搜索策略、代码、模块", text: $searchText)
                        .font(.subheadline)
                        .foregroundStyle(ThsTheme.homeTextPrimary)
                        .submitLabel(.search)
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .background(.white, in: Capsule())

                Button { nav.openTerminal() } label: {
                    Image(systemName: "cpu")
                        .font(.body.weight(.medium))
                        .foregroundStyle(.white)
                }
                .buttonStyle(.plain)

                Button {
                    Task { await AppServices.refreshAll() }
                } label: {
                    Image(systemName: "bell")
                        .font(.body.weight(.medium))
                        .foregroundStyle(.white)
                }
                .buttonStyle(.plain)
            }
            .padding(.horizontal, 14)
            .padding(.top, 6)
            .padding(.bottom, 12)
        }
        .background(
            LinearGradient(
                colors: [ThsTheme.homeHeaderRed, ThsTheme.homeHeaderRed.opacity(0.92)],
                startPoint: .top,
                endPoint: .bottom
            )
        )
    }

    // MARK: - Body

    private var homeBody: some View {
        VStack(spacing: 10) {
            iconGridSection
            bannerSection
            watchlistCard
            opportunitySnapshotCard
            aiDiagnosisCard
            hotFeedSection
            moreFeaturesSection
        }
        .padding(.bottom, 24)
    }

    // MARK: - 金刚区 5×2

    @ViewBuilder
    private var iconGridSection: some View {
        let entries = gridEntries
        if !entries.isEmpty {
            LazyVGrid(columns: iconColumns, spacing: 18) {
                ForEach(entries) { item in
                    switch item {
                    case .feature(let feat):
                        NavigationLink(value: feat) {
                            ThsHomeGridIcon(feature: feat, categories: manifestLoader.manifest?.categories ?? [])
                        }
                        .buttonStyle(.plain)
                    case .tab(let title, let icon, let color, let tab):
                        Button {
                            nav.selectedTab = tab
                        } label: {
                            ThsHomeGridAction(title: title, icon: icon, color: color)
                        }
                        .buttonStyle(.plain)
                    case .category(let cat):
                        NavigationLink(value: cat) {
                            ThsHomeGridCategory(category: cat)
                        }
                        .buttonStyle(.plain)
                    }
                }
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 16)
            .background(ThsTheme.homeCard)
        }
    }

    private var gridEntries: [HomeGridEntry] {
        var items: [HomeGridEntry] = []
        let quick = filteredQuickEntries
        for feat in quick.prefix(8) {
            items.append(.feature(feat))
        }
        items.append(.tab(title: "全部", icon: "square.grid.2x2.fill", color: Color(hex: "#6366F1") ?? .indigo, tab: 1))
        items.append(.tab(title: "终端", icon: "chart.xyaxis.line", color: Color(hex: "#0EA5E9") ?? .cyan, tab: 3))
        return items
    }

    private var filteredQuickEntries: [ManifestFeature] {
        let all = manifestLoader.manifest?.quickEntries ?? []
        let q = searchText.trimmingCharacters(in: .whitespaces)
        guard !q.isEmpty else { return all }
        return all.filter {
            $0.name.localizedCaseInsensitiveContains(q)
                || $0.detail.localizedCaseInsensitiveContains(q)
        }
    }

    // MARK: - Banner

    private var bannerSection: some View {
        TabView(selection: $bannerIndex) {
            regimeBanner(index: 0)
            strategyBanner(index: 1)
            terminalBanner(index: 2)
        }
        .tabViewStyle(.page(indexDisplayMode: .automatic))
        .frame(height: 88)
        .padding(.horizontal, 12)
    }

    private func regimeBanner(index: Int) -> some View {
        Button { nav.openPicks() } label: {
            HStack(spacing: 12) {
                Image(systemName: pickLoader.document?.regime?.bull == true ? "chart.line.uptrend.xyaxis" : "chart.line.downtrend.xyaxis")
                    .font(.title2)
                    .foregroundStyle(.white)
                VStack(alignment: .leading, spacing: 4) {
                    Text(pickLoader.document?.regime?.label ?? "运行 daily_pick 获取大盘开关")
                        .font(.subheadline.weight(.semibold))
                        .foregroundStyle(.white)
                    if let spy = pickLoader.document?.regime?.spy, let ma = pickLoader.document?.regime?.ma50 {
                        Text(String(format: "SPY %.2f · MA50 %.2f", spy, ma))
                            .font(.caption)
                            .foregroundStyle(.white.opacity(0.85))
                    }
                }
                Spacer()
                Image(systemName: "chevron.right")
                    .font(.caption.weight(.bold))
                    .foregroundStyle(.white.opacity(0.7))
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 14)
            .background(
                LinearGradient(
                    colors: [Color(hex: "#FF6B00") ?? .orange, ThsTheme.homeHeaderRed],
                    startPoint: .leading,
                    endPoint: .trailing
                ),
                in: RoundedRectangle(cornerRadius: 10, style: .continuous)
            )
        }
        .buttonStyle(.plain)
        .tag(index)
        .padding(.horizontal, 4)
    }

    private func strategyBanner(index: Int) -> some View {
        let actionable = pickLoader.document?.summary?.actionable ?? 0
        let highWin = pickLoader.document?.summary?.highWinActionable ?? 0
        return Button { nav.openPicks() } label: {
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text("今日量化信号")
                        .font(.subheadline.weight(.semibold))
                        .foregroundStyle(.white)
                    Text("可开仓 \(actionable) · 高胜率 \(highWin)")
                        .font(.caption)
                        .foregroundStyle(.white.opacity(0.9))
                }
                Spacer()
                Text("查看")
                    .font(.caption.weight(.bold))
                    .foregroundStyle(ThsTheme.homeHeaderRed)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 6)
                    .background(.white, in: Capsule())
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 14)
            .background(
                LinearGradient(
                    colors: [Color(hex: "#3B82F6") ?? .blue, Color(hex: "#6366F1") ?? .indigo],
                    startPoint: .leading,
                    endPoint: .trailing
                ),
                in: RoundedRectangle(cornerRadius: 10, style: .continuous)
            )
        }
        .buttonStyle(.plain)
        .tag(index)
        .padding(.horizontal, 4)
    }

    private func terminalBanner(index: Int) -> some View {
        Button { nav.openTerminal() } label: {
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text("量化策略终端")
                        .font(.subheadline.weight(.semibold))
                        .foregroundStyle(.white)
                    Text("Streamlit 全功能 · 回测 · 体检")
                        .font(.caption)
                        .foregroundStyle(.white.opacity(0.9))
                }
                Spacer()
                Image(systemName: "arrow.up.right.square")
                    .foregroundStyle(.white)
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 14)
            .background(
                LinearGradient(
                    colors: [Color(hex: "#10B981") ?? .green, Color(hex: "#0EA5E9") ?? .cyan],
                    startPoint: .leading,
                    endPoint: .trailing
                ),
                in: RoundedRectangle(cornerRadius: 10, style: .continuous)
            )
        }
        .buttonStyle(.plain)
        .tag(index)
        .padding(.horizontal, 4)
    }

    // MARK: - 自选概览（全宽纵向）

    private var watchlistCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("自选概览")
                    .font(.headline)
                    .foregroundStyle(ThsTheme.homeTextPrimary)
                Spacer()
                if let date = pickLoader.document?.pickDate {
                    Text(date)
                        .font(.caption)
                        .foregroundStyle(ThsTheme.homeTextSecondary)
                }
            }

            VStack(spacing: 8) {
                watchRow(
                    title: "可开仓机会",
                    value: pickLoader.document?.summary?.actionable ?? 0,
                    tint: ThsTheme.up,
                    icon: "arrow.up.circle.fill"
                ) { nav.openPicks() }

                watchRow(
                    title: "观望观察",
                    value: pickLoader.document?.summary?.watching ?? 0,
                    tint: .orange,
                    icon: "eye.circle.fill"
                ) { nav.openPicks() }

                watchRow(
                    title: "高胜率信号",
                    value: pickLoader.document?.summary?.highWinActionable
                        ?? pickLoader.document?.highWinPicks.count ?? 0,
                    tint: Color(hex: "#6366F1") ?? .indigo,
                    icon: "star.circle.fill"
                ) { nav.openPicks() }

                watchRow(
                    title: "接入模块",
                    value: pickLoader.document?.summary?.moduleCount ?? manifestLoader.manifest?.totalFeatures ?? 0,
                    tint: Color(hex: "#3B82F6") ?? .blue,
                    icon: "puzzlepiece.extension.fill"
                ) { nav.selectedTab = 1 }
            }
        }
        .padding(14)
        .background(ThsTheme.homeCard)
        .padding(.horizontal, 12)
    }

    private func watchRow(title: String, value: Int, tint: Color, icon: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            HStack(spacing: 12) {
                Image(systemName: icon)
                    .font(.title3)
                    .foregroundStyle(tint)
                    .frame(width: 28)
                Text(title)
                    .font(.subheadline)
                    .foregroundStyle(ThsTheme.homeTextPrimary)
                Spacer()
                Text("\(value)")
                    .font(.headline.weight(.bold))
                    .foregroundStyle(tint)
                Image(systemName: "chevron.right")
                    .font(.caption2.weight(.bold))
                    .foregroundStyle(ThsTheme.homeTextSecondary.opacity(0.6))
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 10)
            .background(ThsTheme.homeBackground, in: RoundedRectangle(cornerRadius: 8, style: .continuous))
        }
        .buttonStyle(.plain)
    }

    @ViewBuilder
    private var opportunitySnapshotCard: some View {
        let picks = Array((pickLoader.document?.topOpportunities ?? []).prefix(3))
        if !picks.isEmpty {
            VStack(alignment: .leading, spacing: 12) {
                HStack {
                    Label("今日机会", systemImage: "scope")
                        .font(.headline)
                        .foregroundStyle(ThsTheme.homeTextPrimary)
                    Spacer()
                    Button { nav.openPicks() } label: {
                        Text("全部")
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(ThsTheme.homeHeaderRed)
                    }
                }

                ForEach(Array(picks.enumerated()), id: \.element.id) { idx, row in
                    Button { selectedPick = row } label: {
                        HStack(spacing: 12) {
                            Text("\(idx + 1)")
                                .font(.caption.weight(.black))
                                .foregroundStyle(.white)
                                .frame(width: 24, height: 24)
                                .background(rankTint(for: idx), in: Circle())
                            VStack(alignment: .leading, spacing: 3) {
                                Text("\(row.ticker) · \(row.opportunityGrade)")
                                    .font(.subheadline.weight(.semibold))
                                    .foregroundStyle(ThsTheme.homeTextPrimary)
                                Text(row.reason)
                                    .font(.caption)
                                    .foregroundStyle(ThsTheme.homeTextSecondary)
                                    .lineLimit(1)
                            }
                            Spacer()
                            Text("\(row.opportunityScore)")
                                .font(.headline.weight(.black))
                                .foregroundStyle(scoreTint(for: row))
                                .accessibilityLabel("机会评分 \(row.opportunityScore)")
                            Image(systemName: "chevron.right")
                                .font(.caption2.weight(.bold))
                                .foregroundStyle(ThsTheme.homeTextSecondary.opacity(0.6))
                        }
                        .padding(.horizontal, 12)
                        .padding(.vertical, 10)
                        .background(ThsTheme.homeBackground, in: RoundedRectangle(cornerRadius: 10, style: .continuous))
                    }
                    .buttonStyle(.plain)
                }
            }
            .padding(14)
            .background(ThsTheme.homeCard)
            .padding(.horizontal, 12)
        }
    }

    private func rankTint(for index: Int) -> Color {
        switch index {
        case 0: return ThsTheme.homeHeaderRed
        case 1: return Color(hex: "#F97316") ?? .orange
        default: return Color(hex: "#6366F1") ?? .indigo
        }
    }

    private func scoreTint(for row: PickRow) -> Color {
        row.opportunityScore >= 85 ? ThsTheme.homeHeaderRed : ThsTheme.homeTextPrimary
    }

    // MARK: - AI 诊断（全宽纵向）

    private var aiDiagnosisCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Image(systemName: "sparkles")
                    .foregroundStyle(ThsTheme.homeHeaderRed)
                Text("AI 诊断")
                    .font(.headline)
                    .foregroundStyle(ThsTheme.homeTextPrimary)
                Spacer()
                Button { nav.openTerminal() } label: {
                    Text("详情")
                        .font(.caption)
                        .foregroundStyle(ThsTheme.homeHeaderRed)
                }
            }

            Text(diagnosisText)
                .font(.subheadline)
                .foregroundStyle(ThsTheme.homeTextSecondary)
                .lineSpacing(4)
                .frame(maxWidth: .infinity, alignment: .leading)

            if let playbook = pickLoader.document?.regime?.playbook, !playbook.isEmpty {
                Text(playbook)
                    .font(.caption)
                    .foregroundStyle(ThsTheme.homeHeaderRed)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 6)
                    .background(ThsTheme.homeHeaderRed.opacity(0.08), in: Capsule())
            }
        }
        .padding(14)
        .background(ThsTheme.homeCard)
        .padding(.horizontal, 12)
    }

    private var diagnosisText: String {
        if let doc = pickLoader.document {
            let market = doc.summary?.market ?? doc.regime?.label ?? "—"
            let mode = doc.summary?.mode ?? doc.regime?.mode ?? "—"
            let modules = (doc.summary?.activeModules ?? []).prefix(4).joined(separator: "、")
            let modText = modules.isEmpty ? "暂无活跃模块" : "活跃：\(modules)"
            return "大盘 \(market)，模式 \(mode)。\(modText)。共 \(doc.summary?.total ?? 0) 条信号，其中 \(doc.summary?.actionable ?? 0) 条可开仓。"
        }
        return "暂无诊断数据。请运行 daily_pick.py 或在「我的」中配置 JSON 服务地址。"
    }

    // MARK: - 实时热点 Feed

    private var hotFeedSection: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Text("实时热点")
                    .font(.headline)
                    .foregroundStyle(ThsTheme.homeTextPrimary)
                Spacer()
                Button { nav.openPicks() } label: {
                    HStack(spacing: 2) {
                        Text("更多")
                        Image(systemName: "chevron.right")
                    }
                    .font(.caption)
                    .foregroundStyle(ThsTheme.homeTextSecondary)
                }
            }
            .padding(.horizontal, 14)
            .padding(.top, 14)
            .padding(.bottom, 10)

            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 20) {
                    ForEach(HomeHotTab.allCases) { tab in
                        Button { hotTab = tab } label: {
                            VStack(spacing: 6) {
                                Text(tab.title)
                                    .font(.subheadline.weight(hotTab == tab ? .semibold : .regular))
                                    .foregroundStyle(hotTab == tab ? ThsTheme.homeHeaderRed : ThsTheme.homeTextSecondary)
                                Rectangle()
                                    .fill(hotTab == tab ? ThsTheme.homeHeaderRed : .clear)
                                    .frame(height: 2)
                                    .frame(maxWidth: .infinity)
                            }
                        }
                        .buttonStyle(.plain)
                        .frame(minWidth: 44)
                    }
                }
                .padding(.horizontal, 14)
            }

            Divider().background(ThsTheme.homeDivider)

            LazyVStack(spacing: 0) {
                ForEach(Array(hotFeedItems.enumerated()), id: \.offset) { idx, item in
                    hotFeedRow(item, isFirst: idx == 0)
                    if idx < hotFeedItems.count - 1 {
                        Divider().padding(.leading, 14)
                    }
                }
            }
            .padding(.bottom, 8)

            if hotFeedItems.isEmpty {
                Text("暂无热点，请先刷新选股数据")
                    .font(.footnote)
                    .foregroundStyle(ThsTheme.homeTextSecondary)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 24)
            }
        }
        .background(ThsTheme.homeCard)
        .padding(.horizontal, 12)
    }

    private var hotFeedItems: [HomeFeedItem] {
        guard let doc = pickLoader.document else { return [] }
        switch hotTab {
        case .follow:
            return (doc.highWinWatch + doc.highWinPicks).prefix(8).map { .pick($0) }
        case .recommended:
            return doc.primaryPicks.prefix(10).map { .pick($0) }
        case .hot:
            return doc.sortedModules.prefix(6).map { name, stats in
                .module(name: name, stats: stats)
            }
        case .news:
            return (doc.moduleRuns ?? []).prefix(8).map { .run($0) }
        }
    }

    private func hotFeedRow(_ item: HomeFeedItem, isFirst: Bool) -> some View {
        Group {
            switch item {
            case .pick(let row):
                Button { selectedPick = row } label: {
                    VStack(alignment: .leading, spacing: 8) {
                        if isFirst {
                            Text("头条")
                                .font(.caption2.weight(.bold))
                                .foregroundStyle(.white)
                                .padding(.horizontal, 6)
                                .padding(.vertical, 2)
                                .background(ThsTheme.homeHeaderRed, in: RoundedRectangle(cornerRadius: 3))
                        }
                        HStack(alignment: .top, spacing: 10) {
                            Circle()
                                .fill(ThsTheme.homeHeaderRed.opacity(0.15))
                                .frame(width: 36, height: 36)
                                .overlay {
                                    Text(String(row.ticker.prefix(2)))
                                        .font(.caption.weight(.bold))
                                        .foregroundStyle(ThsTheme.homeHeaderRed)
                                }
                            VStack(alignment: .leading, spacing: 4) {
                                HStack {
                                    Text(row.module)
                                        .font(.caption)
                                        .foregroundStyle(ThsTheme.homeTextSecondary)
                                    Spacer()
                                    StatusBadge(status: row.status)
                                }
                                Text("\(row.ticker) · \(row.direction)")
                                    .font(.subheadline.weight(.semibold))
                                    .foregroundStyle(ThsTheme.homeTextPrimary)
                                Text(row.reason)
                                    .font(.footnote)
                                    .foregroundStyle(ThsTheme.homeTextSecondary)
                                    .lineLimit(2)
                                    .multilineTextAlignment(.leading)
                            }
                        }
                    }
                    .padding(14)
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
                .buttonStyle(.plain)
            case .module(let name, let stats):
                HStack {
                    VStack(alignment: .leading, spacing: 4) {
                        Text(name)
                            .font(.subheadline.weight(.semibold))
                            .foregroundStyle(ThsTheme.homeTextPrimary)
                        Text("可开 \(stats.actionable ?? 0) · 观望 \(stats.watching ?? 0)")
                            .font(.caption)
                            .foregroundStyle(ThsTheme.homeTextSecondary)
                    }
                    Spacer()
                    if let tickers = stats.tickers?.prefix(3), !tickers.isEmpty {
                        Text(tickers.joined(separator: " "))
                            .font(.caption2)
                            .foregroundStyle(ThsTheme.homeHeaderRed)
                    }
                }
                .padding(14)
            case .run(let run):
                HStack(spacing: 10) {
                    Image(systemName: run.ok ? "checkmark.circle.fill" : "xmark.circle.fill")
                        .foregroundStyle(run.ok ? ThsTheme.up : ThsTheme.down)
                    VStack(alignment: .leading, spacing: 2) {
                        Text(run.moduleID)
                            .font(.subheadline)
                            .foregroundStyle(ThsTheme.homeTextPrimary)
                        if let err = run.error, !err.isEmpty {
                            Text(err)
                                .font(.caption2)
                                .foregroundStyle(ThsTheme.down)
                                .lineLimit(1)
                        }
                    }
                    Spacer()
                    if let c = run.count {
                        Text("\(c) 条")
                            .font(.caption)
                            .foregroundStyle(ThsTheme.homeTextSecondary)
                    }
                }
                .padding(14)
            }
        }
    }

    // MARK: - 更多功能（纵向列表）

    private var moreFeaturesSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("功能分类")
                .font(.headline)
                .foregroundStyle(ThsTheme.homeTextPrimary)
                .padding(.horizontal, 14)
                .padding(.top, 4)

            VStack(spacing: 0) {
                ForEach(manifestLoader.manifest?.categories ?? []) { cat in
                    let items = manifestLoader.features(in: cat.id)
                    if !items.isEmpty {
                        NavigationLink(value: cat) {
                            ThsHomeCategoryRow(
                                category: cat,
                                count: items.count,
                                actionable: items.reduce(0) { $0 + ($1.actionable ?? 0) }
                            )
                        }
                        .buttonStyle(.plain)
                        Divider().padding(.leading, 56)
                    }
                }
            }
            .background(ThsTheme.homeCard)
            .clipShape(RoundedRectangle(cornerRadius: 0))
        }
        .padding(.horizontal, 12)
    }
}

// MARK: - Grid Models

private enum HomeGridEntry: Identifiable {
    case feature(ManifestFeature)
    case tab(title: String, icon: String, color: Color, tab: Int)
    case category(ManifestCategory)

    var id: String {
        switch self {
        case .feature(let f): return "f-\(f.id)"
        case .tab(let t, _, _, _): return "t-\(t)"
        case .category(let c): return "c-\(c.id)"
        }
    }
}

private enum HomeHotTab: String, CaseIterable, Identifiable {
    case follow, recommended, hot, news

    var id: String { rawValue }

    var title: String {
        switch self {
        case .follow: return "关注"
        case .recommended: return "推荐"
        case .hot: return "热榜"
        case .news: return "资讯"
        }
    }
}

private enum HomeFeedItem {
    case pick(PickRow)
    case module(name: String, stats: ModuleStats)
    case run(ModuleRun)
}

// MARK: - Grid Icons

struct ThsHomeGridIcon: View {
    let feature: ManifestFeature
    let categories: [ManifestCategory]

    var body: some View {
        VStack(spacing: 6) {
            ZStack(alignment: .topTrailing) {
                RoundedRectangle(cornerRadius: 14, style: .continuous)
                    .fill(iconColor.opacity(0.12))
                    .frame(width: 48, height: 48)
                    .overlay {
                        Image(systemName: feature.icon)
                            .font(.title3)
                            .foregroundStyle(iconColor)
                    }
                if (feature.actionable ?? 0) > 0 {
                    Text("\(feature.actionable ?? 0)")
                        .font(.system(size: 9, weight: .bold))
                        .foregroundStyle(.white)
                        .padding(.horizontal, 4)
                        .padding(.vertical, 1)
                        .background(ThsTheme.homeHeaderRed, in: Capsule())
                        .offset(x: 4, y: -4)
                }
            }
            Text(feature.shortName)
                .font(.system(size: 11))
                .foregroundStyle(ThsTheme.homeTextPrimary)
                .lineLimit(2)
                .multilineTextAlignment(.center)
                .frame(height: 30)
        }
    }

    private var iconColor: Color {
        if let cat = categories.first(where: { $0.id == feature.thsCategory }),
           let hex = cat.color, let c = Color(hex: hex) {
            return c
        }
        return ThsTheme.homeHeaderRed
    }
}

struct ThsHomeGridAction: View {
    let title: String
    let icon: String
    let color: Color

    var body: some View {
        VStack(spacing: 6) {
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .fill(color.opacity(0.12))
                .frame(width: 48, height: 48)
                .overlay {
                    Image(systemName: icon)
                        .font(.title3)
                        .foregroundStyle(color)
                }
            Text(title)
                .font(.system(size: 11))
                .foregroundStyle(ThsTheme.homeTextPrimary)
                .lineLimit(2)
                .multilineTextAlignment(.center)
                .frame(height: 30)
        }
    }
}

struct ThsHomeGridCategory: View {
    let category: ManifestCategory

    var body: some View {
        VStack(spacing: 6) {
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .fill(categoryColor.opacity(0.12))
                .frame(width: 48, height: 48)
                .overlay {
                    Image(systemName: category.icon)
                        .font(.title3)
                        .foregroundStyle(categoryColor)
                }
            Text(category.name)
                .font(.system(size: 11))
                .foregroundStyle(ThsTheme.homeTextPrimary)
                .lineLimit(2)
                .multilineTextAlignment(.center)
                .frame(height: 30)
        }
    }

    private var categoryColor: Color {
        if let hex = category.color, let c = Color(hex: hex) { return c }
        return ThsTheme.homeHeaderRed
    }
}

struct ThsHomeCategoryRow: View {
    let category: ManifestCategory
    let count: Int
    let actionable: Int

    var body: some View {
        HStack(spacing: 14) {
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .fill(rowColor.opacity(0.12))
                .frame(width: 36, height: 36)
                .overlay {
                    Image(systemName: category.icon)
                        .font(.body)
                        .foregroundStyle(rowColor)
                }
            VStack(alignment: .leading, spacing: 2) {
                Text(category.name)
                    .font(.subheadline.weight(.medium))
                    .foregroundStyle(ThsTheme.homeTextPrimary)
                Text("\(count) 项功能")
                    .font(.caption)
                    .foregroundStyle(ThsTheme.homeTextSecondary)
            }
            Spacer()
            if actionable > 0 {
                Text("\(actionable) 信号")
                    .font(.caption2.weight(.bold))
                    .foregroundStyle(ThsTheme.homeHeaderRed)
            }
            Image(systemName: "chevron.right")
                .font(.caption2.weight(.bold))
                .foregroundStyle(ThsTheme.homeTextSecondary.opacity(0.5))
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 12)
    }

    private var rowColor: Color {
        if let hex = category.color, let c = Color(hex: hex) { return c }
        return ThsTheme.homeHeaderRed
    }
}

extension ManifestFeature {
    var shortName: String {
        if let sep = name.firstIndex(of: "·") {
            let left = name[..<sep].trimmingCharacters(in: .whitespaces)
            if left.count <= 6 { return left }
        }
        if name.count <= 5 { return name }
        return String(name.prefix(4))
    }
}

#Preview {
    HomeHubView()
        .environmentObject(AppNavigation())
        .environmentObject(ManifestLoader.shared)
        .environmentObject(DailyPickLoader.shared)
}
