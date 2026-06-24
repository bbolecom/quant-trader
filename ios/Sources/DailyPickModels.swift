import Foundation

struct DailyPickDocument: Codable {
    let pickDate: String?
    let pickTime: String?
    let philosophy: String?
    let frequencyProfile: String?
    let regime: RegimeInfo?
    let summary: PickSummary?
    let modulesSummary: [String: ModuleStats]?
    let strategySummary: StrategySummaryBlock?
    let highWin: HighWinBlock?
    let moduleRuns: [ModuleRun]?
    let picks: [PickRow]?

    enum CodingKeys: String, CodingKey {
        case pickDate = "选股日期"
        case pickTime = "选股时间"
        case philosophy
        case frequencyProfile = "frequency_profile"
        case regime
        case summary
        case modulesSummary = "modules_summary"
        case strategySummary = "strategy_summary"
        case highWin = "high_win"
        case moduleRuns = "module_runs"
        case picks
    }

    var highWinPicks: [PickRow] {
        highWin?.picks ?? []
    }

    var highWinWatch: [PickRow] {
        highWin?.watch ?? []
    }

    var primaryPicks: [PickRow] {
        if !highWinPicks.isEmpty { return highWinPicks }
        return picks?.filter(\.isActionable) ?? []
    }

    var sortedModules: [(String, ModuleStats)] {
        guard let mods = modulesSummary else { return [] }
        return mods.sorted { lhs, rhs in
            let la = lhs.value.actionable ?? 0
            let ra = rhs.value.actionable ?? 0
            if la != ra { return la > ra }
            return (lhs.value.total ?? 0) > (rhs.value.total ?? 0)
        }
    }
}

struct RegimeInfo: Codable {
    let bull: Bool?
    let label: String?
    let spy: Double?
    let ma50: Double?
    let mode: String?
    let playbook: String?
}

struct PickSummary: Codable {
    let total: Int?
    let actionable: Int?
    let watching: Int?
    let emptyDay: Bool?
    let market: String?
    let mode: String?
    let moduleCount: Int?
    let activeModules: [String]?
    let highWinActionable: Int?
    let runModules: Int?

    enum CodingKeys: String, CodingKey {
        case total = "总条目"
        case actionable = "可开仓"
        case watching = "观望"
        case emptyDay = "是否空仓日"
        case market = "大盘"
        case mode = "模式"
        case moduleCount = "接入模块数"
        case activeModules = "有信号模块"
        case highWinActionable = "高胜率可开仓"
        case runModules = "运行模块数"
    }
}

struct HighWinBlock: Codable {
    let minWinRate: Double?
    let summary: HighWinSummary?
    let picks: [PickRow]?
    let watch: [PickRow]?

    enum CodingKeys: String, CodingKey {
        case minWinRate = "min_win_rate"
        case summary
        case picks
        case watch
    }
}

struct HighWinSummary: Codable {
    let actionable: Int?
    let watchCount: Int?
    let modules: [String]?

    enum CodingKeys: String, CodingKey {
        case actionable = "可开仓高胜率"
        case watchCount = "观察高胜率"
        case modules = "模块"
    }
}

struct ModuleStats: Codable {
    let total: Int?
    let actionable: Int?
    let watching: Int?
    let tickers: [String]?

    enum CodingKeys: String, CodingKey {
        case total = "总条目"
        case actionable = "可开仓"
        case watching = "观望"
        case tickers = "代码"
    }
}

struct ModuleRun: Codable, Identifiable {
    var id: String { moduleID }
    let moduleID: String
    let ok: Bool
    let count: Int?
    let error: String?

    enum CodingKeys: String, CodingKey {
        case moduleID = "id"
        case ok
        case count
        case error
    }
}

struct StrategySummaryBlock: Codable {
    let updated: String?
    let integratedCount: Int?
    let standaloneCount: Int?
    let integratedWithData: Int?
    let actionableModules: [String]?
    let catalog: [StrategyCatalogRow]?

    enum CodingKeys: String, CodingKey {
        case updated
        case integratedCount = "integrated_count"
        case standaloneCount = "standalone_count"
        case integratedWithData = "integrated_with_data"
        case actionableModules = "actionable_modules"
        case catalog
    }
}

struct StrategyCatalogRow: Codable, Identifiable {
    var id: String { strategyID }
    let strategyID: String
    let name: String
    let category: String
    let integrated: Bool
    let moduleLabel: String
    let hasData: Bool
    let actionable: Int
    let watching: Int
    let total: Int
    let dataDate: String
    let detail: String

    enum CodingKeys: String, CodingKey {
        case strategyID = "id"
        case name = "策略"
        case category = "分类"
        case integrated = "已接入每日选股"
        case moduleLabel = "模块标签"
        case hasData = "今日有数据"
        case actionable = "可开仓"
        case watching = "观望"
        case total = "总条目"
        case dataDate = "数据日期"
        case detail = "说明"
    }
}

struct PickRow: Codable, Identifiable {
    var id: String { "\(module)-\(ticker)-\(status)-\(reason.hashValue)" }
    let module: String
    let account: String
    let ticker: String
    let status: String
    let direction: String
    let action: String?
    let hitRate: String?
    let reason: String
    let histWin: Double?
    let histAnn: Double?
    let histDD: Double?
    let backtestNote: String?
    let backtestSource: String?
    let highWinQualified: Bool?

    enum CodingKeys: String, CodingKey {
        case module = "模块"
        case account = "账户"
        case ticker = "代码"
        case status = "状态"
        case direction = "方向"
        case action = "策略动作"
        case hitRate = "历史命中率"
        case reason = "选股理由"
        case histWin = "历史胜率"
        case histAnn = "历史年化"
        case histDD = "最大回撤"
        case backtestNote = "回测摘要"
        case backtestSource = "回测来源"
        case highWinQualified = "高胜率达标"
    }

    var isActionable: Bool { status == "可开仓" }

    var isHighWinQualified: Bool {
        highWinQualified == true || (histWin ?? 0) >= 0.80
    }

    var hasBacktestStats: Bool {
        histWin != nil || histAnn != nil || histDD != nil
    }

    var displayWinRate: String {
        if let w = histWin { return Self.formatPercent(w) }
        if let h = hitRate, !h.isEmpty { return h.contains("%") ? h : "\(h)%" }
        return "—"
    }

    var displayAnnReturn: String {
        guard let v = histAnn else { return "—" }
        return Self.formatPercent(v)
    }

    var displayMaxDD: String {
        guard let v = histDD else { return "—" }
        return Self.formatPercent(v)
    }

    static func formatPercent(_ value: Double) -> String {
        let pct = abs(value) <= 1.5 ? value * 100 : value
        return String(format: "%+.1f%%", pct)
    }
}

enum DailyPickLoaderError: LocalizedError {
    case noURL
    case badResponse
    case decodeFailed
    case noDataAvailable

    var errorDescription: String? {
        switch self {
        case .noURL: return "未配置选股 JSON 地址"
        case .badResponse: return "无法连接 Mac JSON 服务"
        case .decodeFailed: return "选股数据格式错误"
        case .noDataAvailable: return "暂无可用选股数据"
        }
    }
}

enum DailyPickDataSource: String {
    case remote = "实时"
    case github = "GitHub"
    case bundled = "内置快照"

    var label: String {
        switch self {
        case .remote: return "Mac 实时"
        case .github: return "GitHub 云端"
        case .bundled: return "内置快照"
        }
    }
}

@MainActor
final class DailyPickLoader: ObservableObject {
    static let shared = DailyPickLoader()

    @Published var document: DailyPickDocument?
    @Published var isLoading = false
    @Published var errorMessage: String?
    @Published var lastUpdated: Date?
    @Published var loadedFrom: String?
    @Published var dataSource: DailyPickDataSource?

    func reload() async {
        isLoading = true
        errorMessage = nil
        defer { isLoading = false }

        let urls = AppConfig.dailyPickCandidateURLs()
        var lastError: String?

        for url in urls {
            if let doc = await fetchRemote(url: url) {
                apply(doc, source: url.host == "raw.githubusercontent.com" ? .github : .remote, from: url.host)
                return
            }
            lastError = errorMessage
        }

        if let bundled = loadBundled() {
            apply(bundled, source: .bundled, from: "内置")
            errorMessage = lastError == nil ? nil : "Mac 未连接 · 已加载\(dataSource?.label ?? "内置")数据"
            return
        }

        errorMessage = lastError ?? DailyPickLoaderError.noDataAvailable.errorDescription
        document = nil
        dataSource = nil
        loadedFrom = nil
    }

    private func apply(_ doc: DailyPickDocument, source: DailyPickDataSource, from host: String?) {
        document = doc
        dataSource = source
        lastUpdated = Date()
        loadedFrom = host
    }

    private func fetchRemote(url: URL) async -> DailyPickDocument? {
        do {
            var request = URLRequest(url: url)
            request.cachePolicy = .reloadIgnoringLocalCacheData
            request.timeoutInterval = 12
            let (data, response) = try await URLSession.shared.data(for: request)
            guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
                return nil
            }
            return try JSONDecoder().decode(DailyPickDocument.self, from: data)
        } catch is DecodingError {
            errorMessage = DailyPickLoaderError.decodeFailed.errorDescription
            return nil
        } catch {
            return nil
        }
    }

    private func loadBundled() -> DailyPickDocument? {
        guard let url = Bundle.main.url(forResource: "daily_pick_today", withExtension: "json"),
              let data = try? Data(contentsOf: url) else { return nil }
        return try? JSONDecoder().decode(DailyPickDocument.self, from: data)
    }
}
