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
        (highWin?.picks ?? []).filter { !$0.isPlaceholder }
    }

    var highWinWatch: [PickRow] {
        // 高胜率观察池只保留有真实报价的信号：期权类若无真实链（"无真实报价"），
        // 即便历史回测达标也不展示，避免与其它列表不一致地泄漏无效卡片。
        (highWin?.watch ?? []).filter {
            !$0.isPlaceholder && !(($0.isOptionLikePick) && !$0.hasValidTradeData)
        }
    }

    var primaryPicks: [PickRow] {
        if !highWinPicks.isEmpty { return highWinPicks }
        return picks?.filter { $0.isRealOpportunity } ?? []
    }

    var topOpportunities: [PickRow] {
        let combined = highWinPicks + (picks ?? [])
        var seen = Set<String>()
        return combined
            .filter { row in
                let key = "\(row.module)-\(row.ticker)-\(row.status)"
                guard !seen.contains(key) else { return false }
                seen.insert(key)
                return row.isRealOpportunity
            }
            .sorted { lhs, rhs in
                if lhs.opportunityScore != rhs.opportunityScore {
                    return lhs.opportunityScore > rhs.opportunityScore
                }
                return lhs.ticker < rhs.ticker
            }
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
    let rows: Int?
    let error: String?

    enum CodingKeys: String, CodingKey {
        case moduleID = "id"
        case ok
        case count
        case rows
        case error
    }

    var rowCount: Int? { count ?? rows }
}

struct StrategySummaryBlock: Codable {
    let updated: String?
    let coreCount: Int?
    let integratedCount: Int?
    let standaloneCount: Int?
    let integratedWithData: Int?
    let actionableModules: [String]?
    let catalog: [StrategyCatalogRow]?

    enum CodingKeys: String, CodingKey {
        case updated
        case coreCount = "core_count"
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
    let dataDate: String?
    let detail: String

    var dataDateLabel: String {
        let v = dataDate?.trimmingCharacters(in: .whitespaces) ?? ""
        return v.isEmpty ? "—" : v
    }

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
    let account: String?
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
    let strategyRank: Int?
    let strategyTier: String?
    let strategyVerdict: String?
    let strategyScore: Double?
    let strategyWinRate: Double?
    let strategyAnnReturn: Double?
    let pushPriority: Double?
    let explicitOpportunityScore: Double?
    let dataSourceTier: String?
    let dataValid: Bool?
    let tradable: Bool?

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
        case strategyRank = "策略排名"
        case strategyTier = "策略评级"
        case strategyVerdict = "策略审核"
        case strategyScore = "策略分"
        case strategyWinRate = "策略胜率"
        case strategyAnnReturn = "策略年化"
        case pushPriority = "推送优先级"
        case explicitOpportunityScore = "机会评分"
        case dataSourceTier = "数据源"
        case dataValid = "数据有效"
        case tradable = "可交易"
    }

    var isActionable: Bool { status == "可开仓" && hasValidTradeData }

    var isOptionLikePick: Bool {
        let d = direction
        if d.contains("卖Put") || d.contains("卖Call") || d.contains("铁鹰")
            || d.contains("买Put") || d.contains("CSP") { return true }
        let m = module
        return m.contains("CSP") || m.contains("铁鹰") || m.contains("卖Call")
            || m.contains("VRP") || m.contains("期权")
    }

    /// 期权类必须「真实链 + 可交易」；股票信号默认有效。
    var hasValidTradeData: Bool {
        if let t = tradable { return t }
        if let v = dataValid { return v }
        if !isOptionLikePick { return true }
        return dataSourceTier == "真实链"
    }

    var isPlaceholder: Bool {
        let t = ticker.trimmingCharacters(in: .whitespacesAndNewlines)
        let d = direction.trimmingCharacters(in: .whitespacesAndNewlines)
        if status == "扫描失败" || status == "无数据" { return true }
        if t.isEmpty || t == "—" || t == "-" { return true }
        if !isActionable && (d.isEmpty || d == "—" || d == "-") { return true }
        if reason.localizedCaseInsensitiveContains("quick 模式跳过") { return true }
        return false
    }

    var isHighWinQualified: Bool {
        !isPlaceholder && (highWinQualified == true || (histWin ?? 0) >= 0.80)
    }

    var isRealOpportunity: Bool {
        !isPlaceholder && isActionable && hasValidTradeData
    }

    var hasBacktestStats: Bool {
        histWin != nil || histAnn != nil || histDD != nil
    }

    private var rawOpportunityScore: Int {
        if isPlaceholder { return 0 }
        if let explicitOpportunityScore {
            return min(100, max(0, Int(explicitOpportunityScore.rounded())))
        }
        var score = 0.0
        if status == "可开仓" && hasValidTradeData { score += 28 }
        if isHighWinQualified { score += 24 }
        if let histWin {
            score += min(max(histWin, 0), 1) * 22
        } else if let hitRateValue {
            score += min(max(hitRateValue, 0), 1) * 16
        }
        if let histAnn {
            score += min(max(histAnn, 0), 0.8) / 0.8 * 14
        }
        if let histDD {
            let drawdown = abs(histDD)
            score += max(0, 12 - min(drawdown, 0.4) / 0.4 * 12)
        }
        if let strategyRank {
            score += max(0, 16 - Double(strategyRank - 1) * 1.4)
        }
        if let strategyScore {
            score += min(max(strategyScore, 0), 1) * 10
        }
        if strategyVerdict == "主力" || strategyVerdict == "核心" {
            score += 6
        }
        if let pushPriority {
            score += min(max(pushPriority, 0), 150) / 150 * 8
        }
        return min(100, max(0, Int(score.rounded())))
    }

    var opportunityScore: Int {
        if !hasValidTradeData && isOptionLikePick {
            return min(rawOpportunityScore, 35)
        }
        return rawOpportunityScore
    }

    var opportunityGrade: String {
        if !hasValidTradeData && isOptionLikePick {
            return "无真实报价"
        }
        switch opportunityScore {
        case 85...: return "强机会"
        case 70..<85: return "稳健"
        case 55..<70: return "观察"
        default: return isActionable ? "可跟踪" : "等待"
        }
    }

    var riskLevel: String {
        if !hasValidTradeData && isOptionLikePick { return "无真实链" }
        if let histDD, abs(histDD) >= 0.20 { return "高风险" }
        if !isActionable { return "待确认" }
        if isHighWinQualified { return "低风险" }
        return "中风险"
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

    private var hitRateValue: Double? {
        guard let hitRate, !hitRate.isEmpty else { return nil }
        let cleaned = hitRate
            .replacingOccurrences(of: "%", with: "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard let value = Double(cleaned) else { return nil }
        return value > 1.5 ? value / 100 : value
    }

    /// 完整展示策略动作；旧 JSON 若仍是 iron_condor 等代码则转为中文说明。
    var displayAction: String {
        guard let raw = action?.trimmingCharacters(in: .whitespacesAndNewlines), !raw.isEmpty else {
            return ""
        }
        if raw.contains("卖") || raw.contains("买") || raw.contains("铁鹰") || raw.contains("CSP") {
            return raw
        }
        let code = raw.lowercased()
        switch code {
        case "iron_condor":
            return "铁鹰 · 卖Call/买Call + 卖Put/买Put · 四腿收租"
        case "put_credit", "put_spread", "pcs":
            return "Put信用价差 · 卖Put/买Put · 下方收租"
        case "csp":
            return "CSP · 卖Put收租 · 愿接货"
        default:
            return raw
        }
    }

    /// 列表卡片用：策略名 + 结构（不含盈利区间）。
    var displayActionBrief: String {
        let parts = displayAction.split(separator: " · ", omittingEmptySubsequences: false)
        guard !parts.isEmpty else { return "" }
        if parts.count >= 2 {
            return "\(parts[0]) · \(parts[1])"
        }
        return String(parts[0])
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

    private init() {}

    func reload() async {
        isLoading = true
        errorMessage = nil

        let urls = AppConfig.dailyPickCandidateURLs()
        var lastError: String?
        for url in urls {
            switch await fetchRemote(url: url) {
            case .success(let doc):
                let source: DailyPickDataSource = url.host?.contains("192.168.") == true
                    || url.host?.hasPrefix("10.") == true ? .remote : .github
                apply(doc, source: source, from: AppConfig.hostLabel(for: url))
                isLoading = false
                return
            case .failure(let reason):
                lastError = reason
            }
        }

        if let bundled = Self.decodeBundledJSON() {
            apply(bundled, source: .bundled, from: "App 内置")
            errorMessage = lastError.map { "云端不可用（\($0)），已切换到内置快照" }
                ?? "云端不可用，已切换到内置快照"
            isLoading = false
            return
        }

        document = nil
        dataSource = nil
        loadedFrom = nil
        errorMessage = lastError ?? "无法从云端加载选股数据，请检查网络后下拉刷新"
        isLoading = false
    }

    private func apply(_ doc: DailyPickDocument, source: DailyPickDataSource, from host: String?) {
        document = doc
        dataSource = source
        lastUpdated = Date()
        loadedFrom = host
    }

    private enum FetchResult {
        case success(DailyPickDocument)
        case failure(String)
    }

    private func fetchRemote(url: URL) async -> FetchResult {
        do {
            let data = try await CloudFetch.data(from: url)
            if let doc = Self.decodeJSON(data) {
                return .success(doc)
            }
            let reason = Self.decodeFailureReason(from: data) ?? "JSON 格式错误"
            return .failure("\(AppConfig.hostLabel(for: url)): \(reason)")
        } catch {
            return .failure("\(AppConfig.hostLabel(for: url)): \(error.localizedDescription)")
        }
    }

    private static func decodeBundledJSON() -> DailyPickDocument? {
        guard let url = Bundle.main.url(forResource: "daily_pick_today", withExtension: "json"),
              let data = try? Data(contentsOf: url) else { return nil }
        return decodeJSON(data)
    }

    private static func decodeJSON(_ data: Data) -> DailyPickDocument? {
        var payload = data
        if payload.starts(with: [0xEF, 0xBB, 0xBF]) {
            payload = Data(payload.dropFirst(3))
        }
        let decoder = JSONDecoder()
        do {
            return try decoder.decode(DailyPickDocument.self, from: payload)
        } catch {
            #if DEBUG
            print("[DailyPickLoader] decode failed:", error)
            #endif
            return nil
        }
    }

    static func decodeFailureReason(from data: Data) -> String? {
        var payload = data
        if payload.starts(with: [0xEF, 0xBB, 0xBF]) { payload = Data(payload.dropFirst(3)) }
        do {
            _ = try JSONDecoder().decode(DailyPickDocument.self, from: payload)
            return nil
        } catch {
            return error.localizedDescription
        }
    }
}

// MARK: - 全市场快扫（5 分钟轮询）

struct MarketScanDocument: Codable {
    let scanTime: String?
    let scanDate: String?
    let elapsedSec: Double?
    let withinBudget: Bool?
    let scanStats: MarketScanStats?
    let summary: MarketScanSummary?
    let signals: [PickRow]?

    enum CodingKeys: String, CodingKey {
        case scanTime = "扫描时间"
        case scanDate = "扫描日期"
        case elapsedSec = "elapsed_sec"
        case withinBudget = "within_budget"
        case scanStats = "scan_stats"
        case summary
        case signals
    }

    var topSignals: [PickRow] {
        (signals ?? []).filter { !$0.isPlaceholder }.prefix(12).map { $0 }
    }
}

struct MarketScanStats: Codable {
    let universe: Int?
    let signals: Int?
    let phase1Sec: Double?
    let phase2Sec: Double?

    enum CodingKeys: String, CodingKey {
        case universe
        case signals
        case phase1Sec = "phase1_sec"
        case phase2Sec = "phase2_sec"
    }
}

struct MarketScanSummary: Codable {
    let total: Int?
    let gainer10: Int?
    let note: String?

    enum CodingKeys: String, CodingKey {
        case total = "总信号"
        case gainer10 = "Gainer10+"
        case note = "说明"
    }
}

@MainActor
final class MarketScanLoader: ObservableObject {
    static let shared = MarketScanLoader()

    @Published private(set) var document: MarketScanDocument?
    @Published private(set) var isLoading = false
    @Published private(set) var loadedFrom: String?
    @Published private(set) var lastError: String?

    private init() {}

    func reload() async {
        isLoading = true
        defer { isLoading = false }

        if let bundled = loadBundled() {
            document = bundled
        }

        for url in AppConfig.cloudJSONURLs(for: "market_scan_today.json") {
            do {
                let data = try await CloudFetch.data(from: url)
                let decoded = try JSONDecoder().decode(MarketScanDocument.self, from: data)
                document = decoded
                loadedFrom = AppConfig.hostLabel(for: url)
                lastError = nil
                return
            } catch {
                lastError = error.localizedDescription
                continue
            }
        }
    }

    private func loadBundled() -> MarketScanDocument? {
        guard let url = Bundle.main.url(forResource: "market_scan_today", withExtension: "json"),
              let data = try? Data(contentsOf: url) else { return nil }
        return try? JSONDecoder().decode(MarketScanDocument.self, from: data)
    }
}
