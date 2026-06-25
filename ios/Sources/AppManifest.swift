import Foundation

struct AppManifest: Codable {
    let version: String?
    let updated: String?
    let appName: String?
    let tagline: String?
    let coreCount: Int?
    let categories: [ManifestCategory]
    let quickEntries: [ManifestFeature]
    let features: [ManifestFeature]
    let strategyAudit: StrategyAudit?
    let totalFeatures: Int?

    enum CodingKeys: String, CodingKey {
        case version, updated, tagline, categories, features
        case appName = "app_name"
        case coreCount = "core_count"
        case quickEntries = "quick_entries"
        case strategyAudit = "strategy_audit"
        case totalFeatures = "total_features"
    }
}

struct StrategyAudit: Codable, Hashable {
    let updated: String?
    let method: String?
    let strategyCount: Int?
    let rows: [StrategyAuditRow]?

    enum CodingKeys: String, CodingKey {
        case updated, method, rows
        case strategyCount = "strategy_count"
    }
}

struct StrategyAuditRow: Codable, Identifiable, Hashable {
    let id: String
    let name: String?
    let auditRank: Int?
    let auditScore: Double?
    let auditTier: String?
    let auditVerdict: String?
    let auditAction: String?

    enum CodingKeys: String, CodingKey {
        case id, name
        case auditRank = "audit_rank"
        case auditScore = "audit_score"
        case auditTier = "audit_tier"
        case auditVerdict = "audit_verdict"
        case auditAction = "audit_action"
    }
}

struct ManifestCategory: Codable, Identifiable, Hashable {
    let id: String
    let name: String
    let icon: String
    let color: String?
}

struct ManifestFeature: Codable, Identifiable, Hashable {
    let id: String
    let name: String
    let category: String
    let thsCategory: String
    let icon: String
    let script: String?
    let config: String?
    let todayJson: String?
    let todayCsv: String?
    let historyCsv: String?
    let description: String?
    let integratedInDailyPick: Bool?
    let dailyPickModule: String?
    let launcher: String?
    let viewType: String?
    let terminalTab: String?
    let actionable: Int?
    let watching: Int?
    let total: Int?
    let hasData: Bool?
    let dataDate: String?
    let trades: Int?
    let winRate: Double?
    let annReturn: Double?
    let maxDd: Double?
    let sharpe: Double?
    let auditRank: Int?
    let auditScore: Double?
    let auditTier: String?
    let auditVerdict: String?
    let auditAction: String?

    enum CodingKeys: String, CodingKey {
        case id, name, category, icon, script, config, description, launcher
        case thsCategory = "ths_category"
        case todayJson = "today_json"
        case todayCsv = "today_csv"
        case historyCsv = "history_csv"
        case integratedInDailyPick = "integrated_in_daily_pick"
        case dailyPickModule = "daily_pick_module"
        case viewType = "view_type"
        case terminalTab = "terminal_tab"
        case actionable, watching, total
        case hasData = "has_data"
        case dataDate = "data_date"
        case trades
        case winRate = "win_rate"
        case annReturn = "ann_return"
        case maxDd = "max_dd"
        case sharpe
        case auditRank = "audit_rank"
        case auditScore = "audit_score"
        case auditTier = "audit_tier"
        case auditVerdict = "audit_verdict"
        case auditAction = "audit_action"
    }

    var detail: String { description ?? "" }
    var integrated: Bool { integratedInDailyPick ?? false }
    var signalTotal: Int { total ?? 0 }
    var isTerminalOnly: Bool { viewType == "terminal_only" }
    var hasJsonFeed: Bool { !(todayJson ?? "").isEmpty }
}

@MainActor
final class ManifestLoader: ObservableObject {
    static let shared = ManifestLoader()

    @Published var manifest: AppManifest?
    @Published var isLoading = false
    @Published var errorMessage: String?
    @Published var loadedFrom: String?

    func reload() async {
        isLoading = true
        errorMessage = nil
        defer { isLoading = false }

        if let gh = AppConfig.githubManifestURL, let remote = await loadFromURL(gh) {
            manifest = remote
            return
        }

        let paths = ["app_manifest.json"]
        for path in paths {
            if let url = AppSettings.shared.jsonURL(for: path) {
                if let remote = await loadFromURL(url) {
                    manifest = remote
                    return
                }
            }
        }

        manifest = Self.fallbackManifest
        errorMessage = "无法从云端加载功能清单，请检查网络"
    }

    private func loadFromURL(_ url: URL) async -> AppManifest? {
        let busted = AppConfig.cacheBustedURL(url) ?? url
        do {
            var req = URLRequest(url: busted)
            req.cachePolicy = .reloadIgnoringLocalCacheData
            req.timeoutInterval = AppConfig.requestTimeout(for: busted)
            let (data, resp) = try await URLSession.shared.data(for: req)
            guard let http = resp as? HTTPURLResponse, (200..<300).contains(http.statusCode) else { return nil }
            let decoded = try JSONDecoder().decode(AppManifest.self, from: data)
            loadedFrom = url.host == "raw.githubusercontent.com" ? "GitHub" : url.host
            return decoded
        } catch {
            return nil
        }
    }

    func features(in categoryId: String) -> [ManifestFeature] {
        (manifest?.features.filter { $0.thsCategory == categoryId } ?? [])
            .sorted {
                let l = $0.auditRank ?? Int.max
                let r = $1.auditRank ?? Int.max
                if l != r { return l < r }
                return $0.name < $1.name
            }
    }

    /// 内置兜底（与 Python app_manifest 同步的核心条目）
    static let fallbackManifest = AppManifest(
        version: "4.0",
        updated: nil,
        appName: "美股量化",
        tagline: "核心9策略 · 同花顺式原生入口",
        coreCount: 9,
        categories: [
            ManifestCategory(id: "hub", name: "聚合", icon: "star.circle.fill", color: "#E93030"),
            ManifestCategory(id: "momentum", name: "动量", icon: "flame.fill", color: "#FF6B00"),
            ManifestCategory(id: "flow", name: "量价", icon: "arrow.left.arrow.right", color: "#3B82F6"),
            ManifestCategory(id: "pattern", name: "规律", icon: "sparkles", color: "#A855F7"),
            ManifestCategory(id: "options", name: "期权", icon: "chart.line.uptrend.xyaxis", color: "#10B981"),
            ManifestCategory(id: "lab", name: "实验室", icon: "flask.fill", color: "#78716C"),
            ManifestCategory(id: "terminal", name: "终端", icon: "chart.xyaxis.line", color: "#0EA5E9"),
        ],
        quickEntries: [],
        features: [],
        strategyAudit: nil,
        totalFeatures: 0
    )
}
