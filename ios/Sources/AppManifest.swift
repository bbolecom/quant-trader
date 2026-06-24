import Foundation

struct AppManifest: Codable {
    let version: String?
    let updated: String?
    let appName: String?
    let tagline: String?
    let categories: [ManifestCategory]
    let quickEntries: [ManifestFeature]
    let features: [ManifestFeature]
    let totalFeatures: Int?

    enum CodingKeys: String, CodingKey {
        case version, updated, tagline, categories, features
        case appName = "app_name"
        case quickEntries = "quick_entries"
        case totalFeatures = "total_features"
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

        if let remote = await loadRemote() {
            manifest = remote
            return
        }
        if let bundled = loadBundled() {
            manifest = bundled
            errorMessage = "使用内置功能清单（未连上 Mac JSON 服务）"
            return
        }
        manifest = Self.fallbackManifest
        errorMessage = "无法加载功能清单"
    }

    private func loadRemote() async -> AppManifest? {
        let paths = ["app_manifest.json"]
        var candidates: [URL] = []
        for path in paths {
            if let url = AppSettings.shared.jsonURL(for: path) {
                candidates.append(url)
            }
        }
        if let gh = AppConfig.githubManifestURL {
            candidates.append(gh)
        }
        for url in candidates {
            do {
                var req = URLRequest(url: url)
                req.cachePolicy = .reloadIgnoringLocalCacheData
                req.timeoutInterval = 12
                let (data, resp) = try await URLSession.shared.data(for: req)
                guard let http = resp as? HTTPURLResponse, (200..<300).contains(http.statusCode) else { continue }
                let decoded = try JSONDecoder().decode(AppManifest.self, from: data)
                loadedFrom = url.host == "raw.githubusercontent.com" ? "GitHub" : url.host
                return decoded
            } catch {
                continue
            }
        }
        return nil
    }

    private func loadBundled() -> AppManifest? {
        guard let url = Bundle.main.url(forResource: "app_manifest", withExtension: "json"),
              let data = try? Data(contentsOf: url) else { return nil }
        return try? JSONDecoder().decode(AppManifest.self, from: data)
    }

    func features(in categoryId: String) -> [ManifestFeature] {
        manifest?.features.filter { $0.thsCategory == categoryId } ?? []
    }

    /// 内置兜底（与 Python app_manifest 同步的核心条目）
    static let fallbackManifest = AppManifest(
        version: "3.0",
        updated: nil,
        appName: "美股量化",
        tagline: "全策略入口",
        categories: [
            ManifestCategory(id: "hub", name: "聚合", icon: "star.circle.fill", color: "#E93030"),
            ManifestCategory(id: "momentum", name: "动量", icon: "flame.fill", color: "#FF6B00"),
            ManifestCategory(id: "flow", name: "量价", icon: "arrow.left.arrow.right", color: "#3B82F6"),
            ManifestCategory(id: "pattern", name: "规律", icon: "sparkles", color: "#A855F7"),
            ManifestCategory(id: "options", name: "期权", icon: "chart.line.uptrend.xyaxis", color: "#10B981"),
            ManifestCategory(id: "composite", name: "综合", icon: "square.grid.3x3.fill", color: "#6366F1"),
            ManifestCategory(id: "screen", name: "筛选", icon: "line.3.horizontal.decrease", color: "#64748B"),
            ManifestCategory(id: "lab", name: "实验室", icon: "flask.fill", color: "#78716C"),
        ],
        quickEntries: [],
        features: [],
        totalFeatures: 0
    )
}
