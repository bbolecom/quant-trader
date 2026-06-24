import Foundation

/// 编译时默认；运行时可在「我的」修改（AppSettings）
enum AppConfig {
    static let defaultServerURLString = "https://quant-trader-fd3mch56aixtttm5rgyc6i.streamlit.app"
    /// Render 一键部署 `render.yaml` 后的默认 K 线实时 API（可在「我的」覆盖）
    static let defaultChartAPIBase = "https://quant-trader-chart-api.onrender.com"
    static let githubRepo = "bbolecom/quant-trader"
    static let githubBranch = "main"

    static var githubDailyPickURL: URL? {
        cacheBustedURL(
            URL(string: "https://raw.githubusercontent.com/\(githubRepo)/\(githubBranch)/research/daily_pick_today.json")
        )
    }

    static var githubManifestURL: URL? {
        cacheBustedURL(
            URL(string: "https://raw.githubusercontent.com/\(githubRepo)/\(githubBranch)/research/app_manifest.json")
        )
    }

    /// 兼容旧代码
    static var serverURLString: String {
        AppSettings.shared.streamlitURL
    }

    static var fallbackServerURL: URL {
        URL(string: defaultServerURLString) ?? URL(string: "https://streamlit.io")!
    }

    static var serverURL: URL {
        AppSettings.shared.serverURL
    }

    static var dailyPickJSONURLString: String {
        AppSettings.shared.jsonBaseURL
    }

    static var dailyPickURLHint: String {
        AppSettings.shared.jsonURLHint
    }

    static func dailyPickCandidateURLs() -> [URL] {
        jsonCandidateURLs(for: "daily_pick_today.json")
    }

    /// 模块 JSON 相对路径，如 `gain15_daily_today.json`
    static func jsonRelativePath(_ path: String) -> String {
        path
            .replacingOccurrences(of: "research/", with: "")
            .trimmingCharacters(in: CharacterSet(charactersIn: "/"))
    }

    /// 模块 JSON 候选 URL：GitHub 云端（实时）→ Mac 局域网 JSON 服务
    static func jsonCandidateURLs(for path: String) -> [URL] {
        var urls: [URL] = []
        let rel = jsonRelativePath(path)

        if let gh = githubJSONURL(for: rel) {
            urls.append(gh)
        }
        if let u = AppSettings.shared.jsonURL(for: path) {
            let busted = cacheBustedURL(u)
            if !urls.contains(busted) { urls.append(busted) }
        }
        if !AppSettings.shared.jsonBaseURL.isEmpty,
           let u = URL(string: AppSettings.shared.jsonBaseURL + rel) {
            let busted = cacheBustedURL(u)
            if !urls.contains(busted) { urls.append(busted) }
        }
        return urls
    }

    static func githubJSONURL(for relPath: String) -> URL? {
        cacheBustedURL(
            URL(string: "https://raw.githubusercontent.com/\(githubRepo)/\(githubBranch)/research/\(relPath)")
        )
    }

    static func githubChartURL(ticker: String) -> URL? {
        cacheBustedURL(
            URL(string: "https://raw.githubusercontent.com/\(githubRepo)/\(githubBranch)/research/charts/\(ticker.uppercased()).json")
        )
    }

    /// K 线实时 API URL（每次请求拉 yfinance）
    static func liveChartURL(ticker: String, period: ChartPeriod) -> URL? {
        let sym = ticker.uppercased()
        for base in AppSettings.shared.chartAPIBaseCandidates() {
            guard var comp = URLComponents(string: base + "v1/chart/\(sym)") else { continue }
            comp.queryItems = [
                URLQueryItem(name: "period", value: period.rawValue),
                URLQueryItem(name: "_t", value: String(Int(Date().timeIntervalSince1970))),
            ]
            if let url = comp.url { return url }
        }
        return nil
    }

    /// 禁止缓存，确保每次拉取云端最新
    static func cacheBustedURL(_ url: URL?) -> URL? {
        guard let url else { return nil }
        guard var comp = URLComponents(url: url, resolvingAgainstBaseURL: false) else { return url }
        var items = comp.queryItems ?? []
        items.removeAll { $0.name == "_t" }
        items.append(URLQueryItem(name: "_t", value: String(Int(Date().timeIntervalSince1970))))
        comp.queryItems = items
        return comp.url ?? url
    }

    static func requestTimeout(for url: URL) -> TimeInterval {
        guard let host = url.host?.lowercased() else { return 12 }
        if host.contains("onrender.com") { return 45 }
        if host == "raw.githubusercontent.com" { return 20 }
        if host == "localhost" || host.hasPrefix("127.") || host.hasPrefix("192.168.") || host.hasPrefix("10.") {
            return 4
        }
        return 12
    }
}
