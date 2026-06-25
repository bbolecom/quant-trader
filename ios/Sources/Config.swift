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

    /// research/ 下 JSON 的云端镜像（jsDelivr 优先，国内更易访问；GitHub Raw 备用）。
    static func cloudJSONURLs(for relPath: String) -> [URL] {
        let rel = jsonRelativePath(relPath)
        let templates = [
            "https://cdn.jsdelivr.net/gh/\(githubRepo)@\(githubBranch)/research/\(rel)",
            "https://raw.githubusercontent.com/\(githubRepo)/\(githubBranch)/research/\(rel)",
        ]
        return templates.compactMap { cacheBustedURL(URL(string: $0)) }
    }

    static func cloudChartURLs(ticker: String) -> [URL] {
        let sym = ticker.uppercased()
        let templates = [
            "https://cdn.jsdelivr.net/gh/\(githubRepo)@\(githubBranch)/research/charts/\(sym).json",
            "https://raw.githubusercontent.com/\(githubRepo)/\(githubBranch)/research/charts/\(sym).json",
        ]
        return templates.compactMap { cacheBustedURL(URL(string: $0)) }
    }

    static func cloudManifestURLs() -> [URL] {
        cloudJSONURLs(for: "app_manifest.json")
    }

    static func hostLabel(for url: URL) -> String {
        guard let host = url.host?.lowercased() else { return "云端" }
        if host.contains("jsdelivr") { return "jsDelivr CDN" }
        if host == "raw.githubusercontent.com" { return "GitHub" }
        if host.contains("onrender.com") { return "Render" }
        if host == "localhost" || host.hasPrefix("192.168.") || host.hasPrefix("10.") {
            return "Mac 局域网"
        }
        return host
    }

    /// 模块 JSON 相对路径，如 `gain15_daily_today.json`
    static func jsonRelativePath(_ path: String) -> String {
        path
            .replacingOccurrences(of: "research/", with: "")
            .trimmingCharacters(in: CharacterSet(charactersIn: "/"))
    }

    /// 模块 JSON 候选 URL：Mac 局域网 → jsDelivr CDN → GitHub Raw
    static func jsonCandidateURLs(for path: String) -> [URL] {
        var urls: [URL] = []
        let rel = jsonRelativePath(path)

        if !AppSettings.shared.jsonBaseURL.isEmpty,
           let u = URL(string: AppSettings.shared.jsonBaseURL + rel),
           let busted = cacheBustedURL(u) {
            urls.append(busted)
        }
        if let u = AppSettings.shared.jsonURL(for: path),
           let busted = cacheBustedURL(u),
           !urls.contains(busted) {
            urls.append(busted)
        }
        for u in cloudJSONURLs(for: rel) where !urls.contains(u) {
            urls.append(u)
        }
        return urls
    }

    static func githubJSONURL(for relPath: String) -> URL? {
        cloudJSONURLs(for: relPath).first
    }

    static func githubChartURL(ticker: String) -> URL? {
        cloudChartURLs(ticker: ticker).first
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
        // Render 免费版冷启动较慢，但它现在只作后台 live 升级，不阻塞首屏 → 适度收敛。
        if host.contains("onrender.com") { return 25 }
        if host.contains("jsdelivr") { return 18 }
        if host == "raw.githubusercontent.com" { return 15 }
        if host == "localhost" || host.hasPrefix("127.") || host.hasPrefix("192.168.") || host.hasPrefix("10.") {
            return 4
        }
        return 12
    }
}

/// 云端 JSON / 快照拉取（User-Agent + 超时）。
enum CloudFetch {
    static let userAgent = "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) QuantTrader/3.0"

    static func data(from url: URL) async throws -> Data {
        var req = URLRequest(url: url)
        req.cachePolicy = .reloadIgnoringLocalCacheData
        req.timeoutInterval = AppConfig.requestTimeout(for: url)
        req.setValue(userAgent, forHTTPHeaderField: "User-Agent")
        req.setValue("application/json", forHTTPHeaderField: "Accept")
        let (data, resp) = try await URLSession.shared.data(for: req)
        guard let http = resp as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            throw URLError(.badServerResponse)
        }
        return data
    }
}
