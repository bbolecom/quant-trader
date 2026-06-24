import Foundation

/// 编译时默认；运行时可在「我的」修改（AppSettings）
enum AppConfig {
    static let defaultServerURLString = "https://quant-trader-fd3mch56aixtttm5rgyc6i.streamlit.app"
    static let githubRepo = "bbolecom/quant-trader"
    static let githubBranch = "main"

    static var githubDailyPickURL: URL? {
        URL(string: "https://raw.githubusercontent.com/\(githubRepo)/\(githubBranch)/research/daily_pick_today.json")
    }

    static var githubManifestURL: URL? {
        URL(string: "https://raw.githubusercontent.com/\(githubRepo)/\(githubBranch)/research/app_manifest.json")
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

    /// 模块 JSON 候选 URL：GitHub 云端 → 本地 Mac（避免局域网超时阻塞）
    static func jsonCandidateURLs(for path: String) -> [URL] {
        var urls: [URL] = []
        let rel = jsonRelativePath(path)

        if let gh = githubJSONURL(for: rel) {
            urls.append(gh)
        }
        if let u = AppSettings.shared.jsonURL(for: path), !urls.contains(u) {
            urls.append(u)
        }
        if !AppSettings.shared.jsonBaseURL.isEmpty,
           let u = URL(string: AppSettings.shared.jsonBaseURL + rel),
           !urls.contains(u) {
            urls.append(u)
        }
        return urls
    }

    static func githubJSONURL(for relPath: String) -> URL? {
        URL(string: "https://raw.githubusercontent.com/\(githubRepo)/\(githubBranch)/research/\(relPath)")
    }

    /// App 包内内置 JSON（离线 / Mac 未开时使用）
    static func bundledJSONURL(for path: String) -> URL? {
        let rel = jsonRelativePath(path)
        let base = (rel as NSString).deletingPathExtension
        let ext = (rel as NSString).pathExtension
        let useExt = ext.isEmpty ? "json" : ext
        return Bundle.main.url(forResource: base, withExtension: useExt)
    }

    static func requestTimeout(for url: URL) -> TimeInterval {
        guard let host = url.host?.lowercased() else { return 12 }
        if host == "raw.githubusercontent.com" { return 20 }
        if host == "localhost" || host.hasPrefix("127.") || host.hasPrefix("192.168.") || host.hasPrefix("10.") {
            return 4
        }
        return 12
    }
}
