import Foundation

/// 编译时默认；运行时可在「我的」修改（AppSettings）
enum AppConfig {
    static let defaultServerURLString = "https://quant-trader-fd3mch56aixtttm5rgyc6i.streamlit.app"

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
        var urls: [URL] = []
        if let u = AppSettings.shared.jsonURL(for: "daily_pick_today.json") {
            urls.append(u)
        }
        if !AppSettings.shared.jsonBaseURL.isEmpty,
           let u = URL(string: AppSettings.shared.jsonBaseURL + "daily_pick_today.json") {
            if !urls.contains(u) { urls.append(u) }
        }
        return urls
    }
}
