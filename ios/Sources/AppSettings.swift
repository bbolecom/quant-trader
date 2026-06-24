import Foundation
import SwiftUI

/// 运行时配置（同花顺「我的」可改服务器地址）
final class AppSettings: ObservableObject {
    static let shared = AppSettings()

    @AppStorage("jsonBaseURL") var jsonBaseURL: String = ""
    @AppStorage("streamlitURL") var streamlitURL: String = AppConfig.defaultServerURLString
    @AppStorage("chartAPIURL") var chartAPIURL: String = AppConfig.defaultChartAPIBase

    var serverURL: URL {
        URL(string: streamlitURL) ?? AppConfig.fallbackServerURL
    }

    /// 拼接 research 下 JSON 路径，如 `daily_pick_today.json`
    func jsonURL(for path: String) -> URL? {
        let rel = path
            .replacingOccurrences(of: "research/", with: "")
            .trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        for base in jsonBaseCandidates() {
            if let url = URL(string: base + rel) { return url }
        }
        return nil
    }

    func jsonBaseCandidates() -> [String] {
        var bases: [String] = []
        if !jsonBaseURL.isEmpty {
            var b = jsonBaseURL
            if !b.hasSuffix("/") { b += "/" }
            bases.append(b)
        }
        if let derived = derivedJSONBase() {
            bases.append(derived)
        }
        return bases
    }

    /// K 线实时 API 基址候选：用户配置 → Mac :8503 → Render 默认
    func chartAPIBaseCandidates() -> [String] {
        var bases: [String] = []
        if !chartAPIURL.isEmpty {
            var b = chartAPIURL.trimmingCharacters(in: .whitespacesAndNewlines)
            if !b.hasSuffix("/") { b += "/" }
            bases.append(b)
        }
        if let derived = derivedChartAPIBase() {
            bases.append(derived)
        }
        if chartAPIURL.isEmpty, !AppConfig.defaultChartAPIBase.isEmpty {
            var b = AppConfig.defaultChartAPIBase
            if !b.hasSuffix("/") { b += "/" }
            if !bases.contains(b) { bases.append(b) }
        }
        return bases
    }

    private func derivedJSONBase() -> String? {
        guard let base = URL(string: streamlitURL),
              let host = base.host,
              base.scheme == "http" else { return nil }
        let port = base.port == 8501 ? 8502 : (base.port ?? 8502)
        return "http://\(host):\(port)/"
    }

    private func derivedChartAPIBase() -> String? {
        guard let base = URL(string: streamlitURL),
              let host = base.host,
              base.scheme == "http" else { return nil }
        return "http://\(host):8503/"
    }

    var jsonURLHint: String {
        if !jsonBaseURL.isEmpty { return jsonBaseURL }
        if let d = derivedJSONBase() { return d }
        return "GitHub 云端（raw.githubusercontent.com）"
    }

    var chartAPIHint: String {
        chartAPIBaseCandidates().first ?? AppConfig.defaultChartAPIBase
    }
}

/// 全局 Tab 切换（从功能页跳转量化终端）
final class AppNavigation: ObservableObject {
    @Published var selectedTab = 0

    func openTerminal() {
        selectedTab = 3
    }

    func openPicks() {
        selectedTab = 2
    }

    func openHome() {
        selectedTab = 0
    }
}
