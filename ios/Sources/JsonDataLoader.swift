import Foundation

@MainActor
final class JsonDataLoader: ObservableObject {
    @Published var root: [String: Any]?
    @Published var isLoading = false
    @Published var errorMessage: String?
    @Published var loadedFrom: String?

    func load(path: String) async {
        isLoading = true
        errorMessage = nil
        defer { isLoading = false }

        if root == nil, let bundled = Self.loadBundled(path: path) {
            apply(bundled, from: "内置")
        }

        var lastError: String?
        var triedRemote = false
        for url in AppConfig.jsonCandidateURLs(for: path) {
            triedRemote = true
            do {
                var req = URLRequest(url: url)
                req.cachePolicy = .reloadIgnoringLocalCacheData
                req.timeoutInterval = AppConfig.requestTimeout(for: url)
                let (data, resp) = try await URLSession.shared.data(for: req)
                guard let http = resp as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
                    lastError = "无法获取数据（\(url.lastPathComponent)）"
                    continue
                }
                let obj = try JSONSerialization.jsonObject(with: data)
                guard let dict = obj as? [String: Any] else {
                    lastError = "JSON 格式不是对象"
                    continue
                }
                apply(dict, from: sourceLabel(for: url))
                return
            } catch {
                lastError = error.localizedDescription
            }
        }

        if root != nil {
            if triedRemote {
                errorMessage = "云端未连接 · 已显示内置快照（可下拉刷新）"
            }
            return
        }

        errorMessage = lastError ?? "无法获取数据"
    }

    private func apply(_ dict: [String: Any], from source: String) {
        root = dict
        loadedFrom = source
    }

    private func sourceLabel(for url: URL) -> String {
        if url.host == "raw.githubusercontent.com" { return "GitHub" }
        return url.host ?? "远程"
    }

    private static func loadBundled(path: String) -> [String: Any]? {
        guard let url = AppConfig.bundledJSONURL(for: path),
              let data = try? Data(contentsOf: url),
              let obj = try? JSONSerialization.jsonObject(with: data),
              let dict = obj as? [String: Any] else {
            return nil
        }
        return dict
    }
}

// MARK: - JSON helpers

enum JsonHelper {
    static func string(_ dict: [String: Any], _ keys: String...) -> String? {
        for k in keys {
            if let v = dict[k] as? String, !v.isEmpty { return v }
            if let v = dict[k] as? Int { return String(v) }
            if let v = dict[k] as? Double { return formatNum(v) }
        }
        return nil
    }

    static func double(_ dict: [String: Any], _ key: String) -> Double? {
        if let v = dict[key] as? Double { return v }
        if let v = dict[key] as? Int { return Double(v) }
        if let v = dict[key] as? String { return Double(v) }
        return nil
    }

    static func array(_ dict: [String: Any], _ keys: String...) -> [[String: Any]] {
        for k in keys {
            if let arr = dict[k] as? [[String: Any]] { return arr }
        }
        return []
    }

    static func formatNum(_ v: Double) -> String {
        if abs(v) >= 100 || v == v.rounded() { return String(format: "%.0f", v) }
        return String(format: "%.2f", v)
    }

    static func formatPct(_ v: Double?) -> String {
        guard let v else { return "—" }
        let pct = abs(v) <= 1.5 ? v * 100 : v
        return String(format: "%+.1f%%", pct)
    }
}
