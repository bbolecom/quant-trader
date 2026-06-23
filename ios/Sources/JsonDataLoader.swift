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
        root = nil
        defer { isLoading = false }

        guard let url = AppSettings.shared.jsonURL(for: path) else {
            errorMessage = "未配置 JSON 地址"
            return
        }

        do {
            var req = URLRequest(url: url)
            req.cachePolicy = .reloadIgnoringLocalCacheData
            req.timeoutInterval = 15
            let (data, resp) = try await URLSession.shared.data(for: req)
            guard let http = resp as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
                errorMessage = "无法获取数据（\(url.lastPathComponent)）"
                return
            }
            let obj = try JSONSerialization.jsonObject(with: data)
            if let dict = obj as? [String: Any] {
                root = dict
                loadedFrom = url.host
            } else {
                errorMessage = "JSON 格式不是对象"
            }
        } catch {
            errorMessage = error.localizedDescription
        }
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
