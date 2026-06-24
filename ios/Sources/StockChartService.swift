import Foundation

struct OHLCVBar: Identifiable, Equatable {
    let id: Int
    let date: Date
    let open: Double
    let high: Double
    let low: Double
    let close: Double
    let volume: Double

    var isUp: Bool { close >= open }
}

enum ChartPeriod: String, CaseIterable, Identifiable {
    case daily
    case weekly
    case monthly

    var id: String { rawValue }

    var title: String {
        switch self {
        case .daily: return "日K"
        case .weekly: return "周K"
        case .monthly: return "月K"
        }
    }

    var interval: String {
        switch self {
        case .daily: return "1d"
        case .weekly: return "1wk"
        case .monthly: return "1mo"
        }
    }

    var range: String {
        switch self {
        case .daily: return "6mo"
        case .weekly: return "2y"
        case .monthly: return "5y"
        }
    }

    var visibleBars: Int {
        switch self {
        case .daily: return 60
        case .weekly: return 52
        case .monthly: return 48
        }
    }
}

enum ChartDataSource: String {
    case bundled = "内置"
    case github = "GitHub"
    case mac = "Mac"
    case yahoo = "Yahoo"
}

enum StockChartService {
    private static let userAgent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    private static let yahooSession: URLSession = {
        let cfg = URLSessionConfiguration.default
        cfg.httpCookieStorage = HTTPCookieStorage.shared
        cfg.httpShouldSetCookies = true
        cfg.httpCookieAcceptPolicy = .always
        return URLSession(configuration: cfg)
    }()
    private static var yahooCrumb: String?

    static func fetchBars(ticker: String, period: ChartPeriod = .daily) async throws -> ([OHLCVBar], ChartDataSource) {
        let sym = ticker.trimmingCharacters(in: .whitespacesAndNewlines).uppercased()
        guard !sym.isEmpty, sym != "—" else { throw URLError(.badURL) }

        var lastError: Error = URLError(.badServerResponse)

        for url in chartCandidateURLs(ticker: sym) {
            do {
                let bars = try await loadSnapshot(url: url, period: period)
                if !bars.isEmpty {
                    let src: ChartDataSource = url.host == "raw.githubusercontent.com" ? .github : (url.isFileURL ? .bundled : .mac)
                    return (bars, src)
                }
            } catch {
                lastError = error
            }
        }

        do {
            let bars = try await fetchYahooBars(symbol: sym, period: period)
            if !bars.isEmpty { return (bars, .yahoo) }
        } catch {
            lastError = error
        }

        throw lastError
    }

    static func chartCandidateURLs(ticker: String) -> [URL] {
        var urls: [URL] = []
        let sym = ticker.uppercased()
        if let u = bundledChartURL(ticker: sym) { urls.append(u) }
        if let gh = githubChartURL(ticker: sym) { urls.append(gh) }
        if let u = AppSettings.shared.jsonURL(for: "charts/\(sym).json") { urls.append(u) }
        return urls
    }

    static func bundledChartURL(ticker: String) -> URL? {
        let sym = ticker.uppercased()
        if let u = Bundle.main.url(forResource: sym, withExtension: "json", subdirectory: "charts") {
            return u
        }
        if let u = Bundle.main.url(forResource: "chart_\(sym)", withExtension: "json") {
            return u
        }
        return nil
    }

    static func githubChartURL(ticker: String) -> URL? {
        URL(string: "https://raw.githubusercontent.com/\(AppConfig.githubRepo)/\(AppConfig.githubBranch)/research/charts/\(ticker.uppercased()).json")
    }

    private static func loadSnapshot(url: URL, period: ChartPeriod) async throws -> [OHLCVBar] {
        let data: Data
        if url.isFileURL {
            data = try Data(contentsOf: url)
        } else {
            var req = URLRequest(url: url)
            req.timeoutInterval = 15
            req.setValue(userAgent, forHTTPHeaderField: "User-Agent")
            let (d, resp) = try await URLSession.shared.data(for: req)
            guard let http = resp as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
                throw URLError(.badServerResponse)
            }
            data = d
        }
        return try parseSnapshotJSON(data, period: period)
    }

    private static func parseSnapshotJSON(_ data: Data, period: ChartPeriod) throws -> [OHLCVBar] {
        let root = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        if let arr = root?["bars"] as? [[String: Any]] {
            let fmt = ISO8601DateFormatter()
            fmt.formatOptions = [.withFullDate]
            var bars: [OHLCVBar] = []
            for (i, row) in arr.enumerated() {
                guard let close = doubleValue(row["close"]) else { continue }
                let dstr = row["date"] as? String ?? ""
                let dt = fmt.date(from: String(dstr.prefix(10))) ?? Date(timeIntervalSince1970: 0)
                bars.append(OHLCVBar(
                    id: i,
                    date: dt,
                    open: doubleValue(row["open"]) ?? close,
                    high: doubleValue(row["high"]) ?? close,
                    low: doubleValue(row["low"]) ?? close,
                    close: close,
                    volume: doubleValue(row["volume"]) ?? 0
                ))
            }
            return resample(bars, period: period)
        }
        return try parseYahooChartJSON(data)
    }

    private static func resample(_ bars: [OHLCVBar], period: ChartPeriod) -> [OHLCVBar] {
        guard period == .daily || bars.isEmpty else { return bars }
        let n = period.visibleBars * 2
        if bars.count > n { return Array(bars.suffix(n)) }
        return bars
    }

    private static func fetchYahooBars(symbol: String, period: ChartPeriod) async throws -> [OHLCVBar] {
        let hosts = ["query1.finance.yahoo.com", "query2.finance.yahoo.com"]
        var lastError: Error = URLError(.badServerResponse)
        for host in hosts {
            do {
                try await ensureYahooCrumb()
                var comp = "https://\(host)/v8/finance/chart/\(symbol)?interval=\(period.interval)&range=\(period.range)"
                if let crumb = yahooCrumb, !crumb.isEmpty, !crumb.contains("Too Many") {
                    comp += "&crumb=\(crumb.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed) ?? crumb)"
                }
                guard let url = URL(string: comp) else { continue }
                var req = URLRequest(url: url)
                req.timeoutInterval = 20
                req.setValue(userAgent, forHTTPHeaderField: "User-Agent")
                req.setValue("application/json", forHTTPHeaderField: "Accept")
                let (data, resp) = try await yahooSession.data(for: req)
                guard let http = resp as? HTTPURLResponse else { throw URLError(.badServerResponse) }
                if http.statusCode == 429 {
                    throw NSError(domain: "StockChart", code: 429, userInfo: [NSLocalizedDescriptionKey: "Yahoo 限流，请使用内置快照"])
                }
                guard (200..<300).contains(http.statusCode) else { throw URLError(.badServerResponse) }
                let bars = try parseYahooChartJSON(data)
                if !bars.isEmpty { return bars }
            } catch {
                lastError = error
            }
        }
        throw lastError
    }

    private static func ensureYahooCrumb() async throws {
        if let c = yahooCrumb, !c.isEmpty, !c.contains("Too Many") { return }
        var warm = URLRequest(url: URL(string: "https://finance.yahoo.com/quote/SPY")!)
        warm.setValue(userAgent, forHTTPHeaderField: "User-Agent")
        _ = try await yahooSession.data(for: warm)
        var crumbReq = URLRequest(url: URL(string: "https://query2.finance.yahoo.com/v1/test/getcrumb")!)
        crumbReq.setValue(userAgent, forHTTPHeaderField: "User-Agent")
        let (data, resp) = try await yahooSession.data(for: crumbReq)
        guard let http = resp as? HTTPURLResponse, (200..<300).contains(http.statusCode) else { return }
        let crumb = String(data: data, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        if !crumb.isEmpty { yahooCrumb = crumb }
    }

    static func movingAverage(_ closes: [Double], period: Int) -> [Double?] {
        guard period > 0 else { return [] }
        var out: [Double?] = Array(repeating: nil, count: closes.count)
        guard closes.count >= period else { return out }
        var sum = closes.prefix(period).reduce(0, +)
        out[period - 1] = sum / Double(period)
        if period < closes.count {
            for i in period..<closes.count {
                sum += closes[i] - closes[i - period]
                out[i] = sum / Double(period)
            }
        }
        return out
    }

    private static func parseYahooChartJSON(_ data: Data) throws -> [OHLCVBar] {
        let root = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        let chart = root?["chart"] as? [String: Any]
        if let err = chart?["error"] as? [String: Any], let desc = err["description"] as? String {
            throw NSError(domain: "StockChart", code: 1, userInfo: [NSLocalizedDescriptionKey: desc])
        }
        let results = chart?["result"] as? [[String: Any]]
        guard let result = results?.first else { throw URLError(.cannotParseResponse) }

        let timestamps = result["timestamp"] as? [Int] ?? []
        let indicators = result["indicators"] as? [String: Any]
        let quotes = indicators?["quote"] as? [[String: Any]]
        guard let quote = quotes?.first else { throw URLError(.cannotParseResponse) }

        let opens = optionalDoubles(from: quote, key: "open")
        let highs = optionalDoubles(from: quote, key: "high")
        let lows = optionalDoubles(from: quote, key: "low")
        let closes = optionalDoubles(from: quote, key: "close")
        let volumes = optionalDoubles(from: quote, key: "volume")

        var bars: [OHLCVBar] = []
        for i in 0..<timestamps.count {
            guard i < opens.count, i < highs.count, i < lows.count, i < closes.count,
                  let o = opens[i], let h = highs[i], let l = lows[i], let c = closes[i] else { continue }
            let vol = (i < volumes.count ? volumes[i] : nil) ?? 0
            let date = Date(timeIntervalSince1970: TimeInterval(timestamps[i]))
            bars.append(OHLCVBar(id: bars.count, date: date, open: o, high: h, low: l, close: c, volume: vol))
        }
        return bars
    }

    private static func optionalDoubles(from quote: [String: Any], key: String) -> [Double?] {
        guard let arr = quote[key] as? [Any] else { return [] }
        return arr.map { value -> Double? in
            if value is NSNull { return nil }
            return doubleValue(value)
        }
    }

    private static func doubleValue(_ value: Any?) -> Double? {
        if value is NSNull { return nil }
        if let n = value as? Double { return n }
        if let n = value as? Int { return Double(n) }
        if let n = value as? NSNumber { return n.doubleValue }
        if let s = value as? String { return Double(s) }
        return nil
    }
}

@MainActor
final class StockChartLoader: ObservableObject {
    @Published var bars: [OHLCVBar] = []
    @Published var ma20: [Double?] = []
    @Published var ma50: [Double?] = []
    @Published var isLoading = false
    @Published var errorMessage: String?
    @Published var dataSource: ChartDataSource?
    @Published var period: ChartPeriod = .daily
    @Published var selectedIndex: Int?

    var displayBars: [OHLCVBar] {
        let n = period.visibleBars
        guard bars.count > n else { return bars }
        return Array(bars.suffix(n))
    }

    var displayMA20: [Double?] { sliceMA(ma20) }
    var displayMA50: [Double?] { sliceMA(ma50) }

    private func sliceMA(_ ma: [Double?]) -> [Double?] {
        let n = period.visibleBars
        guard ma.count > n else { return ma }
        return Array(ma.suffix(n))
    }

    var lastPrice: Double? { bars.last?.close }
    var changePct: Double? {
        guard bars.count >= 2, let last = bars.last?.close else { return nil }
        let prev = bars[bars.count - 2].close
        guard prev != 0 else { return nil }
        return (last / prev - 1) * 100
    }

    var selectedBar: OHLCVBar? {
        guard let idx = selectedIndex, displayBars.indices.contains(idx) else { return nil }
        return displayBars[idx]
    }

    func load(ticker: String) async {
        isLoading = true
        errorMessage = nil
        selectedIndex = nil
        defer { isLoading = false }
        do {
            let (fetched, src) = try await StockChartService.fetchBars(ticker: ticker, period: period)
            bars = fetched
            dataSource = src
            let closes = fetched.map(\.close)
            ma20 = StockChartService.movingAverage(closes, period: 20)
            ma50 = StockChartService.movingAverage(closes, period: 50)
        } catch {
            bars = []
            ma20 = []
            ma50 = []
            dataSource = nil
            errorMessage = friendlyError(error)
        }
    }

    private func friendlyError(_ error: Error) -> String {
        let msg = error.localizedDescription
        if msg.contains("429") || msg.contains("限流") {
            return "行情源限流 · 内置快照未找到，请联网后下拉刷新"
        }
        if (error as NSError).code == NSURLErrorBadServerResponse {
            return "行情源不可用 · 将使用内置/云端快照"
        }
        return msg
    }
}
