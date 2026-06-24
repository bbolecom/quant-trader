import Foundation

struct OHLCVBar: Identifiable, Equatable {
    let id: Int
    let date: Date
    let open: Double
    let high: Double
    let low: Double
    let close: Double
    let volume: Double

    /// 同花顺：红涨绿跌
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

    /// 屏幕上默认展示的 K 线根数
    var visibleBars: Int {
        switch self {
        case .daily: return 60
        case .weekly: return 52
        case .monthly: return 48
        }
    }
}

enum StockChartService {
    private static let userAgent = "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1"

    static func fetchBars(ticker: String, period: ChartPeriod = .daily) async throws -> [OHLCVBar] {
        let sym = ticker.trimmingCharacters(in: .whitespacesAndNewlines).uppercased()
        guard !sym.isEmpty, sym != "—" else { throw URLError(.badURL) }

        let hosts = ["query1.finance.yahoo.com", "query2.finance.yahoo.com"]
        var lastError: Error = URLError(.badServerResponse)

        for host in hosts {
            guard let url = URL(string: "https://\(host)/v8/finance/chart/\(sym)?interval=\(period.interval)&range=\(period.range)") else {
                continue
            }
            do {
                var req = URLRequest(url: url)
                req.timeoutInterval = 20
                req.setValue(userAgent, forHTTPHeaderField: "User-Agent")
                req.setValue("application/json", forHTTPHeaderField: "Accept")
                let (data, resp) = try await URLSession.shared.data(for: req)
                guard let http = resp as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
                    throw URLError(.badServerResponse)
                }
                let bars = try parseChartJSON(data)
                if !bars.isEmpty { return bars }
            } catch {
                lastError = error
            }
        }
        throw lastError
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

    private static func parseChartJSON(_ data: Data) throws -> [OHLCVBar] {
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
            if let n = value as? Double { return n }
            if let n = value as? Int { return Double(n) }
            if let n = value as? NSNumber { return n.doubleValue }
            return nil
        }
    }
}

@MainActor
final class StockChartLoader: ObservableObject {
    @Published var bars: [OHLCVBar] = []
    @Published var ma20: [Double?] = []
    @Published var ma50: [Double?] = []
    @Published var isLoading = false
    @Published var errorMessage: String?
    @Published var period: ChartPeriod = .daily
    @Published var selectedIndex: Int?

    var displayBars: [OHLCVBar] {
        let n = period.visibleBars
        guard bars.count > n else { return bars }
        return Array(bars.suffix(n))
    }

    var displayMA20: [Double?] {
        sliceMA(ma20)
    }

    var displayMA50: [Double?] {
        sliceMA(ma50)
    }

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
            let fetched = try await StockChartService.fetchBars(ticker: ticker, period: period)
            bars = fetched
            let closes = fetched.map(\.close)
            ma20 = StockChartService.movingAverage(closes, period: 20)
            ma50 = StockChartService.movingAverage(closes, period: 50)
        } catch {
            bars = []
            ma20 = []
            ma50 = []
            errorMessage = error.localizedDescription
        }
    }
}
