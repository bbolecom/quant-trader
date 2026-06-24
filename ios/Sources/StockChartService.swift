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

struct StockQuoteSnapshot {
    let ticker: String
    let lastPrice: Double
    let changePct: Double
    let currency: String
}

enum StockChartService {
    static func fetchBars(ticker: String, range: String = "3mo") async throws -> [OHLCVBar] {
        let sym = ticker.trimmingCharacters(in: .whitespacesAndNewlines).uppercased()
        guard !sym.isEmpty, sym != "—" else { throw URLError(.badURL) }
        guard let url = URL(string: "https://query1.finance.yahoo.com/v8/finance/chart/\(sym)?interval=1d&range=\(range)") else {
            throw URLError(.badURL)
        }
        var req = URLRequest(url: url)
        req.timeoutInterval = 20
        let (data, resp) = try await URLSession.shared.data(for: req)
        guard let http = resp as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            throw URLError(.badServerResponse)
        }
        return try parseChartJSON(data)
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
            bars.append(OHLCVBar(id: i, date: date, open: o, high: h, low: l, close: c, volume: vol))
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

    var closes: [Double] { bars.map(\.close) }
    var lastPrice: Double? { bars.last?.close }
    var changePct: Double? {
        guard bars.count >= 2, let last = bars.last?.close else { return nil }
        let prev = bars[bars.count - 2].close
        guard prev != 0 else { return nil }
        return (last / prev - 1) * 100
    }

    func load(ticker: String) async {
        isLoading = true
        errorMessage = nil
        defer { isLoading = false }
        do {
            let fetched = try await StockChartService.fetchBars(ticker: ticker)
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
