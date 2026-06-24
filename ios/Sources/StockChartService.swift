import Foundation

struct OHLCVBar: Identifiable, Equatable, Sendable {
    let id: Int
    let date: Date
    let open: Double
    let high: Double
    let low: Double
    let close: Double
    let volume: Double

    var isUp: Bool { close >= open }
}

enum ChartPeriod: String, CaseIterable, Identifiable, Sendable {
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

enum ChartDataSource: String, Sendable {
    case cloud = "云端实时"
    case github = "GitHub"
    case mac = "Mac"
    case bundled = "内置快照"
}

private actor StockChartCache {
    struct Entry {
        let bars: [OHLCVBar]
        let source: ChartDataSource
        let updatedAt: String?
        let cachedAt: Date
    }

    private var storage: [String: Entry] = [:]
    private let ttl: TimeInterval = 45

    func value(ticker: String, period: ChartPeriod) -> ([OHLCVBar], ChartDataSource, String?)? {
        let key = cacheKey(ticker: ticker, period: period)
        guard let entry = storage[key] else { return nil }
        if Date().timeIntervalSince(entry.cachedAt) > ttl {
            storage[key] = nil
            return nil
        }
        return (entry.bars, entry.source, entry.updatedAt)
    }

    func store(_ bars: [OHLCVBar], source: ChartDataSource, updatedAt: String?, ticker: String, period: ChartPeriod) {
        storage[cacheKey(ticker: ticker, period: period)] = Entry(
            bars: bars,
            source: source,
            updatedAt: updatedAt,
            cachedAt: Date()
        )
    }

    private func cacheKey(ticker: String, period: ChartPeriod) -> String {
        "\(ticker.uppercased())-\(period.rawValue)"
    }
}

enum StockChartService {
    private static let userAgent = "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) QuantTrader/3.0"
    private static let cache = StockChartCache()

    static func normalize(_ ticker: String) -> String {
        ticker.trimmingCharacters(in: .whitespacesAndNewlines).uppercased()
    }

    /// 兼容旧调用：先秒出快照，失败再走 live。
    static func fetchBars(ticker: String, period: ChartPeriod = .daily) async throws -> ([OHLCVBar], ChartDataSource, String?) {
        let sym = normalize(ticker)
        if let snap = try? await fetchSnapshot(ticker: sym, period: period) {
            return snap
        }
        return try await fetchLive(ticker: sym, period: period)
    }

    /// 快照源（快）：内存缓存 → GitHub 快照（每30分钟刷新）→ Mac 局域网 → 内置快照。
    /// 不打 Render 云端，避免冷启动阻塞首屏。
    static func fetchSnapshot(ticker: String, period: ChartPeriod = .daily) async throws -> ([OHLCVBar], ChartDataSource, String?) {
        let sym = normalize(ticker)
        guard !sym.isEmpty, sym != "—" else { throw URLError(.badURL) }
        if let cached = await cache.value(ticker: sym, period: period) {
            return cached
        }

        var lastError: Error = URLError(.badServerResponse)

        if let gh = AppConfig.githubChartURL(ticker: sym) {
            do {
                let (bars, stamp) = try await loadRemoteSnapshot(url: gh, period: period)
                if !bars.isEmpty {
                    await cache.store(bars, source: .github, updatedAt: stamp, ticker: sym, period: period)
                    return (bars, .github, stamp)
                }
            } catch { lastError = error }
        }

        if let u = AppSettings.shared.jsonURL(for: "charts/\(sym).json") {
            let busted = AppConfig.cacheBustedURL(u) ?? u
            do {
                let (bars, stamp) = try await loadRemoteSnapshot(url: busted, period: period)
                if !bars.isEmpty {
                    await cache.store(bars, source: .mac, updatedAt: stamp, ticker: sym, period: period)
                    return (bars, .mac, stamp)
                }
            } catch { lastError = error }
        }

        if let url = Bundle.main.url(forResource: sym, withExtension: "json", subdirectory: "charts") {
            do {
                let data = try Data(contentsOf: url)
                let (bars, stamp) = try parseSnapshotJSON(data, period: period)
                if !bars.isEmpty {
                    await cache.store(bars, source: .bundled, updatedAt: stamp, ticker: sym, period: period)
                    return (bars, .bundled, stamp)
                }
            } catch { lastError = error }
        }

        throw lastError
    }

    /// Live 源：Render 云端实时 API（yfinance 服务端拉取）。仅作后台升级用。
    static func fetchLive(ticker: String, period: ChartPeriod = .daily) async throws -> ([OHLCVBar], ChartDataSource, String?) {
        let sym = normalize(ticker)
        guard !sym.isEmpty, sym != "—" else { throw URLError(.badURL) }
        guard let url = AppConfig.liveChartURL(ticker: sym, period: period) else {
            throw URLError(.badURL)
        }
        let (bars, stamp) = try await loadRemoteSnapshot(url: url, period: period)
        guard !bars.isEmpty else { throw URLError(.zeroByteResource) }
        await cache.store(bars, source: .cloud, updatedAt: stamp, ticker: sym, period: period)
        return (bars, .cloud, stamp)
    }

    private static func loadRemoteSnapshot(url: URL, period: ChartPeriod) async throws -> ([OHLCVBar], String?) {
        var req = URLRequest(url: url)
        req.cachePolicy = .reloadIgnoringLocalCacheData
        req.timeoutInterval = AppConfig.requestTimeout(for: url)
        req.setValue(userAgent, forHTTPHeaderField: "User-Agent")
        let (data, resp) = try await URLSession.shared.data(for: req)
        guard let http = resp as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            throw URLError(.badServerResponse)
        }
        return try parseSnapshotJSON(data, period: period)
    }

    private static func parseSnapshotJSON(_ data: Data, period: ChartPeriod) throws -> ([OHLCVBar], String?) {
        let root = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        let updated = root?["updated"] as? String
        let interval = root?["interval"] as? String ?? "1d"
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
            return (normalizeBars(bars, period: period, interval: interval), updated)
        }
        let bars = try parseYahooChartJSON(data)
        return (normalizeBars(bars, period: period, interval: interval), updated)
    }

    /// 日K 快照转周/月K；云端 API 已按周期返回则不再聚合
    private static func normalizeBars(_ bars: [OHLCVBar], period: ChartPeriod, interval: String) -> [OHLCVBar] {
        guard !bars.isEmpty else { return bars }
        if period == .daily || interval != "1d" { return reindex(bars) }
        return reindex(aggregateBars(bars, period: period))
    }

    private static func reindex(_ bars: [OHLCVBar]) -> [OHLCVBar] {
        bars.enumerated().map { i, bar in
            OHLCVBar(id: i, date: bar.date, open: bar.open, high: bar.high, low: bar.low, close: bar.close, volume: bar.volume)
        }
    }

    private static func aggregateBars(_ bars: [OHLCVBar], period: ChartPeriod) -> [OHLCVBar] {
        let cal = Calendar.current
        var order: [Date] = []
        var groups: [Date: [OHLCVBar]] = [:]

        for bar in bars {
            let key: Date
            switch period {
            case .weekly:
                let comps = cal.dateComponents([.yearForWeekOfYear, .weekOfYear], from: bar.date)
                key = cal.date(from: comps) ?? bar.date
            case .monthly:
                let comps = cal.dateComponents([.year, .month], from: bar.date)
                key = cal.date(from: comps) ?? bar.date
            case .daily:
                key = bar.date
            }
            if groups[key] == nil {
                order.append(key)
                groups[key] = []
            }
            groups[key]?.append(bar)
        }

        return order.enumerated().compactMap { i, key in
            guard let chunk = groups[key], let first = chunk.first, let last = chunk.last else { return nil }
            return OHLCVBar(
                id: i,
                date: key,
                open: first.open,
                high: chunk.map(\.high).max() ?? first.high,
                low: chunk.map(\.low).min() ?? first.low,
                close: last.close,
                volume: chunk.map(\.volume).reduce(0, +)
            )
        }
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
    @Published var ma5: [Double?] = []
    @Published var ma10: [Double?] = []
    @Published var ma20: [Double?] = []
    @Published var ma60: [Double?] = []
    @Published var isLoading = false
    @Published var errorMessage: String?
    @Published var dataSource: ChartDataSource?
    @Published var cloudUpdatedAt: String?
    @Published var lastFetched: Date?
    @Published var period: ChartPeriod = .daily
    @Published var selectedIndex: Int?
    /// 从右缘向左滑动的 K 线根数（0 = 最新）
    @Published var scrollOffset: Int = 0

    private var loadGeneration = 0

    var maxScrollOffset: Int {
        max(0, bars.count - period.visibleBars)
    }

    private var windowRange: Range<Int> {
        let n = period.visibleBars
        guard bars.count > n else { return 0..<bars.count }
        let offset = min(max(scrollOffset, 0), maxScrollOffset)
        let end = bars.count - offset
        let start = max(0, end - n)
        return start..<end
    }

    var displayBars: [OHLCVBar] {
        let r = windowRange
        guard !r.isEmpty else { return [] }
        return Array(bars[r])
    }

    var displayMA5: [Double?] { sliceMAWindow(ma5) }
    var displayMA10: [Double?] { sliceMAWindow(ma10) }
    var displayMA20: [Double?] { sliceMAWindow(ma20) }
    var displayMA60: [Double?] { sliceMAWindow(ma60) }

    private func sliceMAWindow(_ ma: [Double?]) -> [Double?] {
        let r = windowRange
        guard r.upperBound <= ma.count else { return [] }
        return Array(ma[r])
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

    func setPeriod(_ newPeriod: ChartPeriod, ticker: String) {
        guard period != newPeriod else { return }
        period = newPeriod
        scrollOffset = 0
        selectedIndex = nil
        Task { await load(ticker: ticker) }
    }

    /// 两阶段加载：① 先秒出快照让首屏不空白；② 后台拉云端 live 升级（成功才覆盖）。
    func load(ticker: String) async {
        loadGeneration += 1
        let generation = loadGeneration
        isLoading = true
        errorMessage = nil

        var gotSnapshot = false
        do {
            let snap = try await StockChartService.fetchSnapshot(ticker: ticker, period: period)
            guard generation == loadGeneration else { return }
            applyBars(snap.0, source: snap.1, stamp: snap.2)
            gotSnapshot = true
            isLoading = false  // 首屏已出，停转圈；live 在后台静默升级
        } catch {
            // 快照失败不报错，等 live 兜底
        }

        do {
            let live = try await StockChartService.fetchLive(ticker: ticker, period: period)
            guard generation == loadGeneration else { return }
            applyBars(live.0, source: live.1, stamp: live.2)
            isLoading = false
        } catch {
            guard generation == loadGeneration else { return }
            if !gotSnapshot {
                bars = []; ma5 = []; ma10 = []; ma20 = []; ma60 = []
                dataSource = nil
                cloudUpdatedAt = nil
                errorMessage = friendlyError(error)
            }
            isLoading = false
        }
    }

    private func applyBars(_ fetched: [OHLCVBar], source: ChartDataSource, stamp: String?) {
        bars = fetched
        scrollOffset = 0
        selectedIndex = nil
        dataSource = source
        cloudUpdatedAt = stamp
        lastFetched = Date()
        let closes = fetched.map(\.close)
        ma5 = StockChartService.movingAverage(closes, period: 5)
        ma10 = StockChartService.movingAverage(closes, period: 10)
        ma20 = StockChartService.movingAverage(closes, period: 20)
        ma60 = StockChartService.movingAverage(closes, period: 60)
        errorMessage = nil
    }

    private func friendlyError(_ error: Error) -> String {
        let msg = error.localizedDescription
        if (error as NSError).code == NSURLErrorTimedOut {
            return "云端请求超时 · 请检查网络或 K线 API 地址"
        }
        if (error as NSError).code == NSURLErrorBadServerResponse {
            return "云端暂无该标的 K 线 · 请稍后下拉刷新"
        }
        return msg.isEmpty ? "无法从云端加载 K 线" : msg
    }
}
