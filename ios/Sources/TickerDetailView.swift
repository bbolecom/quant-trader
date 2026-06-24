import SwiftUI

/// 股票详情：K 线 + MA20/MA50 + 模块数据字段
struct TickerDetailView: View {
    let ticker: String
    let title: String?
    let metadata: [String: Any]
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    StockChartPanel(ticker: ticker)
                    dataFieldsBlock
                }
                .padding(16)
            }
            .background(ThsTheme.background.ignoresSafeArea())
            .navigationTitle(ticker.uppercased())
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("完成") { dismiss() }.fontWeight(.semibold)
                }
            }
        }
        .preferredColorScheme(.dark)
    }

    private var dataFieldsBlock: some View {
        VStack(alignment: .leading, spacing: 10) {
            if let title, !title.isEmpty {
                ThsSectionHeader(title: title, accent: ThsTheme.textSecondary, icon: "list.bullet.rectangle")
            } else {
                ThsSectionHeader(title: "相关数据", accent: ThsTheme.textSecondary, icon: "list.bullet.rectangle")
            }
            VStack(spacing: 0) {
                ForEach(displayFields, id: \.key) { field in
                    HStack(alignment: .top) {
                        Text(field.key)
                            .font(.caption)
                            .foregroundStyle(ThsTheme.textSecondary)
                            .frame(width: 96, alignment: .leading)
                        Text(field.value)
                            .font(.footnote)
                            .foregroundStyle(ThsTheme.textPrimary)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                    .padding(.vertical, 8)
                    if field.key != displayFields.last?.key {
                        Divider().overlay(ThsTheme.border)
                    }
                }
            }
            .padding(.horizontal, 14)
            .thsCard()
        }
    }

    private var displayFields: [(key: String, value: String)] {
        let skip = Set(["代码", "ticker"])
        let priority = [
            "状态", "信号", "方向", "策略动作", "选股理由", "说明", "规则", "规律",
            "涨幅_pct", "涨幅%", "1日涨%", "5日涨跌%", "量比", "成交额M", "5日命中率",
            "阶段", "类型名", "相似分", "现价", "暴涨收盘价", "账户", "模块"
        ]
        var out: [(String, String)] = []
        for k in priority {
            if skip.contains(k) { continue }
            if let v = formatted(metadata[k]) { out.append((k, v)) }
        }
        for k in metadata.keys.sorted() where !priority.contains(k) && !skip.contains(k) {
            if let v = formatted(metadata[k]) { out.append((k, v)) }
        }
        return out
    }

    private func formatted(_ value: Any?) -> String? {
        switch value {
        case let s as String where !s.isEmpty: return s
        case let n as Double:
            if n == n.rounded() && abs(n) < 10_000 { return String(format: "%.0f", n) }
            return String(format: "%.2f", n)
        case let n as Int: return String(n)
        case let b as Bool: return b ? "是" : "否"
        default: return nil
        }
    }
}

extension JsonHelper {
    static func ticker(from row: [String: Any]) -> String? {
        string(row, "代码", "ticker")?.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    static func isValidTicker(_ raw: String?) -> Bool {
        guard let t = raw?.uppercased(), !t.isEmpty, t != "—", t != "-" else { return false }
        return t.range(of: "^[A-Z][A-Z0-9.-]{0,9}$", options: .regularExpression) != nil
    }
}
