import SwiftUI

struct StrategyCatalogView: View {
    let rows: [StrategyCatalogRow]
    @State private var query = ""
    @State private var filter: CatalogFilter = .all

    enum CatalogFilter: String, CaseIterable {
        case all = "全部"
        case integrated = "已接入"
        case actionable = "有信号"
    }

    private var filtered: [StrategyCatalogRow] {
        rows.filter { row in
            let matchFilter: Bool
            switch filter {
            case .all: matchFilter = true
            case .integrated: matchFilter = row.integrated
            case .actionable: matchFilter = row.actionable > 0 || row.watching > 0
            }
            let q = query.trimmingCharacters(in: .whitespaces)
            let matchQuery = q.isEmpty
                || row.name.localizedCaseInsensitiveContains(q)
                || row.category.localizedCaseInsensitiveContains(q)
                || row.detail.localizedCaseInsensitiveContains(q)
            return matchFilter && matchQuery
        }
    }

    var body: some View {
        VStack(spacing: 0) {
            filterBar
            List(filtered) { row in
                NavigationLink {
                    StrategyRowDestinationView(row: row)
                } label: {
                    StrategyCatalogCard(row: row)
                }
                .listRowInsets(EdgeInsets(top: 6, leading: 16, bottom: 6, trailing: 16))
                .listRowSeparator(.hidden)
                .listRowBackground(Color.clear)
            }
            .listStyle(.plain)
        }
        .scrollContentBackground(.hidden)
        .background(ThsTheme.background)
        .navigationTitle("策略目录")
        .searchable(text: $query, prompt: "搜索策略")
        .preferredColorScheme(.dark)
    }

    private var filterBar: some View {
        Picker("筛选", selection: $filter) {
            ForEach(CatalogFilter.allCases, id: \.self) { f in
                Text(f.rawValue).tag(f)
            }
        }
        .pickerStyle(.segmented)
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
        .background(ThsTheme.card)
    }
}

struct StrategyCatalogCard: View {
    let row: StrategyCatalogRow

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 4) {
                    Text(row.name)
                        .font(.headline)
                        .foregroundStyle(ThsTheme.textPrimary)
                    Text(row.category)
                        .font(.caption)
                        .foregroundStyle(ThsTheme.textSecondary)
                }
                Spacer()
                VStack(alignment: .trailing, spacing: 6) {
                    if row.integrated {
                        tag("已接入", color: ThsTheme.up)
                    }
                    if row.hasData {
                        tag("有数据", color: ThsTheme.accent)
                    }
                }
            }

            HStack(spacing: 16) {
                signalMetric("可开仓", row.actionable, icon: "checkmark.circle.fill", tint: ThsTheme.up)
                signalMetric("观望", row.watching, icon: "eye", tint: .orange)
                signalMetric("合计", row.total, icon: "list.bullet", tint: ThsTheme.textSecondary)
            }

            if !row.detail.isEmpty {
                Text(row.detail)
                    .font(.footnote)
                    .foregroundStyle(ThsTheme.textSecondary)
                    .lineLimit(3)
            }

            if !row.dataDate.isEmpty {
                Text("数据日期 \(row.dataDate)")
                    .font(.caption2)
                    .foregroundStyle(ThsTheme.textTertiary)
            }
        }
        .padding(14)
        .thsCard(
            border: row.actionable > 0 ? ThsTheme.up.opacity(0.3) : ThsTheme.border,
            radius: 14
        )
    }

    private func tag(_ text: String, color: Color) -> some View {
        Text(text)
            .font(.caption2.weight(.bold))
            .foregroundStyle(color)
            .padding(.horizontal, 8)
            .padding(.vertical, 3)
            .background(color.opacity(0.15), in: Capsule())
    }

    private func signalMetric(_ title: String, _ value: Int, icon: String, tint: Color) -> some View {
        HStack(spacing: 4) {
            Image(systemName: icon)
                .font(.caption2)
            Text("\(value)")
                .font(.caption.weight(.bold))
            Text(title)
                .font(.caption2)
        }
        .foregroundStyle(value > 0 ? tint : ThsTheme.textTertiary)
    }
}

#Preview {
    NavigationStack {
        StrategyCatalogView(rows: [])
            .environmentObject(ManifestLoader.shared)
    }
}
