import SwiftUI

struct PickDetailView: View {
    let row: PickRow
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    headerBlock
                    if row.hasBacktestStats {
                        backtestBlock
                    }
                    actionBlock
                    reasonBlock
                    metaBlock
                }
                .padding()
            }
            .background(ThsTheme.background.ignoresSafeArea())
            .navigationTitle(row.ticker)
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("完成") { dismiss() }
                        .fontWeight(.semibold)
                }
            }
        }
    }

    private var headerBlock: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text(row.ticker)
                    .font(.largeTitle.weight(.bold))
                    .foregroundStyle(ThsTheme.textPrimary)
                Spacer()
                if row.isHighWinQualified {
                    Label("≥80%", systemImage: "star.fill")
                        .font(.caption.weight(.bold))
                        .foregroundStyle(.yellow)
                        .padding(.horizontal, 10)
                        .padding(.vertical, 5)
                        .background(Color.yellow.opacity(0.15), in: Capsule())
                }
            }
            Text(row.module)
                .font(.subheadline)
                .foregroundStyle(ThsTheme.textSecondary)
            HStack(spacing: 10) {
                DirectionBadge(direction: row.direction, highlight: row.isActionable)
                StatusBadge(status: row.status)
            if let acct = row.account, !acct.isEmpty, acct != "—" {
                    Text(acct)
                        .font(.caption)
                        .foregroundStyle(ThsTheme.textTertiary)
                }
            }
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .thsCard(border: row.isHighWinQualified ? ThsTheme.up.opacity(0.35) : ThsTheme.border)
    }

    private var backtestBlock: some View {
        VStack(alignment: .leading, spacing: 10) {
            ThsSectionHeader(title: "历史回测", accent: ThsTheme.accent, icon: "chart.bar.doc.horizontal")
            BacktestStrip(row: row)
            if let note = row.backtestNote {
                Text(note)
                    .font(.footnote)
                    .foregroundStyle(ThsTheme.textSecondary)
            }
            if let src = row.backtestSource {
                Text("来源：\(src)")
                    .font(.caption2)
                    .foregroundStyle(ThsTheme.textTertiary)
            }
        }
    }

    private var actionBlock: some View {
        Group {
            let action = row.displayAction
            if !action.isEmpty {
                VStack(alignment: .leading, spacing: 8) {
                    ThsSectionHeader(title: "策略动作", accent: ThsTheme.accent, icon: "bolt.fill")
                    Text(action)
                        .font(.body.weight(.medium))
                        .foregroundStyle(ThsTheme.textPrimary)
                        .lineSpacing(4)
                        .fixedSize(horizontal: false, vertical: true)
                        .padding(14)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .thsCard(border: ThsTheme.accent.opacity(0.25))
                }
            }
        }
    }

    private var reasonBlock: some View {
        VStack(alignment: .leading, spacing: 8) {
            ThsSectionHeader(title: "选股理由", accent: ThsTheme.textSecondary, icon: "text.alignleft")
            Text(row.reason)
                .font(.body)
                .foregroundStyle(ThsTheme.textPrimary)
                .padding(14)
                .frame(maxWidth: .infinity, alignment: .leading)
                .thsCard()
        }
    }

    private var metaBlock: some View {
        VStack(alignment: .leading, spacing: 6) {
            if let hit = row.hitRate, !hit.isEmpty {
                metaRow("历史命中率", hit)
            }
            metaRow("状态", row.status)
            metaRow("方向", row.direction)
        }
        .font(.caption)
        .foregroundStyle(ThsTheme.textTertiary)
        .padding(.top, 8)
    }

    private func metaRow(_ key: String, _ value: String) -> some View {
        HStack {
            Text(key)
            Spacer()
            Text(value)
                .foregroundStyle(ThsTheme.textSecondary)
        }
    }
}

#Preview {
    PickDetailView(row: PickRow(
        module: "5×舰队·CSP",
        account: "AMPX",
        ticker: "AMPX",
        status: "可开仓",
        direction: "卖Put",
        action: "卖Put 30D · Delta 0.25",
        hitRate: "96%",
        reason: "舰队圣杯账户 · 动量过滤通过 · 真实期权链可用",
        histWin: 0.966,
        histAnn: 0.567,
        histDD: -0.052,
        backtestNote: "胜率97% · 年化56.7% · 回撤-5.2%",
        backtestSource: "screen_fleet_stats.json",
        highWinQualified: true
    ))
}
