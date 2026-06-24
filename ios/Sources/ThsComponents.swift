import SwiftUI

// MARK: - Card & Layout

struct ThsCardModifier: ViewModifier {
    var border: Color = ThsTheme.border
    var radius: CGFloat = 14

    func body(content: Content) -> some View {
        content
            .background(ThsTheme.card, in: RoundedRectangle(cornerRadius: radius, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: radius, style: .continuous)
                    .stroke(border, lineWidth: 1)
            )
    }
}

extension View {
    func thsCard(border: Color = ThsTheme.border, radius: CGFloat = 14) -> some View {
        modifier(ThsCardModifier(border: border, radius: radius))
    }
}

// MARK: - Section Header

struct ThsSectionHeader: View {
    let title: String
    var subtitle: String?
    var count: Int?
    var accent: Color = ThsTheme.accent
    var icon: String?

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            if let icon {
                Image(systemName: icon)
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(accent)
            }
            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(.headline)
                    .foregroundStyle(ThsTheme.textPrimary)
                if let subtitle {
                    Text(subtitle)
                        .font(.caption)
                        .foregroundStyle(ThsTheme.textSecondary)
                }
            }
            Spacer()
            if let count {
                Text("\(count)")
                    .font(.caption.weight(.bold))
                    .foregroundStyle(accent)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 4)
                    .background(accent.opacity(0.15), in: Capsule())
            }
        }
    }
}

// MARK: - Metric Tile

struct ThsMetricTile: View {
    let title: String
    let value: String
    var accent: Color = ThsTheme.textPrimary
    var icon: String?

    var body: some View {
        VStack(spacing: 6) {
            if let icon {
                Image(systemName: icon)
                    .font(.caption)
                    .foregroundStyle(accent.opacity(0.85))
            }
            Text(value)
                .font(.title3.weight(.bold))
                .foregroundStyle(accent)
                .minimumScaleFactor(0.8)
                .lineLimit(1)
            Text(title)
                .font(.caption2)
                .foregroundStyle(ThsTheme.textSecondary)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 12)
        .thsCard(border: ThsTheme.border.opacity(0.6), radius: 12)
    }
}

// MARK: - Regime Banner

struct RegimeBanner: View {
    let regime: RegimeInfo?
    let pickDate: String?
    let pickTime: String?

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            ZStack {
                Circle()
                    .fill(regimeTint.opacity(0.18))
                    .frame(width: 44, height: 44)
                Image(systemName: regime?.bull == true ? "chart.line.uptrend.xyaxis" : "chart.line.downtrend.xyaxis")
                    .font(.system(size: 18, weight: .semibold))
                    .foregroundStyle(regimeTint)
            }
            VStack(alignment: .leading, spacing: 4) {
                Text(regime?.label ?? "大盘状态未知")
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(regimeTint)
                if let spy = regime?.spy, let ma = regime?.ma50 {
                    Text(String(format: "SPY %.2f · MA50 %.2f", spy, ma))
                        .font(.caption)
                        .foregroundStyle(ThsTheme.textSecondary)
                }
                if let date = pickDate {
                    Text("\(date)\(pickTime.map { " · \($0)" } ?? "")")
                        .font(.caption2)
                        .foregroundStyle(ThsTheme.textTertiary)
                }
            }
            Spacer()
        }
        .padding(14)
        .background(
            LinearGradient(
                colors: [regimeTint.opacity(0.12), ThsTheme.card],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            ),
            in: RoundedRectangle(cornerRadius: 16, style: .continuous)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .stroke(regimeTint.opacity(0.25), lineWidth: 1)
        )
    }

    private var regimeTint: Color {
        regime?.bull == true ? ThsTheme.up : ThsTheme.down
    }
}

// MARK: - Backtest Strip

struct BacktestStrip: View {
    let row: PickRow

    var body: some View {
        HStack(spacing: 0) {
            backtestCell("胜率", value: row.displayWinRate, accent: row.isHighWinQualified ? ThsTheme.up : ThsTheme.textSecondary)
            divider
            backtestCell("年化", value: row.displayAnnReturn, accent: ThsTheme.accent)
            divider
            backtestCell("回撤", value: row.displayMaxDD, accent: ThsTheme.down)
        }
        .padding(.vertical, 8)
        .background(ThsTheme.elevated, in: RoundedRectangle(cornerRadius: 10, style: .continuous))
    }

    private var divider: some View {
        Rectangle()
            .fill(ThsTheme.border)
            .frame(width: 1, height: 28)
    }

    private func backtestCell(_ title: String, value: String, accent: Color) -> some View {
        VStack(spacing: 2) {
            Text(value)
                .font(.caption.weight(.bold))
                .foregroundStyle(accent)
            Text(title)
                .font(.caption2)
                .foregroundStyle(ThsTheme.textTertiary)
        }
        .frame(maxWidth: .infinity)
    }
}

// MARK: - Pick Card

struct PickCardView: View {
    let row: PickRow
    var highlight: Bool = false
    var onTap: (() -> Void)?

    var body: some View {
        Button {
            onTap?()
        } label: {
            VStack(alignment: .leading, spacing: 10) {
                HStack(alignment: .top) {
                    VStack(alignment: .leading, spacing: 4) {
                        HStack(spacing: 8) {
                            Text(row.ticker)
                                .font(.title3.weight(.bold))
                                .foregroundStyle(ThsTheme.textPrimary)
                            if row.isHighWinQualified {
                                Image(systemName: "star.fill")
                                    .font(.caption2)
                                    .foregroundStyle(.yellow)
                            }
                        }
                        Text(row.module)
                            .font(.caption)
                            .foregroundStyle(ThsTheme.textSecondary)
                            .lineLimit(1)
                    }
                    Spacer()
                    VStack(alignment: .trailing, spacing: 4) {
                        DirectionBadge(direction: row.direction, highlight: highlight)
                        StatusBadge(status: row.status)
                    }
                }

                if row.hasBacktestStats {
                    BacktestStrip(row: row)
                } else if let note = row.backtestNote, !note.isEmpty {
                    Text(note)
                        .font(.caption2)
                        .foregroundStyle(ThsTheme.textTertiary)
                }

                let brief = row.displayActionBrief
                if !brief.isEmpty {
                    Text(brief)
                        .font(.caption.weight(.medium))
                        .foregroundStyle(ThsTheme.accent)
                        .lineLimit(2)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(ThsTheme.accent.opacity(0.12), in: RoundedRectangle(cornerRadius: 8))
                }

                Text(row.reason)
                    .font(.footnote)
                    .foregroundStyle(ThsTheme.textSecondary)
                    .lineLimit(highlight ? 4 : 2)
                    .multilineTextAlignment(.leading)
            }
            .padding(14)
            .thsCard(
                border: highlight ? ThsTheme.up.opacity(0.45) : ThsTheme.border,
                radius: 14
            )
        }
        .buttonStyle(.plain)
    }
}

struct DirectionBadge: View {
    let direction: String
    var highlight: Bool

    var body: some View {
        Text(direction)
            .font(.caption2.weight(.bold))
            .foregroundStyle(tint)
            .padding(.horizontal, 8)
            .padding(.vertical, 3)
            .background(tint.opacity(0.15), in: Capsule())
    }

    private var tint: Color {
        if direction.contains("多") || direction.contains("买") { return ThsTheme.up }
        if direction.contains("空") || direction.contains("卖") || direction.contains("回避") { return ThsTheme.down }
        return highlight ? ThsTheme.accent : ThsTheme.textSecondary
    }
}

struct StatusBadge: View {
    let status: String

    var body: some View {
        Text(status)
            .font(.caption2.weight(.semibold))
            .foregroundStyle(tint)
    }

    private var tint: Color {
        status == "可开仓" ? ThsTheme.up : .orange
    }
}

// MARK: - Module Chip

struct ModuleChip: View {
    let name: String
    let actionable: Int
    let watching: Int

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(name)
                .font(.caption.weight(.semibold))
                .foregroundStyle(ThsTheme.textPrimary)
                .lineLimit(2)
                .multilineTextAlignment(.leading)
            HStack(spacing: 8) {
                Label("\(actionable)", systemImage: "checkmark.circle.fill")
                    .foregroundStyle(actionable > 0 ? ThsTheme.up : ThsTheme.textTertiary)
                Label("\(watching)", systemImage: "eye")
                    .foregroundStyle(ThsTheme.textSecondary)
            }
            .font(.caption2)
        }
        .padding(10)
        .frame(width: 140, alignment: .leading)
        .thsCard(
            border: actionable > 0 ? ThsTheme.up.opacity(0.3) : ThsTheme.border,
            radius: 12
        )
    }
}

// MARK: - Loading Skeleton

struct DailyPickSkeleton: View {
    var body: some View {
        VStack(spacing: 16) {
            RoundedRectangle(cornerRadius: 16)
                .fill(ThsTheme.elevated)
                .frame(height: 88)
            HStack(spacing: 10) {
                ForEach(0..<3, id: \.self) { _ in
                    RoundedRectangle(cornerRadius: 12)
                        .fill(ThsTheme.elevated)
                        .frame(height: 72)
                }
            }
            ForEach(0..<3, id: \.self) { _ in
                RoundedRectangle(cornerRadius: 14)
                    .fill(ThsTheme.elevated)
                    .frame(height: 120)
            }
        }
        .shimmer()
        .padding()
    }
}

private struct ShimmerModifier: ViewModifier {
    @State private var phase: CGFloat = 0

    func body(content: Content) -> some View {
        content
            .overlay {
                LinearGradient(
                    colors: [.clear, ThsTheme.textPrimary.opacity(0.06), .clear],
                    startPoint: .leading,
                    endPoint: .trailing
                )
                .offset(x: phase)
                .onAppear {
                    withAnimation(.linear(duration: 1.2).repeatForever(autoreverses: false)) {
                        phase = 280
                    }
                }
            }
            .clipped()
    }
}

extension View {
    func shimmer() -> some View {
        modifier(ShimmerModifier())
    }
}

// MARK: - Data Source Banner

struct ThsDataSourceBanner: View {
    let source: DailyPickDataSource
    var hint: String?
    var onSetup: (() -> Void)?

    var body: some View {
        HStack(spacing: 10) {
            Image(systemName: icon)
                .font(.subheadline.weight(.semibold))
                .foregroundStyle(tint)
            VStack(alignment: .leading, spacing: 2) {
                Text("数据来源 · \(source.label)")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(ThsTheme.textPrimary)
                Text(subtitle)
                    .font(.caption2)
                    .foregroundStyle(ThsTheme.textSecondary)
            }
            Spacer()
            if let onSetup, source != .remote {
                Button("连接 Mac", action: onSetup)
                    .font(.caption.weight(.bold))
                    .foregroundStyle(ThsTheme.accent)
            }
        }
        .padding(12)
        .thsCard(border: tint.opacity(0.35), radius: 12)
    }

    private var icon: String {
        switch source {
        case .remote: return "antenna.radiowaves.left.and.right"
        case .github: return "cloud.fill"
        case .bundled: return "doc.text.fill"
        }
    }

    private var tint: Color {
        switch source {
        case .remote: return ThsTheme.up
        case .github: return .cyan
        case .bundled: return .orange
        }
    }

    private var subtitle: String {
        if let hint, !hint.isEmpty { return hint }
        switch source {
        case .remote: return "Mac 局域网实时 JSON"
        case .github: return "GitHub 云端 · 自动同步"
        case .bundled: return "App 内置 · 可能不是今日最新"
        }
    }
}

// MARK: - Empty / Setup State

struct ThsEmptyState: View {
    let icon: String
    let title: String
    let message: String
    var primaryTitle: String = "重新加载"
    var primaryAction: () -> Void
    var secondaryTitle: String?
    var secondaryAction: (() -> Void)?

    var body: some View {
        VStack(spacing: 24) {
            Spacer(minLength: 20)
            ZStack {
                Circle()
                    .fill(ThsTheme.accent.opacity(0.12))
                    .frame(width: 96, height: 96)
                Image(systemName: icon)
                    .font(.system(size: 40))
                    .foregroundStyle(ThsTheme.accent)
            }
            VStack(spacing: 8) {
                Text(title)
                    .font(.title3.weight(.bold))
                    .foregroundStyle(ThsTheme.textPrimary)
                Text(message)
                    .font(.footnote)
                    .multilineTextAlignment(.center)
                    .foregroundStyle(ThsTheme.textSecondary)
                    .padding(.horizontal, 8)
            }
            VStack(spacing: 10) {
                Button(action: primaryAction) {
                    Label(primaryTitle, systemImage: "arrow.clockwise")
                        .font(.subheadline.weight(.semibold))
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 14)
                }
                .buttonStyle(.borderedProminent)
                .tint(ThsTheme.accent)
                if let secondaryTitle, let secondaryAction {
                    Button(action: secondaryAction) {
                        Text(secondaryTitle)
                            .font(.subheadline.weight(.medium))
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 12)
                    }
                    .buttonStyle(.bordered)
                    .tint(ThsTheme.textSecondary)
                }
            }
            .padding(.horizontal, 32)
            Spacer(minLength: 20)
        }
        .padding(.horizontal, 20)
    }
}
