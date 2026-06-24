import SwiftUI

/// 功能 → 目标页：统一路由，避免 Home / 功能 / 策略目录三处重复 if-else。
struct FeatureDestinationView: View {
    let feature: ManifestFeature
    @EnvironmentObject private var nav: AppNavigation

    var body: some View {
        Group {
            if feature.id == "daily_pick" {
                DailyPickView(embedded: true)
            } else if feature.isTerminalOnly {
                TerminalFeatureView(feature: feature)
            } else {
                ModuleDetailView(feature: feature)
            }
        }
    }
}

/// 策略目录行 → 目标页（优先 manifest 元数据，避免硬编码 json 路径）。
struct StrategyRowDestinationView: View {
    let row: StrategyCatalogRow
    @EnvironmentObject private var manifestLoader: ManifestLoader

    var body: some View {
        FeatureDestinationView(feature: row.resolvedFeature(manifest: manifestLoader.manifest))
    }
}

/// terminal_only 功能：提示跳转 Streamlit 终端 Tab。
struct TerminalFeatureView: View {
    let feature: ManifestFeature
    @EnvironmentObject private var nav: AppNavigation

    var body: some View {
        VStack(spacing: 20) {
            Image(systemName: feature.icon)
                .font(.system(size: 48))
                .foregroundStyle(ThsTheme.accent)
            Text(feature.name)
                .font(.title2.weight(.bold))
                .foregroundStyle(ThsTheme.textPrimary)
            if !feature.detail.isEmpty {
                Text(feature.detail)
                    .font(.footnote)
                    .foregroundStyle(ThsTheme.textSecondary)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal, 24)
            }
            Text("此功能在 Streamlit 量化终端中运行。")
                .font(.caption)
                .foregroundStyle(ThsTheme.textTertiary)
            Button {
                nav.openTerminal()
            } label: {
                Label("打开量化终端", systemImage: "chart.xyaxis.line")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .tint(ThsTheme.accent)
            .padding(.horizontal, 32)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(ThsTheme.background.ignoresSafeArea())
        .navigationTitle(feature.name)
        .navigationBarTitleDisplayMode(.inline)
        .preferredColorScheme(.dark)
    }
}
