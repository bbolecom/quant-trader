import SwiftUI

/// 同花顺「我的」：连接设置 + 关于
struct ProfileView: View {
    @EnvironmentObject private var nav: AppNavigation
    @EnvironmentObject private var manifestLoader: ManifestLoader
    @ObservedObject private var settings = AppSettings.shared
    @EnvironmentObject private var pickLoader: DailyPickLoader
    @State private var draftJsonBase = ""
    @State private var draftStreamlit = ""
    @State private var draftChartAPI = ""
    @State private var savedToast = false

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 16) {
                    connectionStatusCard
                    serverFormCard
                    presetCard
                    shortcutsCard
                    aboutCard
                }
                .padding(16)
                .padding(.bottom, 24)
            }
            .background(ThsTheme.background.ignoresSafeArea())
            .navigationTitle("我的")
            .navigationBarTitleDisplayMode(.large)
            .onAppear {
                draftJsonBase = settings.jsonBaseURL
                draftStreamlit = settings.streamlitURL
                draftChartAPI = settings.chartAPIURL
            }
            .overlay(alignment: .top) {
                if savedToast {
                    Text("已保存并刷新")
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(.white)
                        .padding(.horizontal, 16)
                        .padding(.vertical, 8)
                        .background(ThsTheme.up, in: Capsule())
                        .padding(.top, 8)
                        .transition(.move(edge: .top).combined(with: .opacity))
                }
            }
        }
        .preferredColorScheme(.dark)
    }

    private var connectionStatusCard: some View {
        VStack(alignment: .leading, spacing: 14) {
            Label("连接状态", systemImage: "link.circle.fill")
                .font(.headline)
                .foregroundStyle(ThsTheme.textPrimary)
            statusLine(
                title: "选股数据",
                value: pickLoader.dataSource?.label ?? "未加载",
                ok: pickLoader.document != nil,
                detail: pickLoader.loadedFrom ?? "—"
            )
            statusLine(
                title: "功能清单",
                value: manifestLoader.manifest != nil ? "已加载" : "未加载",
                ok: manifestLoader.manifest != nil,
                detail: manifestLoader.loadedFrom ?? "—"
            )
            statusLine(
                title: "K 线 API",
                value: settings.chartAPIURL.isEmpty ? "默认 Render" : "已配置",
                ok: true,
                detail: settings.chartAPIHint
            )
            statusLine(
                title: "量化终端",
                value: "Streamlit",
                ok: true,
                detail: settings.streamlitURL
            )
        }
        .padding(16)
        .thsCard()
    }

    private func statusLine(title: String, value: String, ok: Bool, detail: String) -> some View {
        HStack(alignment: .top, spacing: 12) {
            Circle()
                .fill(ok ? ThsTheme.up : .orange)
                .frame(width: 8, height: 8)
                .padding(.top, 6)
            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text(title)
                        .font(.subheadline.weight(.semibold))
                        .foregroundStyle(ThsTheme.textPrimary)
                    Spacer()
                    Text(value)
                        .font(.caption.weight(.bold))
                        .foregroundStyle(ok ? ThsTheme.up : .orange)
                }
                Text(detail)
                    .font(.caption2)
                    .foregroundStyle(ThsTheme.textTertiary)
                    .textSelection(.enabled)
                    .lineLimit(2)
            }
        }
    }

    private var serverFormCard: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("服务器地址")
                .font(.headline)
                .foregroundStyle(ThsTheme.textPrimary)
            fieldBlock(title: "JSON 基址（Mac 局域网，可选）", placeholder: "留空则走 GitHub 云端", text: $draftJsonBase)
            fieldBlock(title: "K 线实时 API", placeholder: AppConfig.defaultChartAPIBase, text: $draftChartAPI)
            fieldBlock(title: "Streamlit 量化终端", placeholder: AppConfig.defaultServerURLString, text: $draftStreamlit)
            Button {
                saveAndRefresh()
            } label: {
                Text("保存并刷新")
                    .font(.subheadline.weight(.bold))
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 14)
            }
            .buttonStyle(.borderedProminent)
            .tint(ThsTheme.accent)
        }
        .padding(16)
        .thsCard()
    }

    private func fieldBlock(title: String, placeholder: String, text: Binding<String>) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title)
                .font(.caption)
                .foregroundStyle(ThsTheme.textSecondary)
            TextField(placeholder, text: text)
                .textInputAutocapitalization(.never)
                .autocorrectionDisabled()
                .font(.footnote)
                .padding(12)
                .background(ThsTheme.elevated, in: RoundedRectangle(cornerRadius: 10))
        }
    }

    private var presetCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("快速配置")
                .font(.headline)
                .foregroundStyle(ThsTheme.textPrimary)
            presetButton("☁️ 云端（默认）", subtitle: "jsDelivr CDN + GitHub 备用 · Render K线") {
                draftStreamlit = AppConfig.defaultServerURLString
                draftJsonBase = ""
                draftChartAPI = AppConfig.defaultChartAPIBase
                saveAndRefresh()
            }
            presetButton("🏠 局域网 Mac", subtitle: "8501 终端 + 8502 JSON + 8503 K线") {
                draftStreamlit = "http://192.168.1.20:8501"
                draftJsonBase = "http://192.168.1.20:8502/"
                draftChartAPI = "http://192.168.1.20:8503"
            }
        }
        .padding(16)
        .thsCard()
    }

    private func presetButton(_ title: String, subtitle: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text(title)
                        .font(.subheadline.weight(.semibold))
                        .foregroundStyle(ThsTheme.textPrimary)
                    Text(subtitle)
                        .font(.caption2)
                        .foregroundStyle(ThsTheme.textSecondary)
                }
                Spacer()
                Image(systemName: "chevron.right")
                    .font(.caption.weight(.bold))
                    .foregroundStyle(ThsTheme.textTertiary)
            }
            .padding(12)
            .background(ThsTheme.elevated, in: RoundedRectangle(cornerRadius: 12))
        }
        .buttonStyle(.plain)
    }

    private var shortcutsCard: some View {
        VStack(spacing: 10) {
            shortcutRow("刷新全部数据", icon: "arrow.clockwise") {
                Task { await AppServices.refreshAll() }
            }
            shortcutRow("打开量化终端", icon: "chart.xyaxis.line") {
                nav.openTerminal()
            }
            shortcutRow("今日选股", icon: "star.circle") {
                nav.openPicks()
            }
        }
        .padding(16)
        .thsCard()
    }

    private func shortcutRow(_ title: String, icon: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            HStack {
                Label(title, systemImage: icon)
                    .font(.subheadline)
                    .foregroundStyle(ThsTheme.textPrimary)
                Spacer()
                Image(systemName: "chevron.right")
                    .font(.caption2.weight(.bold))
                    .foregroundStyle(ThsTheme.textTertiary)
            }
            .padding(.vertical, 4)
        }
        .buttonStyle(.plain)
    }

    private var aboutCard: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("美股量化 v3.0")
                .font(.subheadline.weight(.semibold))
            Text("所有模块数据与 K 线均从云端拉取，全市场快扫每 5 分钟自动刷新。")
                .font(.caption)
                .foregroundStyle(ThsTheme.textSecondary)
            Text("模块 JSON：GitHub raw · K 线：Render 实时 API（可在上方修改）。")
                .font(.caption2)
                .foregroundStyle(ThsTheme.textTertiary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(16)
        .thsCard(border: ThsTheme.border.opacity(0.5))
    }

    private func saveAndRefresh() {
        settings.jsonBaseURL = draftJsonBase.trimmingCharacters(in: .whitespaces)
        settings.streamlitURL = draftStreamlit.trimmingCharacters(in: .whitespaces)
        settings.chartAPIURL = draftChartAPI.trimmingCharacters(in: .whitespaces)
        Task {
            await AppServices.refreshAll()
            withAnimation { savedToast = true }
            try? await Task.sleep(nanoseconds: 1_500_000_000)
            withAnimation { savedToast = false }
        }
    }
}

#Preview {
    ProfileView()
        .environmentObject(AppNavigation())
        .environmentObject(ManifestLoader.shared)
        .environmentObject(DailyPickLoader.shared)
}
