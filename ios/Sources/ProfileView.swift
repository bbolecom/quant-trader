import SwiftUI

/// 同花顺「我的」：连接设置 + 关于
struct ProfileView: View {
    @EnvironmentObject private var nav: AppNavigation
    @ObservedObject private var settings = AppSettings.shared
    @StateObject private var pickLoader = DailyPickLoader.shared
    @State private var draftJsonBase = ""
    @State private var draftStreamlit = ""

    var body: some View {
        NavigationStack {
            List {
                Section("服务器") {
                    VStack(alignment: .leading, spacing: 6) {
                        Text("JSON 基址")
                            .font(.caption)
                            .foregroundStyle(ThsTheme.textSecondary)
                        TextField("http://192.168.1.20:8502/", text: $draftJsonBase)
                            .textInputAutocapitalization(.never)
                            .autocorrectionDisabled()
                            .font(.footnote)
                    }
                    VStack(alignment: .leading, spacing: 6) {
                        Text("Streamlit 量化终端")
                            .font(.caption)
                            .foregroundStyle(ThsTheme.textSecondary)
                        TextField("http://host:8501", text: $draftStreamlit)
                            .textInputAutocapitalization(.never)
                            .autocorrectionDisabled()
                            .font(.footnote)
                    }
                    Button("保存并刷新") {
                        settings.jsonBaseURL = draftJsonBase.trimmingCharacters(in: .whitespaces)
                        settings.streamlitURL = draftStreamlit.trimmingCharacters(in: .whitespaces)
                        Task { await pickLoader.reload() }
                    }
                    .foregroundStyle(ThsTheme.accent)
                }

                Section("快捷操作") {
                    Button {
                        Task { await pickLoader.reload() }
                    } label: {
                        Label("刷新今日选股", systemImage: "arrow.clockwise")
                    }
                    Button {
                        nav.openTerminal()
                    } label: {
                        Label("打开量化终端", systemImage: "chart.xyaxis.line")
                    }
                    Button {
                        nav.openPicks()
                    } label: {
                        Label("今日选股 Tab", systemImage: "star.circle")
                    }
                }

                Section("连接状态") {
                    statusRow("JSON", settings.jsonURLHint)
                    if let host = pickLoader.loadedFrom {
                        statusRow("选股来源", host)
                    }
                    if let t = pickLoader.lastUpdated {
                        statusRow("上次刷新", t.formatted(date: .abbreviated, time: .shortened))
                    }
                }

                Section("使用说明") {
                    instruction("Mac 运行 ./run.sh 或 python daily_pick.py")
                    instruction("research 目录 HTTP 8502 供 App 拉 JSON")
                    instruction("手机与 Mac 同一 Wi-Fi，填 Mac 局域网 IP")
                    instruction("全功能清单：app_manifest.json")
                }

                Section("关于") {
                    Text("美股量化 v3.0")
                    Text("同花顺式全策略 App · 仅供个人研究，不构成投资建议。")
                        .font(.caption)
                        .foregroundStyle(ThsTheme.textSecondary)
                }
            }
            .navigationTitle("我的")
            .navigationBarTitleDisplayMode(.large)
            .onAppear {
                draftJsonBase = settings.jsonBaseURL
                draftStreamlit = settings.streamlitURL
            }
        }
        .preferredColorScheme(.dark)
    }

    private func statusRow(_ title: String, _ value: String) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title).font(.caption).foregroundStyle(ThsTheme.textSecondary)
            Text(value).font(.footnote).textSelection(.enabled)
        }
    }

    private func instruction(_ text: String) -> some View {
        Label(text, systemImage: "circle.fill")
            .font(.caption)
            .labelStyle(.titleAndIcon)
            .foregroundStyle(ThsTheme.textSecondary)
    }
}

#Preview {
    ProfileView()
        .environmentObject(AppNavigation())
}
