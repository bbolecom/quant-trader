import SwiftUI

struct SettingsView: View {
    @ObservedObject var loader: DailyPickLoader
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            List {
                Section("连接") {
                    settingsRow("Streamlit", value: AppConfig.serverURLString, multiline: true)
                    settingsRow("选股 JSON", value: jsonHint, multiline: true)
                    if let host = loader.loadedFrom {
                        settingsRow("上次来源", value: host)
                    }
                    if let updated = loader.lastUpdated {
                        settingsRow("上次刷新", value: updated.formatted(date: .abbreviated, time: .shortened))
                    }
                }

                Section("使用说明") {
                    Label("Mac 运行 python daily_pick.py", systemImage: "1.circle")
                    Label("research 目录启动 8502 静态服务", systemImage: "2.circle")
                    Label("手机与 Mac 同一 Wi-Fi", systemImage: "3.circle")
                    Label("修改 Config.swift 中的地址后重新编译", systemImage: "4.circle")
                }
                .font(.footnote)

                Section {
                    Button {
                        Task { await loader.reload() }
                    } label: {
                        Label("立即刷新选股", systemImage: "arrow.clockwise")
                    }
                }

                Section {
                    Text("版本 3.0 · 在「我的」Tab 配置服务器 · 仅供个人研究。")
                        .font(.caption2)
                        .foregroundStyle(ThsTheme.textTertiary)
                }
            }
            .navigationTitle("设置")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("完成") { dismiss() }
                }
            }
        }
        .preferredColorScheme(.dark)
    }

    private func settingsRow(_ title: String, value: String, multiline: Bool = false) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title)
                .font(.subheadline)
            Text(value)
                .font(.caption)
                .foregroundStyle(ThsTheme.textSecondary)
                .multilineTextAlignment(.leading)
                .fixedSize(horizontal: false, vertical: multiline)
        }
        .padding(.vertical, 2)
    }

    private var jsonHint: String {
        if !AppConfig.dailyPickJSONURLString.isEmpty {
            return AppConfig.dailyPickJSONURLString
        }
        return "自动推导 8502/daily_pick_today.json"
    }
}

#Preview {
    SettingsView(loader: DailyPickLoader())
}
