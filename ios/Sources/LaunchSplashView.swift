import SwiftUI

/// 启动过渡页：红底品牌 + 加载，替代系统白屏。
struct LaunchSplashView: View {
    var message: String = "正在加载策略数据…"

    var body: some View {
        ZStack {
            LinearGradient(
                colors: [
                    ThsTheme.homeHeaderRed,
                    ThsTheme.homeHeaderRed.opacity(0.88),
                    Color(red: 0.55, green: 0.08, blue: 0.10),
                ],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
            .ignoresSafeArea()

            VStack(spacing: 28) {
                Spacer()
                VStack(spacing: 16) {
                    Image("AppLogo")
                        .resizable()
                        .scaledToFit()
                        .frame(width: 88, height: 88)
                        .clipShape(RoundedRectangle(cornerRadius: 20, style: .continuous))
                        .shadow(color: .black.opacity(0.25), radius: 16, y: 8)
                    VStack(spacing: 6) {
                        Text("美股量化")
                            .font(.system(size: 28, weight: .bold))
                            .foregroundStyle(.white)
                        Text("全策略研究 · 个人自用")
                            .font(.subheadline)
                            .foregroundStyle(.white.opacity(0.85))
                    }
                }
                Spacer()
                VStack(spacing: 12) {
                    ProgressView()
                        .tint(.white)
                        .scaleEffect(1.1)
                    Text(message)
                        .font(.footnote)
                        .foregroundStyle(.white.opacity(0.75))
                }
                .padding(.bottom, 48)
            }
        }
    }
}

/// App 根：启动加载 + 注入 Tab 壳。
struct AppRootView: View {
    @EnvironmentObject private var pickLoader: DailyPickLoader
    @EnvironmentObject private var manifestLoader: ManifestLoader
    @State private var showSplash = true

    var body: some View {
        ZStack {
            ContentView()
            if showSplash {
                LaunchSplashView(message: splashMessage)
                    .transition(.opacity)
                    .zIndex(1)
            }
        }
        .task { await bootstrap() }
    }

    private var splashMessage: String {
        if pickLoader.isLoading || manifestLoader.isLoading {
            return "正在同步选股与功能清单…"
        }
        if pickLoader.document != nil {
            return "加载完成"
        }
        return "正在加载策略数据…"
    }

    private func bootstrap() async {
        let start = Date()
        await AppServices.refreshAll()
        let elapsed = Date().timeIntervalSince(start)
        if elapsed < 0.6 {
            try? await Task.sleep(nanoseconds: UInt64((0.6 - elapsed) * 1_000_000_000))
        }
        withAnimation(.easeOut(duration: 0.35)) {
            showSplash = false
        }
    }
}

#Preview {
    LaunchSplashView()
}
