import SwiftUI

struct ContentView: View {
    @StateObject private var nav = AppNavigation()
    @State private var isLoading = true
    @State private var didError = false
    @State private var reloadToken = 0

    var body: some View {
        TabView(selection: $nav.selectedTab) {
            HomeHubView()
                .tabItem { Label("首页", systemImage: "house.fill") }
                .tag(0)

            FeatureHubView()
                .tabItem { Label("功能", systemImage: "square.grid.2x2.fill") }
                .tag(1)

            DailyPickView()
                .tabItem { Label("选股", systemImage: "star.circle.fill") }
                .tag(2)

            terminalTab
                .tabItem { Label("终端", systemImage: "chart.xyaxis.line") }
                .tag(3)

            ProfileView()
                .tabItem { Label("我的", systemImage: "person.fill") }
                .tag(4)
        }
        .environmentObject(nav)
        .tint(ThsTheme.accent)
        .preferredColorScheme(.dark)
        .onAppear { configureTabBar() }
    }

    private func configureTabBar() {
        let appearance = UITabBarAppearance()
        appearance.configureWithOpaqueBackground()
        appearance.backgroundColor = UIColor.thsCard
        UITabBar.appearance().standardAppearance = appearance
        UITabBar.appearance().scrollEdgeAppearance = appearance
    }

    private var terminalTab: some View {
        NavigationStack {
            ZStack {
                ThsTheme.background.ignoresSafeArea()
                WebView(url: AppSettings.shared.serverURL,
                        isLoading: $isLoading,
                        didError: $didError,
                        reloadToken: $reloadToken)
                    .ignoresSafeArea(edges: .bottom)
                if isLoading && !didError { loadingOverlay }
                if didError { errorOverlay }
            }
            .navigationTitle("量化终端")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button {
                        didError = false
                        isLoading = true
                        reloadToken += 1
                    } label: {
                        Image(systemName: "arrow.clockwise")
                    }
                }
            }
        }
        .preferredColorScheme(isLoading && !didError ? .light : .dark)
    }

    private var loadingOverlay: some View {
        VStack(spacing: 20) {
            Image("AppLogo")
                .resizable()
                .scaledToFit()
                .frame(width: 72, height: 72)
                .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
            ProgressView().tint(ThsTheme.accent).scaleEffect(1.2)
            Text("量化策略终端").font(.headline.weight(.bold))
            Text("加载 Streamlit…").font(.footnote).foregroundStyle(ThsTheme.launchTextSecondary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(ThsTheme.launchBackground)
    }

    private var errorOverlay: some View {
        VStack(spacing: 16) {
            Image(systemName: "wifi.exclamationmark")
                .font(.system(size: 44))
                .foregroundStyle(ThsTheme.accent)
            Text("无法连接到服务").font(.headline.weight(.semibold))
            Text(AppSettings.shared.streamlitURL)
                .font(.caption)
                .foregroundStyle(ThsTheme.textTertiary)
                .multilineTextAlignment(.center)
                .textSelection(.enabled)
            Button {
                didError = false
                isLoading = true
                reloadToken += 1
            } label: {
                Label("重试", systemImage: "arrow.clockwise")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .tint(ThsTheme.accent)
        }
        .padding(28)
        .thsCard()
        .padding(.horizontal, 24)
    }
}

#Preview {
    ContentView()
}
