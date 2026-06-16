import SwiftUI

struct ContentView: View {
    @State private var isLoading = true
    @State private var didError = false
    @State private var reloadToken = 0

    var body: some View {
        ZStack {
            (isLoading && !didError ? TigerTheme.launchBackground : TigerTheme.background)
                .ignoresSafeArea()

            WebView(url: AppConfig.serverURL,
                    isLoading: $isLoading,
                    didError: $didError,
                    reloadToken: $reloadToken)
                .ignoresSafeArea(edges: .bottom)

            if isLoading && !didError {
                loadingOverlay
            }

            if didError {
                errorOverlay
            }
        }
        .tint(TigerTheme.orange)
        .preferredColorScheme(isLoading && !didError ? .light : .dark)
        .statusBarHidden(false)
    }

    private var loadingOverlay: some View {
        VStack(spacing: 18) {
            Image("AppLogo")
                .resizable()
                .scaledToFit()
                .frame(width: 56, height: 56)
                .clipShape(RoundedRectangle(cornerRadius: 14))

            ProgressView()
                .tint(TigerTheme.orange)
                .scaleEffect(1.1)

            Text("量化策略")
                .font(.headline.weight(.semibold))
                .foregroundStyle(TigerTheme.launchTextPrimary)

            Text("加载中…")
                .font(.footnote)
                .foregroundStyle(TigerTheme.launchTextSecondary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(TigerTheme.launchBackground)
    }

    private var errorOverlay: some View {
        VStack(spacing: 16) {
            Image(systemName: "wifi.exclamationmark")
                .font(.system(size: 42))
                .foregroundStyle(TigerTheme.orange)

            Text("无法连接到服务")
                .font(.headline.weight(.semibold))
                .foregroundStyle(TigerTheme.textPrimary)

            Text("请确认服务已部署/启动，且网络可访问：\n\(AppConfig.serverURLString)")
                .font(.footnote)
                .multilineTextAlignment(.center)
                .foregroundStyle(TigerTheme.textSecondary)

            Button {
                didError = false
                isLoading = true
                reloadToken += 1
            } label: {
                Label("重试", systemImage: "arrow.clockwise")
                    .font(.subheadline.weight(.semibold))
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .tint(TigerTheme.orange)
        }
        .padding(28)
        .background(TigerTheme.card, in: RoundedRectangle(cornerRadius: 16))
        .overlay(
            RoundedRectangle(cornerRadius: 16)
                .stroke(TigerTheme.border, lineWidth: 1)
        )
        .padding(.horizontal, 24)
    }
}

#Preview {
    ContentView()
}
