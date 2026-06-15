import SwiftUI

struct ContentView: View {
    @State private var isLoading = true
    @State private var didError = false
    @State private var reloadToken = 0

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            WebView(url: AppConfig.serverURL,
                    isLoading: $isLoading,
                    didError: $didError,
                    reloadToken: $reloadToken)
                .ignoresSafeArea(edges: .bottom)

            if isLoading && !didError {
                ProgressView("加载中…")
                    .padding(20)
                    .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 14))
            }

            if didError {
                VStack(spacing: 16) {
                    Image(systemName: "wifi.exclamationmark")
                        .font(.system(size: 46))
                        .foregroundStyle(.orange)
                    Text("无法连接到服务")
                        .font(.headline)
                    Text("请确认服务已部署/启动，且网络可访问：\n\(AppConfig.serverURLString)")
                        .font(.footnote)
                        .multilineTextAlignment(.center)
                        .foregroundStyle(.secondary)
                    Button {
                        didError = false
                        isLoading = true
                        reloadToken += 1
                    } label: {
                        Label("重试", systemImage: "arrow.clockwise")
                            .padding(.horizontal, 8)
                    }
                    .buttonStyle(.borderedProminent)
                }
                .padding(32)
            }
        }
        .preferredColorScheme(.dark)
        .statusBarHidden(false)
    }
}

#Preview {
    ContentView()
}
