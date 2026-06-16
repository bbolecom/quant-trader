import SwiftUI
import WebKit

/// 把 WKWebView 封装成 SwiftUI 视图，处理加载状态、错误与下拉刷新。
struct WebView: UIViewRepresentable {
    let url: URL
    @Binding var isLoading: Bool
    @Binding var didError: Bool
    /// 每次自增即触发重新加载（用于「重试」按钮）。
    @Binding var reloadToken: Int

    func makeCoordinator() -> Coordinator { Coordinator(self) }

    func makeUIView(context: Context) -> WKWebView {
        let configuration = WKWebViewConfiguration()
        configuration.websiteDataStore = .default()
        configuration.allowsInlineMediaPlayback = true

        let webView = WKWebView(frame: .zero, configuration: configuration)
        webView.navigationDelegate = context.coordinator
        webView.allowsBackForwardNavigationGestures = true
        webView.scrollView.contentInsetAdjustmentBehavior = .never
        webView.isOpaque = false
        webView.backgroundColor = .launchBackground
        webView.scrollView.backgroundColor = .launchBackground
        webView.underPageBackgroundColor = .launchBackground

        let refresh = UIRefreshControl()
        refresh.addTarget(context.coordinator,
                          action: #selector(Coordinator.handleRefresh(_:)),
                          for: .valueChanged)
        webView.scrollView.refreshControl = refresh

        context.coordinator.webView = webView
        webView.load(URLRequest(url: url))
        return webView
    }

    func updateUIView(_ webView: WKWebView, context: Context) {
        if context.coordinator.lastReloadToken != reloadToken {
            context.coordinator.lastReloadToken = reloadToken
            webView.load(URLRequest(url: url))
        }
    }

    final class Coordinator: NSObject, WKNavigationDelegate {
        var parent: WebView
        weak var webView: WKWebView?
        var lastReloadToken = 0

        init(_ parent: WebView) { self.parent = parent }

        @objc func handleRefresh(_ sender: UIRefreshControl) {
            webView?.reload()
        }

        func webView(_ webView: WKWebView, didStartProvisionalNavigation navigation: WKNavigation!) {
            parent.isLoading = true
            parent.didError = false
        }

        func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
            parent.isLoading = false
            webView.scrollView.refreshControl?.endRefreshing()
        }

        func webView(_ webView: WKWebView, didFail navigation: WKNavigation!, withError error: Error) {
            finishWithError(webView)
        }

        func webView(_ webView: WKWebView, didFailProvisionalNavigation navigation: WKNavigation!, withError error: Error) {
            finishWithError(webView)
        }

        private func finishWithError(_ webView: WKWebView) {
            parent.isLoading = false
            parent.didError = true
            webView.scrollView.refreshControl?.endRefreshing()
        }
    }
}
