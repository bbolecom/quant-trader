import Foundation

/// 全局配置：把你部署后的服务地址填到这里。
///
/// 两种来源任选其一：
///   1. Streamlit Community Cloud（推荐）：形如 https://your-app.streamlit.app
///   2. 自己电脑/局域网自托管：形如 http://192.168.1.20:8501
///      （自托管走 http 时，需保留 Info.plist 中的 ATS 例外，且手机与电脑在同一 Wi-Fi）
enum AppConfig {
    /// ⬇️ 改成你自己的地址
    static let serverURLString = "https://quant-trader-fd3mch56aixtttm5rgyc6i.streamlit.app"

    static var serverURL: URL {
        URL(string: serverURLString) ?? URL(string: "https://streamlit.io")!
    }
}
