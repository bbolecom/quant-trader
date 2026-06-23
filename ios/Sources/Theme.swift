import SwiftUI

/// 与 `ths_theme.py` 保持一致的同花顺风格色板。
enum ThsTheme {
    static let accent = Color(red: 0.914, green: 0.188, blue: 0.188)       // #E93030
    static let accentHover = Color(red: 1.0, green: 0.267, blue: 0.267)  // #FF4444
    static let background = Color(red: 0.059, green: 0.071, blue: 0.098) // #0F1219
    static let card = Color(red: 0.102, green: 0.118, blue: 0.157)       // #1A1E28
    static let elevated = Color(red: 0.137, green: 0.157, blue: 0.200)  // #232833
    static let border = Color(red: 0.180, green: 0.200, blue: 0.251)     // #2E3340
    static let textPrimary = Color(red: 0.941, green: 0.945, blue: 0.961) // #F0F1F5
    static let textSecondary = Color(red: 0.545, green: 0.569, blue: 0.620) // #8B919E
    static let textTertiary = Color(red: 0.388, green: 0.408, blue: 0.471)  // #636878
    static let up = Color(red: 0.914, green: 0.188, blue: 0.188)         // #E93030 红涨
    static let down = Color(red: 0.0, green: 0.659, blue: 0.329)         // #00A854 绿跌

    static let heroGradient = LinearGradient(
        colors: [
            Color(red: 0.12, green: 0.08, blue: 0.10),
            background,
        ],
        startPoint: .top,
        endPoint: .bottom
    )

    /// 启动页（白底 + 红色 Logo）
    static let launchBackground = Color.white
    static let launchTextPrimary = Color(red: 0.059, green: 0.071, blue: 0.098)
    static let launchTextSecondary = Color(red: 0.388, green: 0.408, blue: 0.471)

    /// 首页浅色区（同花顺白底）
    static let homeBackground = Color(red: 0.965, green: 0.969, blue: 0.976) // #F6F7F9
    static let homeCard = Color.white
    static let homeTextPrimary = Color(red: 0.133, green: 0.133, blue: 0.133)
    static let homeTextSecondary = Color(red: 0.467, green: 0.467, blue: 0.467)
    static let homeDivider = Color(red: 0.922, green: 0.922, blue: 0.922)
    static let homeHeaderRed = Color(red: 0.914, green: 0.188, blue: 0.188)
}

/// 兼容旧命名
typealias TigerTheme = ThsTheme

extension UIColor {
    static let thsBackground = UIColor(red: 0.059, green: 0.071, blue: 0.098, alpha: 1)
    static let thsCard = UIColor(red: 0.102, green: 0.118, blue: 0.157, alpha: 1)
    static let launchBackground = UIColor.white
    static let tigerBackground = thsBackground
    static let tigerCard = thsCard
}

extension Color {
    init?(hex: String) {
        var s = hex.trimmingCharacters(in: .whitespacesAndNewlines).uppercased()
        if s.hasPrefix("#") { s.removeFirst() }
        guard s.count == 6, let v = UInt64(s, radix: 16) else { return nil }
        self.init(
            red: Double((v >> 16) & 0xFF) / 255,
            green: Double((v >> 8) & 0xFF) / 255,
            blue: Double(v & 0xFF) / 255
        )
    }
}
