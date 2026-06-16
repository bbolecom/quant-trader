import SwiftUI
import UIKit

/// 与 `tiger_theme.py` 保持一致的老虎证券风格色板。
enum TigerTheme {
    static let orange = Color(red: 1.0, green: 0.412, blue: 0.0)       // #FF6900
    static let orangeHover = Color(red: 1.0, green: 0.522, blue: 0.2)  // #FF8533
    static let background = Color(red: 0.051, green: 0.051, blue: 0.051) // #0D0D0D
    static let card = Color(red: 0.086, green: 0.086, blue: 0.086)     // #161616
    static let elevated = Color(red: 0.122, green: 0.122, blue: 0.122) // #1F1F1F
    static let border = Color(red: 0.165, green: 0.165, blue: 0.165)   // #2A2A2A
    static let textPrimary = Color.white
    static let textSecondary = Color(red: 0.557, green: 0.557, blue: 0.576) // #8E8E93
    static let textTertiary = Color(red: 0.388, green: 0.388, blue: 0.400)  // #636366
    static let up = Color(red: 0.0, green: 0.753, blue: 0.529)         // #00C087
    static let down = Color(red: 1.0, green: 0.271, blue: 0.271)        // #FF4545

    /// 启动页 / 加载页（白底）
    static let launchBackground = Color.white
    static let launchTextPrimary = Color(red: 0.051, green: 0.051, blue: 0.051)
    static let launchTextSecondary = Color(red: 0.388, green: 0.388, blue: 0.400)
}

extension UIColor {
    static let tigerBackground = UIColor(red: 0.051, green: 0.051, blue: 0.051, alpha: 1)
    static let tigerCard = UIColor(red: 0.086, green: 0.086, blue: 0.086, alpha: 1)
    static let launchBackground = UIColor.white
}
