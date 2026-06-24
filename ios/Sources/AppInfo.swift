import Foundation

enum AppInfo {
    static var version: String {
        Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "3.0"
    }

    static var build: String {
        Bundle.main.infoDictionary?["CFBundleVersion"] as? String ?? "?"
    }

    static var displayVersion: String {
        "v\(version) (\(build))"
    }

    static var bundledDailyPickExists: Bool {
        Bundle.main.url(forResource: "daily_pick_today", withExtension: "json") != nil
    }
}
