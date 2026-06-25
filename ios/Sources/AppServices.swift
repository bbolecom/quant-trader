import Foundation

/// 全局数据服务协调：Manifest + DailyPick 统一刷新入口。
@MainActor
enum AppServices {
    static var manifest: ManifestLoader { ManifestLoader.shared }
    static var dailyPick: DailyPickLoader { DailyPickLoader.shared }
    static var marketScan: MarketScanLoader { MarketScanLoader.shared }

    static func refreshAll() async {
        async let pickTask: Void = dailyPick.reload()
        async let manifestTask: Void = manifest.reload()
        async let scanTask: Void = marketScan.reload()
        _ = await (pickTask, manifestTask, scanTask)
    }
}
