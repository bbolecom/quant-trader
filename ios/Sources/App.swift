import SwiftUI

@main
struct QuantTraderApp: App {
    init() {
        let accent = UIColor(red: 0.914, green: 0.188, blue: 0.188, alpha: 1)
        UIView.appearance(whenContainedInInstancesOf: [UIRefreshControl.self]).tintColor = accent
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}
