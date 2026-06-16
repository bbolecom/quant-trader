import SwiftUI

@main
struct QuantTraderApp: App {
    init() {
        let orange = UIColor(red: 1.0, green: 0.412, blue: 0.0, alpha: 1)
        UIView.appearance(whenContainedInInstancesOf: [UIRefreshControl.self]).tintColor = orange
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}
