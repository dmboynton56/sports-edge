//
//  SportsEdgeAppApp.swift
//  SportsEdgeApp
//
//  Created by Drew Boynton on 3/13/26.
//

import SwiftUI

@main
struct SportsEdgeAppApp: App {
    @StateObject private var appStore = AppStore()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(appStore)
        }
    }
}
