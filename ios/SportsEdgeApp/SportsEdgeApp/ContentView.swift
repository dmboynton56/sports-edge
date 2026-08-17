//
//  ContentView.swift
//  SportsEdgeApp
//
//  Created by Drew Boynton on 3/13/26.
//

import SwiftUI

struct ContentView: View {
    @EnvironmentObject private var appStore: AppStore
    @StateObject private var router = AppRouter()

    var body: some View {
        let theme = appStore.themeVariant.palette

        TabView(selection: $router.selectedTab) {
            HomeView(appStore: appStore, repository: appStore.repository)
                .tabItem { Label(AppTab.home.title, systemImage: AppTab.home.systemImage) }
                .tag(AppTab.home)

            MarketsView(appStore: appStore, repository: appStore.repository, path: $router.marketsPath)
                .tabItem { Label(AppTab.markets.title, systemImage: AppTab.markets.systemImage) }
                .tag(AppTab.markets)

            PerformanceView(repository: appStore.repository)
                .tabItem { Label(AppTab.performance.title, systemImage: AppTab.performance.systemImage) }
                .tag(AppTab.performance)

            InsightsView(repository: appStore.repository)
                .tabItem { Label(AppTab.insights.title, systemImage: AppTab.insights.systemImage) }
                .tag(AppTab.insights)

            SettingsView(appStore: appStore)
                .tabItem { Label(AppTab.settings.title, systemImage: AppTab.settings.systemImage) }
                .tag(AppTab.settings)
        }
        .tint(theme.accent)
        .preferredColorScheme(.dark)
        .background(theme.background.ignoresSafeArea())
        .environment(\.appTheme, theme)
        .onOpenURL { url in
            if let route = AppRouter.route(for: url) {
                appStore.selectedLeague = route.league
            }
            router.open(url: url)
        }
    }
}

#Preview {
    ContentView()
        .environmentObject(AppStore())
}
