import SwiftUI

struct SettingsView: View {
    @ObservedObject var appStore: AppStore
    @State private var cacheCleared = false

    var body: some View {
        NavigationStack {
            Form {
                Section("Data source") {
                    Toggle("Use fixture data", isOn: $appStore.usesFixtureData)
                        .tint(AppTheme.accent)
                    LabeledContent("API endpoint") {
                        Text(AppConfiguration.apiBaseURL?.host ?? "Not configured")
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                    }
                    Button {
                        Task {
                            await appStore.clearCache()
                            cacheCleared = true
                        }
                    } label: {
                        Label("Clear offline cache", systemImage: "trash")
                    }
                    if cacheCleared {
                        Text("Offline cache cleared.")
                            .font(.caption)
                            .foregroundStyle(AppTheme.positive)
                    }
                }

                Section("Tracked leagues") {
                    ForEach(League.allCases) { league in
                        Button {
                            appStore.toggleTrackedLeague(league)
                        } label: {
                            HStack {
                                Label(league.displayName, systemImage: league.systemImage)
                                    .foregroundStyle(.primary)
                                Spacer()
                                if appStore.trackedLeagues.contains(league) {
                                    Image(systemName: "checkmark")
                                        .foregroundStyle(AppTheme.accent)
                                        .fontWeight(.bold)
                                }
                            }
                        }
                        .accessibilityLabel("\(league.displayName) tracked")
                        .accessibilityValue(appStore.trackedLeagues.contains(league) ? "On" : "Off")
                    }
                }

                Section("Market defaults") {
                    Picker("Sort markets by", selection: $appStore.sortOption) {
                        ForEach(MarketSortOption.allCases) { option in
                            Text(option.title).tag(option)
                        }
                    }
                    Picker("Default league", selection: $appStore.selectedLeague) {
                        ForEach(League.allCases) { league in
                            Text(league.displayName).tag(league)
                        }
                    }
                }

                Section("About Sports Edge") {
                    Text("Sports Edge is a public, read-only companion for model edges, prediction explanations, performance evidence, and data freshness.")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                    Text("No login, account pairing, betting execution, or private broker data is supported in this version.")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                    LabeledContent("Version", value: "1.0 vertical slice")
                    LabeledContent("Minimum iOS", value: "18.0")
                }
            }
            .navigationTitle("Settings")
            .navigationBarTitleDisplayMode(.large)
        }
    }
}
