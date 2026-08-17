import SwiftUI

struct SettingsView: View {
    @ObservedObject var appStore: AppStore
    @Environment(\.appTheme) private var theme
    @State private var cacheCleared = false

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    ForEach(ThemeVariant.allCases) { variant in
                        Button {
                            appStore.themeVariant = variant
                        } label: {
                            HStack(spacing: 12) {
                                Image(systemName: variant.systemImage)
                                    .font(.headline)
                                    .foregroundStyle(variant.palette.accent)
                                    .frame(width: 28, height: 28)
                                    .background(variant.palette.accentSoft, in: RoundedRectangle(cornerRadius: 8, style: .continuous))

                                VStack(alignment: .leading, spacing: 2) {
                                    Text(variant.title)
                                        .font(.headline)
                                        .foregroundStyle(theme.textPrimary)
                                    Text(variant.subtitle)
                                        .font(.caption)
                                        .foregroundStyle(theme.textSecondary)
                                }

                                Spacer()

                                if appStore.themeVariant == variant {
                                    Image(systemName: "checkmark.circle.fill")
                                        .foregroundStyle(variant.palette.accent)
                                }
                            }
                        }
                        .buttonStyle(.plain)
                        .accessibilityLabel(variant.title)
                        .accessibilityValue(appStore.themeVariant == variant ? "Selected" : "Not selected")
                    }
                } header: {
                    Text("Visual identity")
                } footer: {
                    Text("Sports Edge stays dark-first; this changes the app's signal color and atmosphere.")
                }

                Section("Data source") {
                    Toggle("Use fixture data", isOn: $appStore.usesFixtureData)
                        .tint(theme.accent)
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
                            .foregroundStyle(theme.positive)
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
                                        .foregroundStyle(theme.accent)
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
            .scrollContentBackground(.hidden)
            .background(theme.background)
            .navigationTitle("Settings")
            .navigationBarTitleDisplayMode(.large)
        }
    }
}
