import Foundation
import Combine
import SwiftUI

enum MarketSortOption: String, CaseIterable, Identifiable {
    case edge
    case time
    case favorites

    var id: String { rawValue }

    var title: String {
        switch self {
        case .edge: "Largest edge"
        case .time: "Start time"
        case .favorites: "Favorites first"
        }
    }
}

enum AppConfiguration {
    static let bundleIdentifier = "com.drewboynton.sportsedge"
    static let apiBaseURLKey = "MOBILE_API_BASE_URL"

    static var apiBaseURL: URL? {
        guard let raw = Bundle.main.object(forInfoDictionaryKey: apiBaseURLKey) as? String,
              !raw.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return nil }
        return URL(string: raw)
    }
}

@MainActor
final class AppStore: ObservableObject {
    static let fixtureModeKey = "sportsEdge.fixtureMode"
    static let selectedLeagueKey = "sportsEdge.selectedLeague"
    static let trackedLeaguesKey = "sportsEdge.trackedLeagues"
    static let sortOptionKey = "sportsEdge.sortOption"
    static let favoritesKey = "sportsEdge.favorites"
    static let themeVariantKey = "sportsEdge.themeVariant"

    let repository: SportsEdgeRepository
    private let defaults: UserDefaults

    @Published var selectedLeague: League {
        didSet { defaults.set(selectedLeague.rawValue, forKey: Self.selectedLeagueKey) }
    }
    @Published private(set) var trackedLeagues: Set<League> {
        didSet { defaults.set(trackedLeagues.map(\.rawValue).sorted(), forKey: Self.trackedLeaguesKey) }
    }
    @Published var sortOption: MarketSortOption {
        didSet { defaults.set(sortOption.rawValue, forKey: Self.sortOptionKey) }
    }
    @Published private(set) var favorites: Set<String> {
        didSet { defaults.set(Array(favorites).sorted(), forKey: Self.favoritesKey) }
    }
    @Published var usesFixtureData: Bool {
        didSet { defaults.set(usesFixtureData, forKey: Self.fixtureModeKey); repository.mode = usesFixtureData ? .fixture : .live }
    }
    @Published var themeVariant: ThemeVariant {
        didSet { defaults.set(themeVariant.rawValue, forKey: Self.themeVariantKey) }
    }

    init(defaults: UserDefaults = .standard, repository: SportsEdgeRepository? = nil) {
        self.defaults = defaults
        let fixture = defaults.object(forKey: Self.fixtureModeKey) as? Bool ?? true
        self.repository = repository ?? SportsEdgeRepository(mode: fixture ? .fixture : .live)
        self.usesFixtureData = fixture
        self.selectedLeague = League(rawValue: defaults.string(forKey: Self.selectedLeagueKey) ?? "NBA") ?? .nba
        let storedTracked = (defaults.array(forKey: Self.trackedLeaguesKey) as? [String] ?? [])
            .compactMap(League.init(rawValue:))
        self.trackedLeagues = storedTracked.isEmpty ? [.nba, .nfl] : Set(storedTracked)
        self.sortOption = MarketSortOption(rawValue: defaults.string(forKey: Self.sortOptionKey) ?? "edge") ?? .edge
        self.favorites = Set((defaults.array(forKey: Self.favoritesKey) as? [String]) ?? [])
        self.themeVariant = ThemeVariant(rawValue: defaults.string(forKey: Self.themeVariantKey) ?? "") ?? .carbonGreen
    }

    func toggleTrackedLeague(_ league: League) {
        if trackedLeagues.contains(league) {
            guard trackedLeagues.count > 1 else { return }
            trackedLeagues.remove(league)
        } else {
            trackedLeagues.insert(league)
        }
    }

    func toggleFavorite(_ market: EnrichedPick) {
        if favorites.contains(market.id) {
            favorites.remove(market.id)
        } else {
            favorites.insert(market.id)
        }
    }

    func isFavorite(_ market: EnrichedPick) -> Bool {
        favorites.contains(market.id)
    }

    func clearCache() async {
        await repository.clearCache()
    }
}
