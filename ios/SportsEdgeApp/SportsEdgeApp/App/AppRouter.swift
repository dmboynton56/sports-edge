import Foundation
import Combine
import SwiftUI

enum AppTab: String, CaseIterable, Identifiable {
    case home
    case markets
    case performance
    case insights
    case settings

    var id: String { rawValue }

    var title: String {
        switch self {
        case .home: "Top Edges"
        case .markets: "Markets"
        case .performance: "Performance"
        case .insights: "Insights"
        case .settings: "Settings"
        }
    }

    var systemImage: String {
        switch self {
        case .home: "sparkles"
        case .markets: "chart.bar.xaxis"
        case .performance: "chart.xyaxis.line"
        case .insights: "checkmark.shield"
        case .settings: "gearshape"
        }
    }
}

struct GameRoute: Hashable, Identifiable {
    let league: League
    let gameId: String
    let title: String

    var id: String { "\(league.rawValue)-\(gameId)" }
}

@MainActor
final class AppRouter: ObservableObject {
    @Published var selectedTab: AppTab = .home
    @Published var marketsPath: [GameRoute] = []

    func open(url: URL) {
        guard let route = Self.route(for: url) else { return }
        selectedTab = .markets
        marketsPath = [route]
    }

    static func route(for url: URL) -> GameRoute? {
        let path = url.pathComponents.filter { $0 != "/" }
        let values: [String]
        if url.scheme == "sportsedge", url.host == "game" {
            values = path
        } else if url.scheme == "https" || url.scheme == "http" {
            guard let gameIndex = path.firstIndex(of: "game") else { return nil }
            values = Array(path.dropFirst(gameIndex + 1))
        } else {
            return nil
        }

        guard values.count >= 2,
              let league = League(rawValue: values[0].uppercased()) else { return nil }
        let gameId = values[1]
        return GameRoute(league: league, gameId: gameId, title: "Game detail")
    }
}
