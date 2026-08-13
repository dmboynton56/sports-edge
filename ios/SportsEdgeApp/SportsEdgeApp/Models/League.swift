import Foundation

enum League: String, Codable, CaseIterable, Hashable, Identifiable {
    case nba = "NBA"
    case nfl = "NFL"
    case mlb = "MLB"
    case pga = "PGA"

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .nba: "NBA"
        case .nfl: "NFL"
        case .mlb: "MLB"
        case .pga: "PGA"
        }
    }

    var systemImage: String {
        switch self {
        case .nba: "basketball.fill"
        case .nfl: "football.fill"
        case .mlb: "baseball.fill"
        case .pga: "figure.golf"
        }
    }
}
