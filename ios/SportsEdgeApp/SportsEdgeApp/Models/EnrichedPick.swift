import Foundation

enum MarketKind: String, Codable, Hashable {
    case teamSpread = "team_spread"
    case playerMarket = "player_market"
}

struct EnrichedPick: Codable, Hashable, Identifiable {
    let id: String
    let gameId: String
    let league: League
    let kind: MarketKind
    let title: String
    let subtitle: String
    let eventTime: String?
    let homeTeam: String?
    let awayTeam: String?
    let subject: String?
    let market: String
    let book: String?
    let line: Double?
    let price: Double?
    let modelProbability: Double?
    let impliedProbability: Double?
    let edge: Double?
    let ev: Double?
    let confidence: Double?
    let modelVersion: String?
    let freshnessStatus: String
    let predictionTs: String?
    let oddsTs: String?
    let injuryAdjusted: Bool
    let injuryDataMissing: Bool

    var isTeamMarket: Bool { kind == .teamSpread }
}
