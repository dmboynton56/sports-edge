import Foundation

struct LeagueSummary: Codable, Hashable, Identifiable {
    var id: String { league.rawValue }

    let league: League
    let marketCount: Int
    let topEdge: Double?
}

struct HomePayload: Codable, Hashable {
    let topEdges: [EnrichedPick]
    let leagueSummaries: [LeagueSummary]
}
