import Foundation

struct MarketsPayload: Codable, Hashable {
    let league: League
    let windowStart: String?
    let windowEnd: String?
    let markets: [EnrichedPick]
}
