import Foundation

enum MarketSorting {
    static func byEdge(_ markets: [EnrichedPick]) -> [EnrichedPick] {
        markets.sorted { abs($0.edge ?? 0) > abs($1.edge ?? 0) }
    }

    static func filtered(_ markets: [EnrichedPick], league: League) -> [EnrichedPick] {
        markets.filter { $0.league == league }
    }
}
