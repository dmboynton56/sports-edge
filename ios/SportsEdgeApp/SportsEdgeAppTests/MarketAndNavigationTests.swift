import XCTest
@testable import SportsEdgeApp

final class MarketAndNavigationTests: XCTestCase {
    func testEdgeSortingUsesAbsoluteEdge() {
        let sorted = MarketSorting.byEdge(MockData.nbaMarketsForTesting)
        XCTAssertEqual(sorted.first?.id, "NBA-demo-1")
        XCTAssertEqual(sorted.last?.id, "NBA-demo-3")
    }

    func testLeagueFilteringKeepsOnlySelectedLeague() {
        let markets = MockData.nbaMarketsForTesting + MockData.nflMarketsForTesting
        let filtered = MarketSorting.filtered(markets, league: .nfl)

        XCTAssertFalse(filtered.isEmpty)
        XCTAssertTrue(filtered.allSatisfy { $0.league == .nfl })
    }

    func testEmptyAndErrorStatesRemainRepresentable() {
        let empty = APIEnvelope(
            schemaVersion: "1.0",
            generatedAt: "2026-08-12T18:15:00Z",
            data: MarketsPayload(league: .nba, windowStart: nil, windowEnd: nil, markets: []),
            gaps: ["No NBA games in window."],
            freshness: FreshnessMetadata(status: .missing, source: .unavailable, updatedAt: nil, ageSeconds: nil)
        )
        let error: Loadable<APIEnvelope<MarketsPayload>> = .failed("Network unavailable")

        XCTAssertTrue(empty.data.markets.isEmpty)
        XCTAssertEqual(empty.freshness.status, .missing)
        if case .failed(let message) = error {
            XCTAssertEqual(message, "Network unavailable")
        } else {
            XCTFail("Expected a failed loadable state")
        }
    }

    @MainActor
    func testCustomSchemeRoutesToGameDetail() {
        let url = URL(string: "sportsedge://game/NBA/demo-nba-1")!
        let route = AppRouter.route(for: url)

        XCTAssertEqual(route?.league, .nba)
        XCTAssertEqual(route?.gameId, "demo-nba-1")
    }

    @MainActor
    func testHTTPSGameRouteRoutesToGameDetail() {
        let url = URL(string: "https://sports-edge.example/mobile/game/NFL/demo-nfl-2")!
        let route = AppRouter.route(for: url)

        XCTAssertEqual(route?.league, .nfl)
        XCTAssertEqual(route?.gameId, "demo-nfl-2")
    }
}

private extension MockData {
    static var nbaMarketsForTesting: [EnrichedPick] {
        [
            teamMarketForTesting(id: "NBA-demo-1", league: .nba, edge: 3.5),
            teamMarketForTesting(id: "NBA-demo-2", league: .nba, edge: -2.4),
            teamMarketForTesting(id: "NBA-demo-3", league: .nba, edge: 1.8),
        ]
    }

    static var nflMarketsForTesting: [EnrichedPick] {
        [teamMarketForTesting(id: "NFL-demo-1", league: .nfl, edge: 2.9)]
    }

    static func teamMarketForTesting(id: String, league: League, edge: Double) -> EnrichedPick {
        EnrichedPick(
            id: id,
            gameId: id.lowercased(),
            league: league,
            kind: .teamSpread,
            title: id,
            subtitle: "Fixture",
            eventTime: nil,
            homeTeam: "HOME",
            awayTeam: "AWAY",
            subject: nil,
            market: "Spread",
            book: nil,
            line: nil,
            price: nil,
            modelProbability: nil,
            impliedProbability: nil,
            edge: edge,
            ev: nil,
            confidence: nil,
            modelVersion: "test",
            freshnessStatus: "fresh",
            predictionTs: nil,
            oddsTs: nil,
            injuryAdjusted: false,
            injuryDataMissing: false
        )
    }
}
