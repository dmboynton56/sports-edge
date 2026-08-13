import Foundation
import XCTest
@testable import SportsEdgeApp

final class APIAndFreshnessTests: XCTestCase {
    func testMockHomeRoundTripsThroughAPIEnvelopeDecoder() throws {
        let data = try JSONEncoder().encode(MockData.home)
        let decoded = try JSONDecoder().decode(APIEnvelope<HomePayload>.self, from: data)

        XCTAssertEqual(decoded.schemaVersion, "1.0")
        XCTAssertEqual(decoded.data.topEdges.first?.league, .nba)
        XCTAssertEqual(decoded.freshness.source, .fixture)
    }

    func testFreshnessTransitionsFromFreshToStaleAndOffline() {
        let now = Date(timeIntervalSince1970: 1_000_000)
        let recent = Date(timeIntervalSince1970: 999_500)
        let old = Date(timeIntervalSince1970: 1_000_000 - (25 * 60 * 60))

        XCTAssertEqual(FreshnessCalculator.status(updatedAt: recent, now: now, source: .supabase), .fresh)
        XCTAssertEqual(FreshnessCalculator.status(updatedAt: old, now: now, source: .supabase), .stale)
        XCTAssertEqual(FreshnessCalculator.status(updatedAt: recent, now: now, source: .supabase, gaps: ["offline fallback"]), .offline)
        XCTAssertEqual(FreshnessCalculator.status(updatedAt: nil, now: now, source: .unavailable), .missing)
    }

    func testSupabaseTimestampIsDisplayable() {
        XCTAssertNotEqual(
            formattedTimestamp("2026-08-13 12:41:31.675907+00"),
            "Not available"
        )
    }
}
