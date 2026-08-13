import Foundation

enum RepositoryMode {
    case live
    case fixture
}

final class SportsEdgeRepository {
    var mode: RepositoryMode

    private let client: APIClient
    private let cache: CacheStore

    init(client: APIClient = APIClient(), cache: CacheStore = CacheStore(), mode: RepositoryMode = .fixture) {
        self.client = client
        self.cache = cache
        self.mode = mode
    }

    func home() async -> APIEnvelope<HomePayload> {
        await load(
            key: "home",
            path: "api/mobile/v1/home",
            fixture: MockData.home
        )
    }

    func markets(for league: League) async -> APIEnvelope<MarketsPayload> {
        await load(
            key: "markets-\(league.rawValue.lowercased())",
            path: "api/mobile/v1/markets/\(league.rawValue.lowercased())",
            fixture: MockData.markets(for: league)
        )
    }

    func gameDetail(for route: GameRoute) async -> APIEnvelope<GameDetailPayload?> {
        await load(
            key: "game-\(route.league.rawValue.lowercased())-\(route.gameId)",
            path: "api/mobile/v1/games/\(route.league.rawValue.lowercased())/\(route.gameId)",
            fixture: MockData.gameDetail(for: route)
        )
    }

    func performance() async -> APIEnvelope<PerformancePayload> {
        await load(
            key: "performance",
            path: "api/mobile/v1/performance",
            fixture: MockData.performance
        )
    }

    func insights() async -> APIEnvelope<InsightsPayload> {
        await load(
            key: "insights",
            path: "api/mobile/v1/insights",
            fixture: MockData.insights
        )
    }

    func clearCache() async {
        await cache.removeAll()
    }

    private func load<Payload: Codable>(
        key: String,
        path: String,
        fixture: APIEnvelope<Payload>
    ) async -> APIEnvelope<Payload> {
        if case .fixture = mode { return fixture }
        do {
            let response = try await client.get(path) as APIEnvelope<Payload>
            await cache.save(response, key: key)
            return response
        } catch {
            if let cached = await cache.load(APIEnvelope<Payload>.self, key: key) {
                return cached.offlineCopy(reason: "Showing the last successful payload while offline.")
            }
            return fixture.offlineCopy(reason: "Live data is unavailable; showing fixture data.")
        }
    }
}
