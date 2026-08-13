import Foundation
import Combine
import SwiftUI

@MainActor
final class MarketViewModel: ObservableObject {
    @Published private(set) var state: Loadable<APIEnvelope<MarketsPayload>> = .idle
    @Published private(set) var league: League
    private let repository: SportsEdgeRepository

    init(repository: SportsEdgeRepository, league: League) {
        self.repository = repository
        self.league = league
    }

    func select(league: League) async {
        guard self.league != league || isLoadedForDifferentLeague else { return }
        self.league = league
        await load()
    }

    func load() async {
        state = .loading
        let response = await repository.markets(for: league)
        state = .loaded(response)
    }

    private var isLoadedForDifferentLeague: Bool {
        if case .loaded(let response) = state { return response.data.league != league }
        return false
    }
}
