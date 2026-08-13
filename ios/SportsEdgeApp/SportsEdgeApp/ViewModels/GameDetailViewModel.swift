import Foundation
import Combine
import SwiftUI

@MainActor
final class GameDetailViewModel: ObservableObject {
    @Published private(set) var state: Loadable<APIEnvelope<GameDetailPayload?>> = .idle
    private let repository: SportsEdgeRepository
    let route: GameRoute

    init(repository: SportsEdgeRepository, route: GameRoute) {
        self.repository = repository
        self.route = route
    }

    func load() async {
        state = .loading
        let response = await repository.gameDetail(for: route)
        state = .loaded(response)
    }
}
