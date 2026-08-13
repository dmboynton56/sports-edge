import Foundation
import Combine
import SwiftUI

@MainActor
final class InsightsViewModel: ObservableObject {
    @Published private(set) var state: Loadable<APIEnvelope<InsightsPayload>> = .idle
    private let repository: SportsEdgeRepository

    init(repository: SportsEdgeRepository) {
        self.repository = repository
    }

    func load() async {
        state = .loading
        state = .loaded(await repository.insights())
    }
}
