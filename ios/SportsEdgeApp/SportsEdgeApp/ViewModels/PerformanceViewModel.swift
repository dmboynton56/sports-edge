import Foundation
import Combine
import SwiftUI

@MainActor
final class PerformanceViewModel: ObservableObject {
    @Published private(set) var state: Loadable<APIEnvelope<PerformancePayload>> = .idle
    private let repository: SportsEdgeRepository

    init(repository: SportsEdgeRepository) {
        self.repository = repository
    }

    func load() async {
        state = .loading
        state = .loaded(await repository.performance())
    }
}
