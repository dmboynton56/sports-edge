import Foundation
import Combine
import SwiftUI

@MainActor
final class HomeViewModel: ObservableObject {
    @Published private(set) var state: Loadable<APIEnvelope<HomePayload>> = .idle
    private let repository: SportsEdgeRepository

    init(repository: SportsEdgeRepository) {
        self.repository = repository
    }

    func load() async {
        state = .loading
        let response = await repository.home()
        state = .loaded(response)
    }
}
