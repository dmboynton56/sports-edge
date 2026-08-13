import SwiftUI

struct MarketsView: View {
    @ObservedObject private var appStore: AppStore
    @StateObject private var viewModel: MarketViewModel
    @Binding private var path: [GameRoute]

    init(appStore: AppStore, repository: SportsEdgeRepository, path: Binding<[GameRoute]>) {
        self.appStore = appStore
        _viewModel = StateObject(wrappedValue: MarketViewModel(repository: repository, league: appStore.selectedLeague))
        _path = path
    }

    var body: some View {
        NavigationStack(path: $path) {
            Group {
                switch viewModel.state {
                case .idle, .loading:
                    LoadingStateView(message: "Loading \(viewModel.league.displayName) markets…")
                case .failed(let message):
                    ErrorStateView(message: message) { Task { await viewModel.load() } }
                case .loaded(let envelope):
                    marketContent(envelope)
                }
            }
            .navigationTitle("Markets")
            .navigationBarTitleDisplayMode(.large)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Menu {
                        Picker("Sort markets", selection: $appStore.sortOption) {
                            ForEach(MarketSortOption.allCases) { option in
                                Text(option.title).tag(option)
                            }
                        }
                    } label: {
                        Image(systemName: "arrow.up.arrow.down.circle")
                    }
                    .accessibilityLabel("Sort markets")
                }
            }
            .task { await viewModel.load() }
            .onChange(of: appStore.selectedLeague) { _, league in
                Task { await viewModel.select(league: league) }
            }
            .refreshable { await viewModel.load() }
            .navigationDestination(for: GameRoute.self) { gameRoute in
                GameDetailView(repository: appStore.repository, route: gameRoute)
            }
        }
    }

    @ViewBuilder
    private func marketContent(_ envelope: APIEnvelope<MarketsPayload>) -> some View {
        let markets = sorted(envelope.data.markets)
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                leaguePicker
                FreshnessBanner(envelope: envelope.freshness, gaps: envelope.gaps)
                Text("\(markets.count) public model signals")
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(.secondary)

                if markets.isEmpty {
                    EmptyStateView(
                        title: "No \(viewModel.league.displayName) markets",
                        message: "There are no normalized rows for this league in the current serving window.",
                        systemImage: "chart.bar.xaxis"
                    )
                } else {
                    LazyVStack(spacing: 10) {
                        ForEach(markets) { market in
                            HStack(spacing: 8) {
                                if market.isTeamMarket {
                                    NavigationLink(value: route(for: market)) {
                                        MarketRowContent(market: market)
                                    }
                                    .buttonStyle(.plain)
                                } else {
                                    MarketRowContent(market: market)
                                }
                                FavoriteButton(
                                    isFavorite: appStore.isFavorite(market),
                                    action: { appStore.toggleFavorite(market) }
                                )
                            }
                            .padding(.vertical, 5)
                            .accessibilityElement(children: .contain)
                        }
                    }
                }
            }
            .padding(.horizontal)
            .padding(.bottom, 24)
        }
        .scrollIndicators(.hidden)
    }

    private var leaguePicker: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 8) {
                ForEach(League.allCases) { league in
                    Button {
                        appStore.selectedLeague = league
                    } label: {
                        Label(league.displayName, systemImage: league.systemImage)
                            .font(.subheadline.weight(.semibold))
                            .padding(.horizontal, 13)
                            .padding(.vertical, 9)
                            .foregroundStyle(viewModel.league == league ? .white : .primary)
                            .background(viewModel.league == league ? AppTheme.accent : Color.primary.opacity(0.07), in: Capsule())
                    }
                    .buttonStyle(.plain)
                    .accessibilityAddTraits(viewModel.league == league ? [.isSelected] : [])
                }
            }
        }
    }

    private func sorted(_ markets: [EnrichedPick]) -> [EnrichedPick] {
        switch appStore.sortOption {
        case .edge:
            if viewModel.league == .mlb {
                return markets.sorted {
                    let leftProbability = $0.modelProbability ?? -.infinity
                    let rightProbability = $1.modelProbability ?? -.infinity
                    if leftProbability != rightProbability {
                        return leftProbability > rightProbability
                    }
                    return abs($0.edge ?? 0) > abs($1.edge ?? 0)
                }
            }
            return markets.sorted { abs($0.edge ?? 0) > abs($1.edge ?? 0) }
        case .time:
            return markets.sorted { ($0.eventTime ?? "") < ($1.eventTime ?? "") }
        case .favorites:
            return markets.sorted {
                let leftFavorite = appStore.isFavorite($0)
                let rightFavorite = appStore.isFavorite($1)
                if leftFavorite != rightFavorite { return leftFavorite }
                return abs($0.edge ?? 0) > abs($1.edge ?? 0)
            }
        }
    }
}
