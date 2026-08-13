import SwiftUI

struct HomeView: View {
    @ObservedObject private var appStore: AppStore
    @StateObject private var viewModel: HomeViewModel

    init(appStore: AppStore, repository: SportsEdgeRepository) {
        self.appStore = appStore
        _viewModel = StateObject(wrappedValue: HomeViewModel(repository: repository))
    }

    var body: some View {
        NavigationStack {
            Group {
                switch viewModel.state {
                case .idle, .loading:
                    LoadingStateView(message: "Finding today’s strongest edges…")
                case .failed(let message):
                    ErrorStateView(message: message) { Task { await viewModel.load() } }
                case .loaded(let envelope):
                    homeContent(envelope)
                }
            }
            .navigationTitle("Top Edges")
            .navigationBarTitleDisplayMode(.large)
            .task { await viewModel.load() }
            .refreshable { await viewModel.load() }
        }
    }

    @ViewBuilder
    private func homeContent(_ envelope: APIEnvelope<HomePayload>) -> some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                hero
                FreshnessBanner(envelope: envelope.freshness, gaps: envelope.gaps)

                if envelope.data.topEdges.isEmpty {
                    EmptyStateView(
                        title: "No current edges",
                        message: "The public serving feed has no NBA or NFL games in the current window.",
                        systemImage: "chart.bar.xaxis"
                    )
                } else {
                    sectionHeader("Strongest signals", detail: "Sorted by absolute model edge")
                    ForEach(envelope.data.topEdges) { market in
                        NavigationLink(value: route(for: market)) {
                            EdgeCard(
                                market: market,
                                isFavorite: appStore.isFavorite(market),
                                toggleFavorite: { appStore.toggleFavorite(market) }
                            )
                        }
                        .buttonStyle(.plain)
                    }
                }

                sectionHeader("Tracked leagues", detail: "Public, read-only model coverage")
                LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 12) {
                    ForEach(envelope.data.leagueSummaries.filter { appStore.trackedLeagues.contains($0.league) }) { summary in
                        VStack(alignment: .leading, spacing: 8) {
                            Label(summary.league.displayName, systemImage: summary.league.systemImage)
                                .font(.subheadline.weight(.semibold))
                            Text("\(summary.marketCount) markets")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                            Text(formattedEdge(summary.topEdge))
                                .font(.title3.monospacedDigit().weight(.bold))
                                .foregroundStyle(AppTheme.positive)
                        }
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .appCard()
                    }
                }

                Text("Sports Edge is analytics only. No bets, transactions, or account connections are supported.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                    .padding(.horizontal, 4)
            }
            .padding(.horizontal)
            .padding(.bottom, 24)
        }
        .scrollIndicators(.hidden)
        .navigationDestination(for: GameRoute.self) { gameRoute in
            GameDetailView(repository: appStore.repository, route: gameRoute)
        }
    }

    private var hero: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Image(systemName: "waveform.path.ecg")
                    .font(.title2.weight(.bold))
                Spacer()
                Text("COMPANION")
                    .font(.caption2.weight(.bold))
                    .tracking(1.4)
                    .opacity(0.7)
            }
            Text("See where the model sees daylight.")
                .font(.title2.weight(.bold))
                .fixedSize(horizontal: false, vertical: true)
            Text("A clear read on public sports analytics, model edge, and data health.")
                .font(.subheadline)
                .opacity(0.82)
        }
        .foregroundStyle(.white)
        .padding(20)
        .background(AppTheme.heroGradient, in: RoundedRectangle(cornerRadius: 26, style: .continuous))
        .accessibilityElement(children: .combine)
    }

    private func sectionHeader(_ title: String, detail: String) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(title)
                .font(.title3.weight(.bold))
            Text(detail)
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }
}
