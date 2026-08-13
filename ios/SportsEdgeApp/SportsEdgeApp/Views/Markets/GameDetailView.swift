import SwiftUI

struct GameDetailView: View {
    @StateObject private var viewModel: GameDetailViewModel

    init(repository: SportsEdgeRepository, route: GameRoute) {
        _viewModel = StateObject(wrappedValue: GameDetailViewModel(repository: repository, route: route))
    }

    var body: some View {
        Group {
            switch viewModel.state {
            case .idle, .loading:
                LoadingStateView(message: "Loading model detail…")
            case .failed(let message):
                ErrorStateView(message: message) { Task { await viewModel.load() } }
            case .loaded(let envelope):
                detailContent(envelope)
            }
        }
        .navigationTitle(viewModel.route.title)
        .navigationBarTitleDisplayMode(.inline)
        .task { await viewModel.load() }
        .refreshable { await viewModel.load() }
    }

    @ViewBuilder
    private func detailContent(_ envelope: APIEnvelope<GameDetailPayload?>) -> some View {
        if let payload = envelope.data {
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    matchupHeader(payload.game)
                    FreshnessBanner(envelope: envelope.freshness, gaps: envelope.gaps)
                    predictionCard(payload.game)
                    explanationCard(payload.explanation)
                }
                .padding(.horizontal)
                .padding(.bottom, 24)
            }
            .scrollIndicators(.hidden)
        } else {
            EmptyStateView(
                title: "Game not found",
                message: "This game is no longer in the public serving window.",
                systemImage: "questionmark.folder"
            )
        }
    }

    private func matchupHeader(_ market: EnrichedPick) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Label("\(market.league.displayName) · \(market.market)", systemImage: market.league.systemImage)
                .font(.subheadline.weight(.bold))
                .foregroundStyle(AppTheme.accent)
            Text(market.title)
                .font(.system(size: 30, weight: .bold, design: .rounded))
            Text(market.subtitle)
                .font(.subheadline)
                .foregroundStyle(.secondary)
        }
        .padding(.top, 8)
    }

    private func predictionCard(_ market: EnrichedPick) -> some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Prediction")
                .font(.headline)
            HStack(spacing: 10) {
                MetricTile(label: "Model edge", value: formattedEdge(market.edge), tint: (market.edge ?? 0) >= 0 ? AppTheme.positive : AppTheme.danger)
                MetricTile(label: "Win probability", value: formattedPercent(market.modelProbability))
            }
            DetailMetricRow(label: "Book line", value: market.line?.formatted(.number.precision(.fractionLength(1))) ?? "—")
            DetailMetricRow(label: "Model version", value: market.modelVersion ?? "—")
            DetailMetricRow(label: "Updated", value: formattedTimestamp(market.predictionTs))
            HStack(spacing: 8) {
                Text(market.freshnessStatus.capitalized)
                    .font(.caption.weight(.bold))
                    .padding(.horizontal, 9)
                    .padding(.vertical, 5)
                    .background(AppTheme.accent.opacity(0.14), in: Capsule())
                if market.injuryAdjusted {
                    Text("Injury adjusted")
                        .font(.caption.weight(.bold))
                        .padding(.horizontal, 9)
                        .padding(.vertical, 5)
                        .background(Color.orange.opacity(0.16), in: Capsule())
                }
            }
        }
        .appCard()
    }

    @ViewBuilder
    private func explanationCard(_ explanation: GameExplanation?) -> some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Why the model moved")
                .font(.headline)
            if let explanation {
                if explanation.injuryAdjusted {
                    HStack(spacing: 8) {
                        Image(systemName: "cross.case.fill")
                            .foregroundStyle(AppTheme.warning)
                        Text("Injury-adjusted prediction")
                            .font(.subheadline.weight(.semibold))
                    }
                }
                ForEach(explanation.topFeatures) { feature in
                    VStack(alignment: .leading, spacing: 6) {
                        HStack {
                            Text(feature.feature)
                                .font(.subheadline.weight(.semibold))
                            Spacer()
                            Text(feature.impact >= 0 ? "+\(feature.impact, specifier: "%.1f")" : "\(feature.impact, specifier: "%.1f")")
                                .font(.subheadline.monospacedDigit().weight(.bold))
                                .foregroundStyle(feature.impact >= 0 ? AppTheme.positive : AppTheme.danger)
                        }
                        GeometryReader { proxy in
                            Capsule()
                                .fill(feature.impact >= 0 ? AppTheme.positive.opacity(0.68) : AppTheme.danger.opacity(0.68))
                                .frame(width: min(proxy.size.width, max(10, abs(feature.impact) * 28)), height: 6)
                        }
                        .frame(height: 6)
                    }
                }
                if let homeDelta = explanation.homeInjuryDelta, let awayDelta = explanation.awayInjuryDelta {
                    Divider()
                    DetailMetricRow(label: "Home injury delta", value: homeDelta.formatted(.number.precision(.fractionLength(2))))
                    DetailMetricRow(label: "Away injury delta", value: awayDelta.formatted(.number.precision(.fractionLength(2))))
                }
            } else {
                Text("No persisted explanation row is available for this prediction yet.")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }
        }
        .appCard()
    }
}
