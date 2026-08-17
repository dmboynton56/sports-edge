import SwiftUI

struct InsightsView: View {
    @Environment(\.appTheme) private var theme
    @StateObject private var viewModel: InsightsViewModel

    init(repository: SportsEdgeRepository) {
        _viewModel = StateObject(wrappedValue: InsightsViewModel(repository: repository))
    }

    var body: some View {
        NavigationStack {
            Group {
                switch viewModel.state {
                case .idle, .loading:
                    LoadingStateView(message: "Checking data quality…")
                case .failed(let message):
                    ErrorStateView(message: message) { Task { await viewModel.load() } }
                case .loaded(let envelope):
                    content(envelope)
                }
            }
            .navigationTitle("Insights")
            .navigationBarTitleDisplayMode(.large)
            .task { await viewModel.load() }
            .refreshable { await viewModel.load() }
        }
    }

    @ViewBuilder
    private func content(_ envelope: APIEnvelope<InsightsPayload>) -> some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                FreshnessBanner(envelope: envelope.freshness, gaps: envelope.gaps)
                Text("Data quality")
                    .font(.title3.weight(.bold))
                ForEach(envelope.data.dataQuality) { item in
                    qualityRow(item)
                }

                Text("Model evaluations")
                    .font(.title3.weight(.bold))
                    .padding(.top, 6)
                if envelope.data.evaluations.isEmpty {
                    EmptyStateView(title: "No evaluations", message: "Evaluation rows will appear as model runs are promoted.", systemImage: "checkmark.shield")
                } else {
                    ForEach(envelope.data.evaluations) { evaluation in
                        VStack(alignment: .leading, spacing: 8) {
                            HStack {
                                Text("\(evaluation.league) · \(evaluation.modelVersion)")
                                    .font(.headline)
                                Spacer()
                                Text(evaluation.status.capitalized)
                                    .font(.caption.weight(.bold))
                                    .foregroundStyle(evaluation.status == "approved" ? theme.positive : theme.warning)
                            }
                            Text(evaluation.evaluationName)
                                .font(.subheadline)
                                .foregroundStyle(.secondary)
                            HStack(spacing: 10) {
                                MetricTile(label: "ROI", value: evaluation.roi?.formatted(.percent.precision(.fractionLength(1))) ?? "—", tint: theme.positive)
                                MetricTile(label: "AUC", value: evaluation.auc?.formatted(.number.precision(.fractionLength(2))) ?? "—")
                            }
                        }
                        .appCard()
                    }
                }

                Text("Strategy evidence")
                    .font(.title3.weight(.bold))
                    .padding(.top, 6)
                ForEach(envelope.data.strategies) { strategy in
                    HStack {
                        VStack(alignment: .leading, spacing: 4) {
                            Text("\(strategy.league) · \(strategy.strategyId)")
                                .font(.subheadline.weight(.semibold))
                            Text("\(strategy.market) · \(strategy.bets ?? 0) bets")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                        Spacer()
                        Text(strategy.roi?.formatted(.percent.precision(.fractionLength(1))) ?? "—")
                            .font(.subheadline.monospacedDigit().weight(.bold))
                            .foregroundStyle(theme.positive)
                    }
                    .appCard()
                }
            }
            .padding(.horizontal)
            .padding(.bottom, 24)
        }
        .scrollIndicators(.hidden)
    }

    private func qualityRow(_ item: DataQualityItem) -> some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: icon(for: item.status))
                .foregroundStyle(color(for: item.status))
                .font(.title3)
            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text(item.label)
                        .font(.subheadline.weight(.semibold))
                    Spacer()
                    Text(item.status.capitalized)
                        .font(.caption.weight(.bold))
                        .foregroundStyle(color(for: item.status))
                }
                Text(item.detail)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                if item.updatedAt != nil {
                    Text("Updated \(formattedTimestamp(item.updatedAt))")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
            }
        }
        .appCard()
    }

    private func icon(for status: String) -> String {
        switch status {
        case "ok": "checkmark.seal.fill"
        case "warning": "exclamationmark.triangle.fill"
        case "blocked": "xmark.octagon.fill"
        default: "questionmark.diamond.fill"
        }
    }

    private func color(for status: String) -> Color {
        switch status {
        case "ok": theme.positive
        case "warning": theme.warning
        case "blocked": theme.danger
        default: theme.textSecondary
        }
    }
}
