import SwiftUI

struct PerformanceView: View {
    @StateObject private var viewModel: PerformanceViewModel

    init(repository: SportsEdgeRepository) {
        _viewModel = StateObject(wrappedValue: PerformanceViewModel(repository: repository))
    }

    var body: some View {
        NavigationStack {
            Group {
                switch viewModel.state {
                case .idle, .loading:
                    LoadingStateView(message: "Loading model results…")
                case .failed(let message):
                    ErrorStateView(message: message) { Task { await viewModel.load() } }
                case .loaded(let envelope):
                    content(envelope)
                }
            }
            .navigationTitle("Performance")
            .navigationBarTitleDisplayMode(.large)
            .task { await viewModel.load() }
            .refreshable { await viewModel.load() }
        }
    }

    @ViewBuilder
    private func content(_ envelope: APIEnvelope<PerformancePayload>) -> some View {
        let records = envelope.data.records
        let totalBets = records.compactMap(\.bets).reduce(0, +)
        let averageROI = records.compactMap(\.roi).average

        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                FreshnessBanner(envelope: envelope.freshness, gaps: envelope.gaps)
                LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 10) {
                    MetricTile(label: "Tracked bets", value: totalBets.formatted())
                    MetricTile(label: "Average ROI", value: averageROI.formatted(.percent.precision(.fractionLength(1))), tint: AppTheme.positive)
                }

                if records.isEmpty {
                    EmptyStateView(title: "No results yet", message: "Performance history will appear after the first graded run.", systemImage: "chart.xyaxis.line")
                } else {
                    Text("Model scorecards")
                        .font(.title3.weight(.bold))
                    ForEach(records) { record in
                        performanceCard(record)
                    }
                }
            }
            .padding(.horizontal)
            .padding(.bottom, 24)
        }
        .scrollIndicators(.hidden)
    }

    private func performanceCard(_ record: PerformanceRecord) -> some View {
        VStack(alignment: .leading, spacing: 13) {
            HStack {
                VStack(alignment: .leading, spacing: 3) {
                    Text(record.league)
                        .font(.headline)
                    Text("\(record.modelVersion) · \(record.market) · \(record.season)")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                Text(record.productionStatus.capitalized)
                    .font(.caption.weight(.bold))
                    .foregroundStyle(record.productionStatus == "approved" ? AppTheme.positive : AppTheme.warning)
            }
            HStack(spacing: 10) {
                MetricTile(label: "ROI", value: record.roi?.formatted(.percent.precision(.fractionLength(1))) ?? "—", tint: AppTheme.positive)
                MetricTile(label: "Sample", value: record.sampleSize?.formatted() ?? "—")
            }
            DetailMetricRow(label: "Record", value: "\(record.wins ?? 0)-\(record.losses ?? 0)-\(record.pushes ?? 0)")
            DetailMetricRow(label: "Units", value: record.units?.formatted(.number.precision(.fractionLength(1))) ?? "—")
            Divider()
            VStack(alignment: .leading, spacing: 8) {
                ForEach(record.gates) { gate in
                    HStack(spacing: 8) {
                        Image(systemName: gate.status == "pass" ? "checkmark.circle.fill" : "exclamationmark.triangle.fill")
                            .foregroundStyle(gate.status == "pass" ? AppTheme.positive : AppTheme.warning)
                        Text(gate.label)
                            .font(.caption.weight(.semibold))
                        Spacer()
                        Text(gate.status.capitalized)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
            }
        }
        .appCard()
        .accessibilityElement(children: .contain)
    }
}

private extension Collection where Element == Double {
    var average: Double { isEmpty ? 0 : reduce(0, +) / Double(count) }
}
