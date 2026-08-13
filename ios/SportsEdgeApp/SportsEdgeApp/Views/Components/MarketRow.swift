import SwiftUI

struct MarketRowContent: View {
    let market: EnrichedPick

    var body: some View {
        HStack(spacing: 12) {
            ZStack {
                Circle()
                    .fill(AppTheme.accent.opacity(0.14))
                Image(systemName: market.league.systemImage)
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(AppTheme.accent)
            }
            .frame(width: 38, height: 38)

            VStack(alignment: .leading, spacing: 4) {
                Text(market.title)
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(.primary)
                    .lineLimit(1)
                Text(market.subtitle)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                HStack(spacing: 8) {
                    Text(market.market)
                    if market.injuryAdjusted { Text("Injury adjusted") }
                }
                .font(.caption2.weight(.medium))
                .foregroundStyle(.secondary)
            }

            Spacer(minLength: 8)

            VStack(alignment: .trailing, spacing: 4) {
                Text(formattedEdge(market.edge))
                    .font(.subheadline.monospacedDigit().weight(.bold))
                    .foregroundStyle((market.edge ?? 0) >= 0 ? AppTheme.positive : AppTheme.danger)
                if market.isTeamMarket {
                    Text(formattedPercent(market.modelProbability))
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                } else {
                    Text(formattedPercent(market.modelProbability))
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
            }
        }
        .contentShape(Rectangle())
        .accessibilityElement(children: .combine)
        .accessibilityLabel("\(market.league.displayName), \(market.title), edge \(formattedEdge(market.edge)), model probability \(formattedPercent(market.modelProbability))")
    }
}

struct FavoriteButton: View {
    let isFavorite: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Image(systemName: isFavorite ? "star.fill" : "star")
                .foregroundStyle(isFavorite ? .yellow : .secondary)
                .frame(width: 34, height: 34)
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityLabel(isFavorite ? "Remove favorite" : "Add favorite")
    }
}

struct EdgeCard: View {
    let market: EnrichedPick
    let isFavorite: Bool
    let toggleFavorite: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(alignment: .top) {
                Label(market.league.displayName, systemImage: market.league.systemImage)
                    .font(.caption.weight(.bold))
                    .foregroundStyle(AppTheme.accent)
                Spacer()
                FavoriteButton(isFavorite: isFavorite, action: toggleFavorite)
            }
            Text(market.title)
                .font(.title3.weight(.bold))
                .lineLimit(1)
            Text(market.subtitle)
                .font(.subheadline)
                .foregroundStyle(.secondary)
            HStack(alignment: .lastTextBaseline) {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Model edge")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Text(formattedEdge(market.edge))
                        .font(.system(size: 28, weight: .bold, design: .rounded))
                        .foregroundStyle((market.edge ?? 0) >= 0 ? AppTheme.positive : AppTheme.danger)
                }
                Spacer()
                VStack(alignment: .trailing, spacing: 4) {
                    Text("Win probability")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Text(formattedPercent(market.modelProbability))
                        .font(.title3.monospacedDigit().weight(.semibold))
                }
            }
            HStack(spacing: 8) {
                Text(market.modelVersion ?? "Model n/a")
                if market.injuryAdjusted {
                    Text("Injury adjusted")
                }
            }
            .font(.caption.weight(.medium))
            .foregroundStyle(.secondary)
        }
        .appCard()
        .accessibilityElement(children: .contain)
    }
}
