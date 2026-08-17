import SwiftUI

struct MetricTile: View {
    let label: String
    let value: String
    var tint: Color = .primary
    @Environment(\.appTheme) private var theme

    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            Text(label.uppercased())
                .font(.caption2.weight(.bold))
                .foregroundStyle(theme.textSecondary)
            Text(value)
                .font(.title3.monospacedDigit().weight(.bold))
                .foregroundStyle(tint)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(12)
        .background(theme.surfaceMuted, in: RoundedRectangle(cornerRadius: 16, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .strokeBorder(theme.border, lineWidth: 1)
        }
    }
}

struct DetailMetricRow: View {
    let label: String
    let value: String
    @Environment(\.appTheme) private var theme

    var body: some View {
        HStack {
            Text(label)
                .foregroundStyle(theme.textSecondary)
            Spacer()
            Text(value)
                .font(.body.monospacedDigit().weight(.semibold))
                .foregroundStyle(theme.textPrimary)
        }
        .padding(.vertical, 5)
    }
}
