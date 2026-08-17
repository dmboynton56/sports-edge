import SwiftUI

struct LoadingStateView: View {
    let message: String
    @Environment(\.appTheme) private var theme

    var body: some View {
        VStack(spacing: 12) {
            ProgressView()
                .controlSize(.large)
                .tint(theme.accent)
            Text(message)
                .font(.subheadline)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, minHeight: 220)
        .accessibilityElement(children: .combine)
        .accessibilityLabel(message)
    }
}

struct ErrorStateView: View {
    let message: String
    let retry: () -> Void
    @Environment(\.appTheme) private var theme

    var body: some View {
        VStack(spacing: 14) {
            Image(systemName: "wifi.exclamationmark")
                .font(.system(size: 30, weight: .semibold))
                .foregroundStyle(theme.warning)
            Text("Couldn’t load this view")
                .font(.headline)
            Text(message)
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
            Button("Try again", action: retry)
                .buttonStyle(.borderedProminent)
                .tint(theme.accent)
        }
        .frame(maxWidth: .infinity, minHeight: 260)
        .padding(24)
        .accessibilityElement(children: .contain)
    }
}

struct EmptyStateView: View {
    let title: String
    let message: String
    var systemImage: String = "tray"

    var body: some View {
        VStack(spacing: 12) {
            Image(systemName: systemImage)
                .font(.system(size: 30, weight: .semibold))
                .foregroundStyle(.secondary)
            Text(title)
                .font(.headline)
            Text(message)
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
        }
        .frame(maxWidth: .infinity, minHeight: 220)
        .padding(24)
        .accessibilityElement(children: .combine)
    }
}

struct FreshnessBanner: View {
    let envelope: FreshnessMetadata
    let gaps: [String]
    @Environment(\.appTheme) private var theme

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: envelope.status == .offline ? "icloud.slash" : "clock.badge.checkmark")
                .foregroundStyle(envelope.status == .fresh ? theme.positive : theme.warning)
            VStack(alignment: .leading, spacing: 3) {
                Text("\(envelope.status.label) · \(envelope.source.label)")
                    .font(.subheadline.weight(.semibold))
                Text(envelope.updatedAt == nil ? "No update timestamp" : "Updated \(formattedTimestamp(envelope.updatedAt))")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                if let gap = gaps.first {
                    Text(gap)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(2)
                }
            }
            Spacer(minLength: 0)
        }
        .padding(12)
        .background(envelope.status == .fresh ? theme.accentSoft.opacity(0.75) : theme.warning.opacity(0.12), in: RoundedRectangle(cornerRadius: 16, style: .continuous))
        .accessibilityElement(children: .combine)
    }
}
