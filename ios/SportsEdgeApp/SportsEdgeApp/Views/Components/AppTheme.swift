import SwiftUI

enum AppTheme {
    static let accent = Color(red: 0.18, green: 0.66, blue: 0.56)
    static let accentSoft = Color(red: 0.83, green: 0.96, blue: 0.91)
    static let ink = Color(red: 0.08, green: 0.12, blue: 0.16)
    static let warning = Color.orange
    static let danger = Color.red
    static let positive = Color(red: 0.10, green: 0.55, blue: 0.35)

    static let heroGradient = LinearGradient(
        colors: [Color(red: 0.09, green: 0.20, blue: 0.22), Color(red: 0.18, green: 0.46, blue: 0.39)],
        startPoint: .topLeading,
        endPoint: .bottomTrailing
    )
}

struct AppCardModifier: ViewModifier {
    func body(content: Content) -> some View {
        content
            .padding(16)
            .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 22, style: .continuous))
            .overlay {
                RoundedRectangle(cornerRadius: 22, style: .continuous)
                    .strokeBorder(.primary.opacity(0.07), lineWidth: 1)
            }
    }
}

extension View {
    func appCard() -> some View { modifier(AppCardModifier()) }
}

func formattedTimestamp(_ raw: String?) -> String {
    guard let raw, let date = ISO8601DateFormatter().date(from: raw) else { return "Not available" }
    return date.formatted(date: .abbreviated, time: .shortened)
}

func formattedPercent(_ value: Double?, digits: Int = 1) -> String {
    guard let value else { return "—" }
    return value.formatted(.percent.precision(.fractionLength(digits)))
}

func formattedEdge(_ value: Double?) -> String {
    guard let value else { return "—" }
    return "\(value >= 0 ? "+" : "")\(value.formatted(.number.precision(.fractionLength(1)))) pts"
}

func route(for market: EnrichedPick) -> GameRoute {
    GameRoute(league: market.league, gameId: market.gameId, title: market.title)
}
