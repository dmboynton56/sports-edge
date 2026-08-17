import SwiftUI

enum ThemeVariant: String, CaseIterable, Identifiable {
    case carbonGreen
    case midnightBlue

    var id: String { rawValue }

    var title: String {
        switch self {
        case .carbonGreen: "Carbon + Signal"
        case .midnightBlue: "Midnight + Electric"
        }
    }

    var subtitle: String {
        switch self {
        case .carbonGreen: "A high-energy green edge"
        case .midnightBlue: "A cool, analytical blue edge"
        }
    }

    var systemImage: String {
        switch self {
        case .carbonGreen: "bolt.fill"
        case .midnightBlue: "waveform.path.ecg"
        }
    }

    var palette: ThemePalette {
        switch self {
        case .carbonGreen:
            ThemePalette(
                background: Color(red: 0.035, green: 0.047, blue: 0.063),
                surface: Color(red: 0.071, green: 0.094, blue: 0.122),
                surfaceElevated: Color(red: 0.102, green: 0.133, blue: 0.169),
                surfaceMuted: Color(red: 0.055, green: 0.071, blue: 0.094),
                border: Color.white.opacity(0.09),
                accent: Color(red: 0.30, green: 0.91, blue: 0.62),
                accentSoft: Color(red: 0.07, green: 0.22, blue: 0.15),
                positive: Color(red: 0.30, green: 0.91, blue: 0.62),
                warning: Color(red: 1.00, green: 0.72, blue: 0.28),
                danger: Color(red: 1.00, green: 0.35, blue: 0.40),
                textPrimary: Color(red: 0.95, green: 0.98, blue: 1.00),
                textSecondary: Color(red: 0.58, green: 0.66, blue: 0.75),
                heroGradient: LinearGradient(
                    colors: [Color(red: 0.08, green: 0.12, blue: 0.15), Color(red: 0.04, green: 0.36, blue: 0.24)],
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                )
            )
        case .midnightBlue:
            ThemePalette(
                background: Color(red: 0.028, green: 0.047, blue: 0.094),
                surface: Color(red: 0.055, green: 0.086, blue: 0.145),
                surfaceElevated: Color(red: 0.090, green: 0.137, blue: 0.227),
                surfaceMuted: Color(red: 0.043, green: 0.071, blue: 0.122),
                border: Color.white.opacity(0.10),
                accent: Color(red: 0.30, green: 0.65, blue: 1.00),
                accentSoft: Color(red: 0.07, green: 0.18, blue: 0.33),
                positive: Color(red: 0.31, green: 0.88, blue: 0.63),
                warning: Color(red: 1.00, green: 0.77, blue: 0.35),
                danger: Color(red: 1.00, green: 0.40, blue: 0.48),
                textPrimary: Color(red: 0.95, green: 0.97, blue: 1.00),
                textSecondary: Color(red: 0.58, green: 0.65, blue: 0.79),
                heroGradient: LinearGradient(
                    colors: [Color(red: 0.08, green: 0.10, blue: 0.24), Color(red: 0.04, green: 0.31, blue: 0.48)],
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                )
            )
        }
    }
}

struct ThemePalette {
    let background: Color
    let surface: Color
    let surfaceElevated: Color
    let surfaceMuted: Color
    let border: Color
    let accent: Color
    let accentSoft: Color
    let positive: Color
    let warning: Color
    let danger: Color
    let textPrimary: Color
    let textSecondary: Color
    let heroGradient: LinearGradient
}

private struct AppThemeKey: EnvironmentKey {
    static let defaultValue = ThemeVariant.carbonGreen.palette
}

extension EnvironmentValues {
    var appTheme: ThemePalette {
        get { self[AppThemeKey.self] }
        set { self[AppThemeKey.self] = newValue }
    }
}

struct AppCardModifier: ViewModifier {
    @Environment(\.appTheme) private var theme

    func body(content: Content) -> some View {
        content
            .padding(16)
            .background(theme.surface, in: RoundedRectangle(cornerRadius: 22, style: .continuous))
            .overlay {
                RoundedRectangle(cornerRadius: 22, style: .continuous)
                    .strokeBorder(theme.border, lineWidth: 1)
            }
            .shadow(color: .black.opacity(0.22), radius: 14, y: 8)
    }
}

extension View {
    func appCard() -> some View { modifier(AppCardModifier()) }
}

func formattedTimestamp(_ raw: String?) -> String {
    guard let raw, let date = parsedTimestamp(raw) else { return "Not available" }
    return date.formatted(date: .abbreviated, time: .shortened)
}

private func parsedTimestamp(_ raw: String) -> Date? {
    var normalized = raw.replacingOccurrences(of: " ", with: "T")
    if normalized.hasSuffix("+00") {
        normalized += ":00"
    }

    let isoFormatter = ISO8601DateFormatter()
    isoFormatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
    if let date = isoFormatter.date(from: normalized) {
        return date
    }

    isoFormatter.formatOptions = [.withInternetDateTime]
    if let date = isoFormatter.date(from: normalized) {
        return date
    }

    let formatter = DateFormatter()
    formatter.locale = Locale(identifier: "en_US_POSIX")
    formatter.timeZone = TimeZone(secondsFromGMT: 0)
    for format in ["yyyy-MM-dd'T'HH:mm:ss.SSSSSSXXXXX", "yyyy-MM-dd'T'HH:mm:ssXXXXX"] {
        formatter.dateFormat = format
        if let date = formatter.date(from: normalized) {
            return date
        }
    }
    return nil
}

func formattedPercent(_ value: Double?, digits: Int = 1) -> String {
    guard let value else { return "—" }
    return value.formatted(.percent.precision(.fractionLength(digits)))
}

func formattedEdge(_ value: Double?) -> String {
    guard let value else { return "—" }
    return "\(value >= 0 ? "+" : "")\(value.formatted(.number.precision(.fractionLength(1)))) pts"
}

func formattedMarketEdge(_ market: EnrichedPick) -> String {
    guard let edge = market.edge else { return "—" }
    if market.isTeamMarket {
        return formattedEdge(edge)
    }
    return "\(edge >= 0 ? "+" : "")\(edge.formatted(.percent.precision(.fractionLength(1))))"
}

func route(for market: EnrichedPick) -> GameRoute {
    GameRoute(league: market.league, gameId: market.gameId, title: market.title)
}
