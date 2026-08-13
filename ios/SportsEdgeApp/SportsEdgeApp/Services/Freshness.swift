import Foundation

enum FreshnessCalculator {
    static func status(
        updatedAt: Date?,
        now: Date = Date(),
        source: DataSource,
        gaps: [String] = []
    ) -> FreshnessStatus {
        if source == .fixture { return .fresh }
        if gaps.contains(where: { $0.localizedCaseInsensitiveContains("offline") }) { return .offline }
        guard let updatedAt else { return .missing }
        return now.timeIntervalSince(updatedAt) > 24 * 60 * 60 ? .stale : .fresh
    }

    static func ageSeconds(updatedAt: Date?, now: Date = Date()) -> Int? {
        guard let updatedAt else { return nil }
        return max(0, Int(now.timeIntervalSince(updatedAt)))
    }
}
