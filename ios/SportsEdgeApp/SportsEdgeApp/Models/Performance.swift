import Foundation

struct ProductionGate: Codable, Hashable, Identifiable {
    let id: String
    let label: String
    let status: String
    let detail: String
}

struct PerformanceRecord: Codable, Hashable, Identifiable {
    var id: String { "\(league)-\(modelVersion)-\(market)-\(season)" }

    let league: String
    let modelVersion: String
    let season: String
    let market: String
    let sampleSize: Int?
    let roi: Double?
    let units: Double?
    let bets: Int?
    let wins: Int?
    let losses: Int?
    let pushes: Int?
    let productionStatus: String
    let gates: [ProductionGate]
}

struct PerformancePayload: Codable, Hashable {
    let generatedAt: String?
    let records: [PerformanceRecord]
}
