import Foundation

struct DataQualityItem: Codable, Hashable, Identifiable {
    let id: String
    let label: String
    let status: String
    let updatedAt: String?
    let detail: String
}

struct EvaluationSummary: Codable, Hashable, Identifiable {
    let id: String
    let league: String
    let modelVersion: String
    let evaluationName: String
    let generatedAt: String
    let status: String
    let roi: Double?
    let auc: Double?
}

struct StrategySummary: Codable, Hashable, Identifiable {
    let id: String
    let league: String
    let modelVersion: String
    let strategyId: String
    let market: String
    let sampleSize: Int?
    let bets: Int?
    let roi: Double?
}

struct InsightsPayload: Codable, Hashable {
    let dataQuality: [DataQualityItem]
    let evaluations: [EvaluationSummary]
    let strategies: [StrategySummary]
}

struct FeatureDriver: Codable, Hashable, Identifiable {
    var id: String { feature }

    let feature: String
    let value: Double
    let impact: Double
    let isHeuristic: Bool?
}

struct GameExplanation: Codable, Hashable {
    let gameId: String
    let league: League
    let modelVersion: String
    let predictionTs: String
    let topFeatures: [FeatureDriver]
    let injuryAdjusted: Bool
    let homeInjuryDelta: Double?
    let awayInjuryDelta: Double?
    let baseVsAdjusted: [String: JSONValue]?
}

enum JSONValue: Codable, Hashable {
    case string(String)
    case number(Double)
    case bool(Bool)
    case object([String: JSONValue])
    case array([JSONValue])
    case null

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() { self = .null }
        else if let value = try? container.decode(Bool.self) { self = .bool(value) }
        else if let value = try? container.decode(Double.self) { self = .number(value) }
        else if let value = try? container.decode(String.self) { self = .string(value) }
        else if let value = try? container.decode([String: JSONValue].self) { self = .object(value) }
        else if let value = try? container.decode([JSONValue].self) { self = .array(value) }
        else { throw DecodingError.dataCorruptedError(in: container, debugDescription: "Unsupported JSON value") }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .string(let value): try container.encode(value)
        case .number(let value): try container.encode(value)
        case .bool(let value): try container.encode(value)
        case .object(let value): try container.encode(value)
        case .array(let value): try container.encode(value)
        case .null: try container.encodeNil()
        }
    }
}

struct GameDetailPayload: Codable, Hashable {
    let game: EnrichedPick
    let explanation: GameExplanation?
}
