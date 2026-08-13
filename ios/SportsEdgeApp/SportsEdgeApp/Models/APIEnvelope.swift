import Foundation

enum FreshnessStatus: String, Codable, CaseIterable {
    case fresh
    case stale
    case missing
    case offline

    var label: String {
        switch self {
        case .fresh: "Fresh"
        case .stale: "Stale"
        case .missing: "Missing"
        case .offline: "Offline"
        }
    }
}

enum DataSource: String, Codable {
    case supabase
    case staticJSON = "static_json"
    case mixed
    case fixture
    case unavailable

    var label: String {
        switch self {
        case .supabase: "Supabase"
        case .staticJSON: "Static artifact"
        case .mixed: "Mixed sources"
        case .fixture: "Fixture data"
        case .unavailable: "Unavailable"
        }
    }
}

struct FreshnessMetadata: Codable, Equatable {
    let status: FreshnessStatus
    let source: DataSource
    let updatedAt: String?
    let ageSeconds: Int?
}

struct APIEnvelope<Payload: Codable>: Codable {
    let schemaVersion: String
    let generatedAt: String
    let data: Payload
    let gaps: [String]
    let freshness: FreshnessMetadata

    func offlineCopy(reason: String) -> APIEnvelope<Payload> {
        APIEnvelope(
            schemaVersion: schemaVersion,
            generatedAt: generatedAt,
            data: data,
            gaps: Array(Set(gaps + [reason])).sorted(),
            freshness: FreshnessMetadata(
                status: .offline,
                source: freshness.source,
                updatedAt: freshness.updatedAt,
                ageSeconds: freshness.ageSeconds
            )
        )
    }
}

enum Loadable<Value> {
    case idle
    case loading
    case loaded(Value)
    case failed(String)
}
