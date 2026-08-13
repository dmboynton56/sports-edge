import Foundation

enum APIClientError: LocalizedError {
    case missingBaseURL
    case invalidResponse
    case httpError(Int)

    var errorDescription: String? {
        switch self {
        case .missingBaseURL: "The mobile API URL is not configured."
        case .invalidResponse: "The mobile API returned an invalid response."
        case .httpError(let status): "The mobile API returned HTTP \(status)."
        }
    }
}

struct APIClient {
    let baseURL: URL?
    let session: URLSession
    private let decoder: JSONDecoder

    init(baseURL: URL? = AppConfiguration.apiBaseURL, session: URLSession = .shared) {
        self.baseURL = baseURL
        self.session = session
        self.decoder = JSONDecoder()
    }

    func get<Payload: Codable>(_ path: String) async throws -> APIEnvelope<Payload> {
        guard let baseURL else { throw APIClientError.missingBaseURL }
        let url = baseURL.appendingPathComponent(path)
        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        request.timeoutInterval = 20
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        request.setValue("SportsEdge-iOS/1.0", forHTTPHeaderField: "X-Sports-Edge-Client")

        let (data, response) = try await session.data(for: request)
        guard let response = response as? HTTPURLResponse else { throw APIClientError.invalidResponse }
        guard (200..<300).contains(response.statusCode) else { throw APIClientError.httpError(response.statusCode) }
        return try decoder.decode(APIEnvelope<Payload>.self, from: data)
    }
}
