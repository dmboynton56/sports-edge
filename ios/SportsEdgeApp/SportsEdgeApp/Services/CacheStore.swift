import Foundation

actor CacheStore {
    private let directory: URL
    private let encoder: JSONEncoder
    private let decoder: JSONDecoder

    init(fileManager: FileManager = .default) {
        let base = fileManager.urls(for: .cachesDirectory, in: .userDomainMask).first
            ?? fileManager.temporaryDirectory
        self.directory = base.appendingPathComponent("SportsEdge", isDirectory: true)
        self.encoder = JSONEncoder()
        self.decoder = JSONDecoder()
    }

    func load<Payload: Codable>(_ type: APIEnvelope<Payload>.Type, key: String) -> APIEnvelope<Payload>? {
        let url = directory.appendingPathComponent("\(key).json")
        guard let data = try? Data(contentsOf: url) else { return nil }
        return try? decoder.decode(type, from: data)
    }

    func save<Payload: Codable>(_ value: APIEnvelope<Payload>, key: String) {
        do {
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            let data = try encoder.encode(value)
            try data.write(to: directory.appendingPathComponent("\(key).json"), options: .atomic)
        } catch {
            // Cache failure must never make a successful API response fail.
        }
    }

    func removeAll() {
        try? FileManager.default.removeItem(at: directory)
    }
}
