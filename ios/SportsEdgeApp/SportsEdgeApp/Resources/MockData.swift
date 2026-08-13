import Foundation

enum MockData {
    static let generatedAt = "2026-08-12T18:15:00Z"
    static let predictionAt = "2026-08-12T17:45:00Z"

    static var home: APIEnvelope<HomePayload> {
        let markets = nbaMarkets + nflMarkets
        let topEdges = markets.sorted { abs($0.edge ?? 0) > abs($1.edge ?? 0) }
        return envelope(
            HomePayload(
                topEdges: Array(topEdges.prefix(8)),
                leagueSummaries: [
                    LeagueSummary(league: .nba, marketCount: nbaMarkets.count, topEdge: nbaMarkets.first?.edge),
                    LeagueSummary(league: .nfl, marketCount: nflMarkets.count, topEdge: nflMarkets.first?.edge),
                ]
            )
        )
    }

    static var performance: APIEnvelope<PerformancePayload> {
        envelope(
            PerformancePayload(
                generatedAt: generatedAt,
                records: [
                    PerformanceRecord(
                        league: "NBA",
                        modelVersion: "nba-v3",
                        season: "2025-26",
                        market: "Spread",
                        sampleSize: 412,
                        roi: 0.084,
                        units: 34.6,
                        bets: 412,
                        wins: 218,
                        losses: 178,
                        pushes: 16,
                        productionStatus: "approved",
                        gates: passingGates
                    ),
                    PerformanceRecord(
                        league: "NFL",
                        modelVersion: "nfl-v1",
                        season: "2025",
                        market: "Spread",
                        sampleSize: 186,
                        roi: 0.061,
                        units: 11.3,
                        bets: 186,
                        wins: 101,
                        losses: 78,
                        pushes: 7,
                        productionStatus: "candidate",
                        gates: candidateGates
                    ),
                    PerformanceRecord(
                        league: "MLB",
                        modelVersion: "mlb-hr-v1",
                        season: "2026",
                        market: "Home run",
                        sampleSize: 928,
                        roi: 0.032,
                        units: 9.8,
                        bets: 928,
                        wins: 213,
                        losses: 694,
                        pushes: 21,
                        productionStatus: "candidate",
                        gates: candidateGates
                    ),
                ]
            )
        )
    }

    static var insights: APIEnvelope<InsightsPayload> {
        envelope(
            InsightsPayload(
                dataQuality: [
                    DataQualityItem(
                        id: "serving",
                        label: "Serving feed",
                        status: "ok",
                        updatedAt: predictionAt,
                        detail: "Public NBA and NFL serving rows are available for the current window."
                    ),
                    DataQualityItem(
                        id: "freshness",
                        label: "Prediction freshness",
                        status: "ok",
                        updatedAt: predictionAt,
                        detail: "Latest fixture predictions are inside the 24-hour freshness gate."
                    ),
                    DataQualityItem(
                        id: "injuries",
                        label: "Injury coverage",
                        status: "warning",
                        updatedAt: predictionAt,
                        detail: "Injury-aware schema is present; coverage varies by game and league."
                    ),
                ],
                evaluations: [
                    EvaluationSummary(
                        id: "nba-v3-eval",
                        league: "NBA",
                        modelVersion: "nba-v3",
                        evaluationName: "2025-26 regular season",
                        generatedAt: generatedAt,
                        status: "approved",
                        roi: 0.084,
                        auc: 0.63
                    ),
                    EvaluationSummary(
                        id: "nfl-v1-eval",
                        league: "NFL",
                        modelVersion: "nfl-v1",
                        evaluationName: "2025 regular season",
                        generatedAt: generatedAt,
                        status: "candidate",
                        roi: 0.061,
                        auc: 0.60
                    ),
                ],
                strategies: [
                    StrategySummary(
                        id: "nba-edge-15",
                        league: "NBA",
                        modelVersion: "nba-v3",
                        strategyId: "spread-edge-1.5",
                        market: "Spread",
                        sampleSize: 188,
                        bets: 188,
                        roi: 0.097
                    ),
                    StrategySummary(
                        id: "nfl-edge-2",
                        league: "NFL",
                        modelVersion: "nfl-v1",
                        strategyId: "spread-edge-2.0",
                        market: "Spread",
                        sampleSize: 74,
                        bets: 74,
                        roi: 0.074
                    ),
                ]
            )
        )
    }

    static func markets(for league: League) -> APIEnvelope<MarketsPayload> {
        let markets: [EnrichedPick]
        switch league {
        case .nba: markets = nbaMarkets
        case .nfl: markets = nflMarkets
        case .mlb: markets = mlbMarkets
        case .pga: markets = pgaMarkets
        }
        return envelope(
            MarketsPayload(
                league: league,
                windowStart: "2026-08-12",
                windowEnd: league == .nfl ? "2026-08-19" : "2026-08-13",
                markets: markets
            )
        )
    }

    static func gameDetail(for route: GameRoute) -> APIEnvelope<GameDetailPayload?> {
        let market = (nbaMarkets + nflMarkets).first { $0.gameId == route.gameId && $0.league == route.league }
        let data = market.map {
            GameDetailPayload(
                game: $0,
                explanation: GameExplanation(
                    gameId: $0.gameId,
                    league: $0.league,
                    modelVersion: $0.modelVersion ?? "n/a",
                    predictionTs: $0.predictionTs ?? predictionAt,
                    topFeatures: [
                        FeatureDriver(feature: "Recent net rating", value: 6.2, impact: 1.9, isHeuristic: false),
                        FeatureDriver(feature: "Opponent pace", value: 101.4, impact: 1.1, isHeuristic: false),
                        FeatureDriver(feature: "Home-court prior", value: 1.0, impact: 0.8, isHeuristic: true),
                        FeatureDriver(feature: "Rest differential", value: 1.0, impact: 0.6, isHeuristic: false),
                        FeatureDriver(feature: "Injury adjustment", value: -0.7, impact: -0.7, isHeuristic: false),
                    ],
                    injuryAdjusted: $0.injuryAdjusted,
                    homeInjuryDelta: $0.injuryAdjusted ? -0.7 : 0,
                    awayInjuryDelta: 0,
                    baseVsAdjusted: ["baseSpread": .number(-1.8), "adjustedSpread": .number(-2.5)]
                )
            )
        }
        return envelope(data)
    }

    private static var nbaMarkets: [EnrichedPick] {
        [
            teamMarket(
                id: "NBA-demo-1",
                gameId: "demo-nba-1",
                title: "Denver @ Phoenix",
                subtitle: "Today · 7:00 PM",
                home: "PHX",
                away: "DEN",
                line: -2.5,
                modelProbability: 0.62,
                edge: 3.5,
                modelSpread: -6.0,
                model: "nba-v3",
                injuryAdjusted: true
            ),
            teamMarket(
                id: "NBA-demo-2",
                gameId: "demo-nba-2",
                title: "Boston @ Cleveland",
                subtitle: "Today · 8:30 PM",
                home: "CLE",
                away: "BOS",
                line: 1.5,
                modelProbability: 0.56,
                edge: -2.4,
                modelSpread: -0.9,
                model: "nba-v3"
            ),
            teamMarket(
                id: "NBA-demo-3",
                gameId: "demo-nba-3",
                title: "Minnesota @ Dallas",
                subtitle: "Tomorrow · 6:00 PM",
                home: "DAL",
                away: "MIN",
                line: -1.0,
                modelProbability: 0.54,
                edge: 1.8,
                modelSpread: -2.8,
                model: "nba-v3"
            ),
        ]
    }

    private static var nflMarkets: [EnrichedPick] {
        [
            teamMarket(
                id: "NFL-demo-1",
                gameId: "demo-nfl-1",
                title: "Buffalo @ Kansas City",
                subtitle: "Sun · 2:25 PM · Week 1",
                home: "KC",
                away: "BUF",
                line: -1.5,
                modelProbability: 0.59,
                edge: 2.9,
                modelSpread: -4.4,
                model: "nfl-v1",
                injuryAdjusted: true
            ),
            teamMarket(
                id: "NFL-demo-2",
                gameId: "demo-nfl-2",
                title: "Green Bay @ Chicago",
                subtitle: "Sun · 11:00 AM · Week 1",
                home: "CHI",
                away: "GB",
                line: 2.0,
                modelProbability: 0.53,
                edge: 1.7,
                modelSpread: 0.3,
                model: "nfl-v1"
            ),
            teamMarket(
                id: "NFL-demo-3",
                gameId: "demo-nfl-3",
                title: "Cincinnati @ Pittsburgh",
                subtitle: "Sun · 11:00 AM · Week 1",
                home: "PIT",
                away: "CIN",
                line: -3.0,
                modelProbability: 0.57,
                edge: -1.4,
                modelSpread: -1.6,
                model: "nfl-v1"
            ),
        ]
    }

    private static var mlbMarkets: [EnrichedPick] {
        [
            playerMarket(id: "MLB-demo-1", gameId: "demo-mlb-1", subject: "Aaron Judge", market: "Home run", probability: 0.21, edge: 0.045, model: "mlb-hr-v1"),
            playerMarket(id: "MLB-demo-2", gameId: "demo-mlb-2", subject: "Kyle Schwarber", market: "Home run", probability: 0.18, edge: 0.031, model: "mlb-hr-v1"),
        ]
    }

    private static var pgaMarkets: [EnrichedPick] {
        [
            playerMarket(id: "PGA-demo-1", gameId: "demo-pga-1", subject: "Scottie Scheffler", market: "Win", probability: 0.14, edge: 0.021, model: "pga-v2"),
            playerMarket(id: "PGA-demo-2", gameId: "demo-pga-1", subject: "Rory McIlroy", market: "Top 10", probability: 0.39, edge: 0.018, model: "pga-v2"),
        ]
    }

    private static func teamMarket(
        id: String,
        gameId: String,
        title: String,
        subtitle: String,
        home: String,
        away: String,
        line: Double,
        modelProbability: Double,
        edge: Double,
        modelSpread: Double,
        model: String,
        injuryAdjusted: Bool = false
    ) -> EnrichedPick {
        let league: League = id.hasPrefix("NBA") ? .nba : .nfl
        return EnrichedPick(
            id: id,
            gameId: gameId,
            league: league,
            kind: .teamSpread,
            title: title,
            subtitle: subtitle,
            eventTime: league == .nba ? "2026-08-12T19:00:00Z" : "2026-09-13T18:25:00Z",
            homeTeam: home,
            awayTeam: away,
            subject: nil,
            market: "Spread",
            book: "Market consensus",
            line: line,
            price: -110,
            modelProbability: modelProbability,
            impliedProbability: 0.524,
            edge: edge,
            ev: edge / 100,
            confidence: modelProbability,
            modelVersion: model,
            freshnessStatus: "fresh",
            predictionTs: predictionAt,
            oddsTs: predictionAt,
            injuryAdjusted: injuryAdjusted,
            injuryDataMissing: false
        )
    }

    private static func playerMarket(id: String, gameId: String, subject: String, market: String, probability: Double, edge: Double, model: String) -> EnrichedPick {
        let league: League = id.hasPrefix("MLB") ? .mlb : .pga
        return EnrichedPick(
            id: id,
            gameId: gameId,
            league: league,
            kind: .playerMarket,
            title: subject,
            subtitle: "Model probability · \(market)",
            eventTime: "2026-08-13T18:00:00Z",
            homeTeam: nil,
            awayTeam: nil,
            subject: subject,
            market: market,
            book: "Model",
            line: nil,
            price: nil,
            modelProbability: probability,
            impliedProbability: probability - edge,
            edge: edge,
            ev: edge,
            confidence: probability,
            modelVersion: model,
            freshnessStatus: "fresh",
            predictionTs: predictionAt,
            oddsTs: nil,
            injuryAdjusted: false,
            injuryDataMissing: false
        )
    }

    private static let passingGates = [
        ProductionGate(id: "sample", label: "Sample", status: "pass", detail: "412 graded games."),
        ProductionGate(id: "calibration", label: "Calibration", status: "pass", detail: "Brier and AUC evidence recorded."),
        ProductionGate(id: "strategy", label: "Strategy ROI", status: "pass", detail: "Best strategy ROI +8.4%."),
    ]

    private static let candidateGates = [
        ProductionGate(id: "sample", label: "Sample", status: "pass", detail: "Sample is above the minimum target."),
        ProductionGate(id: "calibration", label: "Calibration", status: "pass", detail: "Calibration evidence recorded."),
        ProductionGate(id: "injuries", label: "Injuries", status: "warning", detail: "Live injury coverage still needs rows."),
    ]

    private static func envelope<Payload: Codable>(_ data: Payload) -> APIEnvelope<Payload> {
        APIEnvelope(
            schemaVersion: "1.0",
            generatedAt: generatedAt,
            data: data,
            gaps: [],
            freshness: FreshnessMetadata(
                status: .fresh,
                source: .fixture,
                updatedAt: predictionAt,
                ageSeconds: 1800
            )
        )
    }
}
