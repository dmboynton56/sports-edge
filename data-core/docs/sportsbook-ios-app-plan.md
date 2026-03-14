# Plan: Sports-Edge iOS Analytics App (The "Sportsbook Mirror")

## Overview
This plan outlines the skeleton of a native iOS app that acts as a **sports betting helper and analytics tool**. Rather than being a place to place bets, it is a unified dashboard to analyze markets, find mathematical edges, expected value (E[V]), and view top model predictions. 

The UI will deliberately mimic modern sportsbooks (like DraftKings or FanDuel) so users find it intuitively familiar—acting as a "sportsbook mirror"—but the primary focus is entirely on surfacing actionable data and model advantages across major American sports (NFL, NBA, MLB, NHL) and Golf.

## 1. App Architecture & Tech Stack
*   **Framework:** SwiftUI (Declarative UI, perfect for real-time dashboards and complex data state).
*   **Architecture:** MVVM (Model-View-ViewModel) to cleanly separate our raw database representations from the UI state.
*   **Networking:** `URLSession` combined with Swift `async/await` to fetch data from our Next.js/Node API backend (which reads from our PostgreSQL/Redis setup).

## 2. Core UI & Navigation (The "Mirror" Interface)
The navigation matches what bettors are used to, but the content inside the views prioritizes analytics over simple bet slips.

*   **Main Tab Bar (`TabView`)**
    *   **Top Edges (Home):** The dashboard. Shows the highest E[V] plays and most confident model predictions across *all* active sports for the day.
    *   **Markets / Sports:** Scrollable top menu (NFL, NBA, MLB, NHL, PGA) to drill down into specific leagues. Looks like a sportsbook lobby, but displays games with their associated edge/EV overlay.
    *   **My Tracker / Portfolio:** Where the user tracks their actual bets against our model's performance to see if they are beating the closing line value (CLV).
    *   **Settings / Subscriptions:** Profile, dark/light mode, notification toggles, and premium tier access.

*   **The "Analytics Card" Component**
    *   A reusable UI component (`AnalyticsCardView`) used across the app. Replaces the traditional "Bet Card".
    *   Displays: Matchup (Home vs Away), Game Time, and Team Logos.
    *   **Market Consensus:** The average or most common book line (e.g., Lakers -3.5 @ -110).
    *   **Model Prediction:** Our true line (e.g., Lakers -5.5).
    *   **The Edge Indicators:** 
        *   E[V] Percentage (e.g., +4.2% EV).
        *   Point Edge (e.g., 2.0 pts).
        *   Hit Rate (e.g., "Model is 8-2 (80%) on LAL spreads").
    *   **Sort/Filter Bar:** Above any list of games, allowing users to instantly re-sort the board by:
        *   Highest E[V]
        *   Highest Model Win Probability (%)
        *   Recent Model Hit Rate

*   **Future Feature: Line Shopping**
    *   Tapping into a game card will eventually reveal an "Odds Screen" showing real-time lines from multiple books (DraftKings, FanDuel, BetMGM) so the user can see where to grab the best number to maximize the calculated E[V].

## 3. Data Models (Mapped to Current Database)
These Swift structs map directly to the JSON payloads from our backend, derived from `WHOLE PROJECT TABLES.json` and our SQL schema (`games`, `odds_snapshots`, `model_predictions`).

```swift
// Maps to `games` / `raw_schedules`
struct Game: Identifiable, Codable {
    let id: UUID
    let league: String
    let season: Int
    let gameTimeUtc: Date
    let homeTeam: String
    let awayTeam: String
}

// Maps to `odds_snapshots`
struct OddsSnapshot: Codable {
    let book: String
    let market: String // 'spread', 'moneyline', 'total'
    let line: Double
    let price: Double
}

// Maps to `model_predictions`
struct ModelPrediction: Codable {
    let modelVersion: String
    let predictedSpread: Double
    let homeWinProb: Double
}

// Maps to `games_today_enriched` (The unified view for the UI)
struct EnrichedPick: Identifiable, Codable {
    let id: UUID // Maps to game_id
    let game: Game
    let currentOdds: OddsSnapshot
    let prediction: ModelPrediction
    
    // Computed properties for the UI
    var edgePts: Double {
        return prediction.predictedSpread - currentOdds.line
    }
}
```

## 4. Recommended Xcode File Structure
When generating the Xcode project, organize your folders (Groups) like this to easily scale as we transition from static edges to live odds comparison.

```text
SportsEdgeApp/
│
├── App/
│   ├── SportsEdgeApp.swift        // Main App entry point
│   └── Theme.swift                // Standardized colors (e.g., neon green for +EV, specific reds for -EV)
│
├── Models/
│   ├── Game.swift                 // Data structures defined above
│   ├── Odds.swift
│   ├── Prediction.swift
│   └── EnrichedPick.swift
│
├── Services/
│   ├── APIClient.swift            // Generic network requests
│   └── EdgeDataService.swift      // Fetches /api/games_today_enriched from our backend
│
├── ViewModels/
│   ├── TopEdgesViewModel.swift    // Handles sorting (by EV, Prob, Hit Rate)
│   └── MarketViewModel.swift      // Handles fetching data for specific leagues
│
├── Views/
│   ├── Components/
│   │   ├── AnalyticsCardView.swift// The core reusable game/edge card
│   │   ├── SortFilterBar.swift    // The sort toggle (EV vs Prob)
│   │   └── EdgeBadgeView.swift    // Reusable pill/badge for highlighting the E[V] number
│   │
│   ├── Main/
│   │   ├── MainTabView.swift      // The bottom navigation bar
│   │   └── HomeDashboardView.swift// The "Top Edges" landing page
│   │
│   └── Leagues/
│       ├── LeagueFeedView.swift   // Dynamic view for NFL, NBA, Golf, etc.
│       └── GameDetailView.swift   // Future: Deep dive into a game for Line Shopping
│
└── Assets.xcassets                // Team logos, custom icons
```

## 5. Next Steps to Get Started in Xcode
1. **Create the Project:** Open Xcode -> Create New Project -> iOS App -> Use "SwiftUI" for Interface.
2. **Setup the Skeleton:** Replicate the folder structure above using Xcode Groups.
3. **Mock the Data:** Create a `MockData.swift` file that provides static arrays of `EnrichedPick` objects. This allows UI development before the API backend is fully live.
4. **Build the `AnalyticsCardView`:** This is the most important piece of UI in the app. It needs to clearly convey the market line vs. your model's line without feeling cluttered. Spend time styling this with mock data to ensure sorting (e.g., high EV vs low EV) works and looks great.
