import { PickCard, type EnrichedPick } from '@/components/PickCard';

// Dummy data for the prototype
const MOCK_PICKS: EnrichedPick[] = [
  {
    id: "1",
    game: {
      id: "game1",
      league: "NBA",
      homeTeam: "Lakers",
      awayTeam: "Warriors",
      gameTimeUtc: "2026-03-12T02:30:00Z"
    },
    currentOdds: {
      book: "Consensus",
      market: "spread",
      line: -3.5,
      price: -110
    },
    prediction: {
      predictedSpread: -5.5,
      homeWinProb: 0.68
    },
    edgePts: 2.0 // Model likes Lakers by more (-5.5 vs -3.5)
  },
  {
    id: "2",
    game: {
      id: "game2",
      league: "NBA",
      homeTeam: "Celtics",
      awayTeam: "Nuggets",
      gameTimeUtc: "2026-03-12T00:00:00Z"
    },
    currentOdds: {
      book: "Consensus",
      market: "spread",
      line: 2.5,
      price: -110
    },
    prediction: {
      predictedSpread: 0.5,
      homeWinProb: 0.48
    },
    edgePts: 2.0 // Model thinks Celtics lose by less (0.5 vs 2.5)
  },
  {
    id: "3",
    game: {
      id: "game3",
      league: "NFL",
      homeTeam: "Chiefs",
      awayTeam: "Ravens",
      gameTimeUtc: "2026-09-10T00:20:00Z"
    },
    currentOdds: {
      book: "Consensus",
      market: "spread",
      line: -4.0,
      price: -110
    },
    prediction: {
      predictedSpread: -3.5,
      homeWinProb: 0.62
    },
    edgePts: -0.5 // Model doesn't like Chiefs as much as market
  }
];

export default function Home() {
  const topEdges = MOCK_PICKS.filter(p => p.edgePts > 0).sort((a, b) => b.edgePts - a.edgePts);

  return (
    <div className="container mx-auto px-4 py-8 max-w-5xl">
      <div className="flex flex-col space-y-8">
        
        {/* Header Section */}
        <div className="flex flex-col space-y-2">
          <h1 className="text-3xl font-bold tracking-tight">Today's Top Edges</h1>
          <p className="text-muted-foreground">
            Machine learning generated betting lines compared against real-time market consensus.
          </p>
        </div>

        {/* Dashboard Stats */}
        <div className="grid gap-4 md:grid-cols-3">
          <div className="rounded-xl border border-border bg-card p-6 shadow-sm">
            <h3 className="text-sm font-medium text-muted-foreground">Active Models</h3>
            <div className="mt-2 text-3xl font-bold">4</div>
          </div>
          <div className="rounded-xl border border-border bg-card p-6 shadow-sm">
            <h3 className="text-sm font-medium text-muted-foreground">Games Analyzed Today</h3>
            <div className="mt-2 text-3xl font-bold">12</div>
          </div>
          <div className="rounded-xl border border-border bg-card p-6 shadow-sm">
            <h3 className="text-sm font-medium text-muted-foreground">7-Day Hit Rate</h3>
            <div className="mt-2 text-3xl font-bold text-accent">58.4%</div>
          </div>
        </div>

        {/* Main Feed */}
        <div>
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-semibold tracking-tight">Highest Expected Value (E[V])</h2>
            <div className="flex space-x-2">
              <button className="px-3 py-1 text-sm rounded-md bg-secondary text-secondary-foreground hover:bg-secondary/80 transition-colors">
                All Sports
              </button>
              <button className="px-3 py-1 text-sm rounded-md text-muted-foreground hover:bg-secondary hover:text-secondary-foreground transition-colors">
                NBA Only
              </button>
            </div>
          </div>
          
          <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
            {topEdges.map(pick => (
              <PickCard key={pick.id} pick={pick} />
            ))}
          </div>
        </div>

        {/* All Matches */}
        <div className="mt-8 pt-8 border-t border-border">
          <h2 className="text-xl font-semibold tracking-tight mb-4">Other Analyzed Games</h2>
          <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3 opacity-70">
            {MOCK_PICKS.filter(p => p.edgePts <= 0).map(pick => (
              <PickCard key={pick.id} pick={pick} />
            ))}
          </div>
        </div>

      </div>
    </div>
  );
}
