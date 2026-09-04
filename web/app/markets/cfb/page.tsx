import { EmptyState } from "@/components/dashboard/EmptyState";
import { MarketsTable } from "@/components/dashboard/MarketsTable";
import { PageHeader, SectionHeading } from "@/components/dashboard/PageHeader";
import { Card } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { getCfbMarketFeed } from "@/lib/data/cfb-markets";
import { formatDateTime, formatPct } from "@/lib/format";

export const dynamic = "force-dynamic";

export default async function CfbMarketsPage() {
  const feed = await getCfbMarketFeed();
  const positiveEdges = feed.predictions
    .filter((row) => row.ev != null && row.ev > 0)
    .sort((left, right) => (right.ev ?? 0) - (left.ev ?? 0));

  return (
    <div>
      <PageHeader
        title="College Football Markets"
        description="Daily projected points, win chances, moneylines, spreads, and totals from a leakage-safe team model with current sportsbook prices. This is a research board until historical closing-line performance is validated."
        meta={feed.generatedAt}
      />

      <SectionHeading title="Projected scores & win chances" note="All scheduled games, including model-only moneylines" />
      {feed.games.length ? (
        <Card className="overflow-hidden">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Kickoff</TableHead>
                <TableHead>Matchup</TableHead>
                <TableHead>Projected score</TableHead>
                <TableHead>Win chance</TableHead>
                <TableHead>Total</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {feed.games.map((game) => (
                <TableRow key={game.eventId}>
                  <TableCell>{formatDateTime(game.eventTime)}</TableCell>
                  <TableCell className="font-semibold text-foreground">{game.awayTeam} @ {game.homeTeam}</TableCell>
                  <TableCell>{game.awayTeam} {game.predictedAwayPoints.toFixed(1)} · {game.homeTeam} {game.predictedHomePoints.toFixed(1)}</TableCell>
                  <TableCell>{game.homeTeam} {formatPct(game.homeWinProbability)}</TableCell>
                  <TableCell>{game.predictedTotal.toFixed(1)}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </Card>
      ) : (
        <EmptyState title="No CFB slate right now" description="The board publishes when ESPN lists an upcoming FBS slate." />
      )}

      <SectionHeading title="Positive-EV priced rows" note="Research signal · sort and filter by market or book" />
      <Card className="overflow-hidden p-5">
        <MarketsTable
          initialPredictions={positiveEdges}
          initialGaps={feed.gaps}
          defaultSortKey="ev"
        />
      </Card>
    </div>
  );
}
