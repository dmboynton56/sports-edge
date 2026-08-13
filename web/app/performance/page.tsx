import Link from "next/link";
import { Activity, AlertTriangle, BarChart3, LineChart } from "lucide-react";

import { MetricCard } from "@/components/dashboard/MetricCard";
import { PageHeader } from "@/components/dashboard/PageHeader";
import { PerformanceTables } from "@/components/dashboard/PerformanceTables";
import { RoiChart } from "@/components/dashboard/RoiChart";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { getPerformanceHistory } from "@/lib/data/performance";
import { formatNumber, formatPct } from "@/lib/format";
import { SPORTS, type SportEntry } from "@/lib/markets-registry";
import { cn } from "@/lib/utils";

function SportCard({ sport, muted = false }: { sport: SportEntry; muted?: boolean }) {
  return (
    <Link href={`/performance/${sport.slug}`} className="block">
      <Card className={cn("h-full transition-colors hover:border-accent/50 hover:bg-accent/5", muted && "bg-muted/20 text-muted-foreground")}>
        <CardHeader>
          <div className="flex items-center justify-between gap-3">
            <CardTitle className={cn(muted && "text-base")}>{sport.label}</CardTitle>
            {sport.emphasis !== "primary" ? (
              <Badge variant="outline" className="text-muted-foreground">{sport.emphasis}</Badge>
            ) : null}
          </div>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">{sport.description}</p>
          <p className="mt-3 text-xs text-muted-foreground">
            {sport.markets.length} {sport.markets.length === 1 ? "market" : "markets"} in the performance tree
          </p>
        </CardContent>
      </Card>
    </Link>
  );
}

function LeagueSection({
  title,
  records,
}: {
  title: string;
  records: ReturnType<typeof getPerformanceHistory> extends Promise<infer T>
    ? T extends { records: infer R }
      ? R
      : never
    : never;
}) {
  if (!records.length) return null;
  return (
    <Card>
      <CardHeader>
        <CardTitle>{title}</CardTitle>
      </CardHeader>
      <CardContent>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Model</TableHead>
              <TableHead>Market</TableHead>
              <TableHead>Sample</TableHead>
              <TableHead>ROI</TableHead>
              <TableHead>Status</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {records.map((record) => (
              <TableRow key={`${record.sport}-${record.modelVersion}-${record.market}`}>
                <TableCell>{record.modelVersion}</TableCell>
                <TableCell>{record.market}</TableCell>
                <TableCell>{formatNumber(record.sampleSize)}</TableCell>
                <TableCell>{formatPct(record.roi)}</TableCell>
                <TableCell>{record.productionStatus}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  );
}

export default async function PerformancePage() {
  const history = await getPerformanceHistory();
  const nbaRecords = history.records.filter((record) => record.sport === "NBA");
  const nflRecords = history.records.filter((record) => record.sport === "NFL");
  const roiRecords = history.records.filter((record) => typeof record.roi === "number");
  const positiveRoi = roiRecords.filter((record) => (record.roi ?? 0) > 0).length;
  const bestRoi = roiRecords.toSorted((a, b) => (b.roi ?? -Infinity) - (a.roi ?? -Infinity))[0];
  const blocked = history.records.filter((record) => record.productionStatus === "blocked").length;
  const candidates = history.records.filter((record) => record.productionStatus === "candidate").length;
  const primarySports = SPORTS.filter((sport) => sport.emphasis === "primary");
  const secondarySports = SPORTS.filter((sport) => sport.emphasis !== "primary");

  return (
    <div>
      <PageHeader
        title="Performance"
        description="Choose a sport and market to inspect windowed graded results and persisted backtests."
      />

      <div className="grid gap-4 lg:grid-cols-3">
        {primarySports.map((sport) => <SportCard key={sport.slug} sport={sport} />)}
      </div>
      <div className="mt-4 grid gap-3 md:grid-cols-3">
        {secondarySports.map((sport) => <SportCard key={sport.slug} sport={sport} muted />)}
      </div>

      <div className="mb-4 mt-8 border-t border-border pt-6">
        <h2 className="text-xl font-semibold">Model artifacts &amp; production gates</h2>
        <p className="mt-1 text-sm text-muted-foreground">
          Existing local artifact metrics remain available alongside live grades and persisted evaluations.
        </p>
        {history.generatedAt ? <Badge variant="outline" className="mt-3">Updated {history.generatedAt}</Badge> : null}
      </div>

      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <MetricCard title="Sports" value={formatNumber(history.records.length)} detail="Performance records loaded." icon={Activity} />
        <MetricCard title="Positive ROI Records" value={formatNumber(positiveRoi)} detail="Only measured ROI values count." icon={LineChart} tone={positiveRoi ? "accent" : "default"} />
        <MetricCard title="Top ROI" value={bestRoi ? formatPct(bestRoi.roi) : "n/a"} detail={bestRoi ? `${bestRoi.sport} ${bestRoi.market}` : "Missing ROI data"} icon={BarChart3} />
        <MetricCard title="Production Blocks" value={formatNumber(blocked)} detail={`${formatNumber(candidates)} candidate records.`} icon={AlertTriangle} tone={blocked ? "warning" : "accent"} />
      </div>

      <div className="mt-4 grid gap-4 xl:grid-cols-[0.8fr_1.2fr]">
        <Card>
          <CardHeader>
            <CardTitle>ROI History</CardTitle>
          </CardHeader>
          <CardContent>
            <RoiChart records={history.records} />
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>Metrics and Coverage</CardTitle>
          </CardHeader>
          <CardContent>
            <PerformanceTables records={history.records} />
          </CardContent>
        </Card>
      </div>

      <div className="mt-4 grid gap-4 lg:grid-cols-2">
        <LeagueSection title="NBA Performance" records={nbaRecords} />
        <LeagueSection title="NFL Performance" records={nflRecords} />
      </div>
    </div>
  );
}
