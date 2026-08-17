import { Activity, AlertTriangle, BarChart3, LineChart } from "lucide-react";

import { ChannelCard } from "@/components/dashboard/ChannelCard";
import { MetricCard } from "@/components/dashboard/MetricCard";
import { PageHeader, SectionHeading } from "@/components/dashboard/PageHeader";
import { PerformanceTables } from "@/components/dashboard/PerformanceTables";
import { RoiChart } from "@/components/dashboard/RoiChart";
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
import { isFiniteNumber } from "@/lib/data/json";
import { formatNumber, formatPct } from "@/lib/format";
import { SPORTS } from "@/lib/markets-registry";
import type { Performance } from "@/lib/data/types";

function sportCardFor(sport: (typeof SPORTS)[number], records: Performance[]) {
  const record = records.find(
    (row) => row.sport.toLowerCase() === sport.slug.toLowerCase(),
  );
  const roi = record?.roi;

  return (
    <ChannelCard
      key={sport.slug}
      sport={sport.slug}
      href={`/performance/${sport.slug}`}
      title={sport.label}
      description={sport.description}
      figures={[
        {
          value: isFiniteNumber(roi) ? formatPct(roi) : "—",
          label: "ROI",
          tone: isFiniteNumber(roi) ? (roi < 0 ? "down" : "up") : "default",
        },
        {
          value: formatNumber(record?.sampleSize ?? 0),
          label: "Graded",
        },
      ]}
      cta={`Open ${sport.label}`}
    />
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
  const roiRecords = history.records.filter((record) => isFiniteNumber(record.roi));
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
        description="Pick a league to inspect its graded results and persisted backtests."
        meta={history.generatedAt}
      />

      <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-3">
        {primarySports.map((sport) => sportCardFor(sport, history.records))}
      </div>

      <SectionHeading title="Research and seasonal" note="Not yet in production" />
      <div className="grid gap-3 md:grid-cols-3">
        {secondarySports.map((sport) => sportCardFor(sport, history.records))}
      </div>

      <SectionHeading
        title="Model artifacts and production gates"
        note="Local artifact metrics alongside live grades"
      />

      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <MetricCard
          title="Models tracked"
          value={formatNumber(history.records.length)}
          detail="One record per sport and model version."
          icon={Activity}
        />
        <MetricCard
          title="Profitable models"
          value={formatNumber(positiveRoi)}
          detail={`Of ${formatNumber(roiRecords.length)} with measured ROI.`}
          icon={LineChart}
          tone={positiveRoi ? "accent" : "warning"}
        />
        <MetricCard
          title="Best ROI"
          value={bestRoi ? formatPct(bestRoi.roi) : "n/a"}
          detail={bestRoi ? `${bestRoi.sport} ${bestRoi.market}` : "No measured ROI yet."}
          icon={BarChart3}
          tone={bestRoi && (bestRoi.roi ?? 0) > 0 ? "accent" : "warning"}
        />
        <MetricCard
          title="Blocked from production"
          value={formatNumber(blocked)}
          detail={`${formatNumber(candidates)} more are candidates.`}
          icon={AlertTriangle}
          tone={blocked ? "warning" : "accent"}
        />
      </div>

      <div className="mt-3 grid gap-3 xl:grid-cols-[0.8fr_1.2fr]">
        <Card>
          <CardHeader>
            <CardTitle>ROI by league</CardTitle>
          </CardHeader>
          <CardContent>
            <RoiChart records={history.records} />
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>Metrics and coverage</CardTitle>
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
