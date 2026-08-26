import Link from "next/link";
import { PageHeader, SectionHeading } from "@/components/dashboard/PageHeader";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { SportSwatch } from "@/components/dashboard/SportChip";
import { getPerformanceHistory } from "@/lib/data/performance";
import { isFiniteNumber } from "@/lib/data/json";
import { formatNumber, formatPct } from "@/lib/format";

export const dynamic = "force-dynamic";

export default async function RecordPage() {
  const history = await getPerformanceHistory();

  return (
    <div>
      <PageHeader
        title="Record"
        description="Backtest results, performance history, and official graded outcomes. Both live board health and historical ROI live here."
      />

      <SectionHeading title="Season performance" note="Backtest to date" />

      <Card className="overflow-hidden">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Sport</TableHead>
              <TableHead className="hidden sm:table-cell">Model</TableHead>
              <TableHead>Market</TableHead>
              <TableHead className="text-right">Sample</TableHead>
              <TableHead className="text-right">ROI</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {history.records.map((record) => {
              const roi = record.roi;
              return (
                <TableRow key={`${record.sport}-${record.modelVersion}`}>
                  <TableCell><SportSwatch sport={record.sport} label={record.sport} /></TableCell>
                  <TableCell className="hidden sm:table-cell">{record.modelVersion}</TableCell>
                  <TableCell>{record.market}</TableCell>
                  <TableCell className="text-right">{formatNumber(record.sampleSize)}</TableCell>
                  <TableCell className={isFiniteNumber(roi) ? `figure text-right text-[17px] ${roi < 0 ? "text-destructive" : "text-positive"}` : "text-right text-sm"}>
                    {isFiniteNumber(roi) ? formatPct(roi) : "No odds history"}
                  </TableCell>
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      </Card>

      <div className="mt-8 flex gap-3">
        <Button asChild variant="outline">
          <Link href="/results">View detailed results</Link>
        </Button>
        <Button asChild variant="outline">
          <Link href="/performance">Performance by sport</Link>
        </Button>
      </div>
    </div>
  );
}
