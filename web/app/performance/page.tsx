import { Activity, AlertTriangle, BarChart3, LineChart } from "lucide-react";

import { MetricCard } from "@/components/dashboard/MetricCard";
import { PageHeader } from "@/components/dashboard/PageHeader";
import { PerformanceTables } from "@/components/dashboard/PerformanceTables";
import { RoiChart } from "@/components/dashboard/RoiChart";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { getPerformanceHistory } from "@/lib/data/performance";
import { formatNumber, formatPct } from "@/lib/format";

export default async function PerformancePage() {
  const history = await getPerformanceHistory();
  const roiRecords = history.records.filter((record) => typeof record.roi === "number");
  const positiveRoi = roiRecords.filter((record) => (record.roi ?? 0) > 0).length;
  const bestRoi = roiRecords.toSorted((a, b) => (b.roi ?? -Infinity) - (a.roi ?? -Infinity))[0];
  const warnings = history.records.filter((record) => record.gaps.length > 0).length;

  return (
    <div>
      <PageHeader
        title="Performance"
        description="Cross-sport model metrics, measured ROI, odds coverage status, and documented threshold/mode availability."
        meta={history.generatedAt}
      />

      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <MetricCard title="Sports" value={formatNumber(history.records.length)} detail="Performance records loaded." icon={Activity} />
        <MetricCard title="Positive ROI Records" value={formatNumber(positiveRoi)} detail="Only measured ROI values count." icon={LineChart} tone={positiveRoi ? "accent" : "default"} />
        <MetricCard title="Top ROI" value={bestRoi ? formatPct(bestRoi.roi) : "n/a"} detail={bestRoi ? `${bestRoi.sport} ${bestRoi.market}` : "Missing ROI data"} icon={BarChart3} />
        <MetricCard title="Odds Warnings" value={formatNumber(warnings)} detail="Partial or missing odds coverage." icon={AlertTriangle} tone={warnings ? "warning" : "accent"} />
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
    </div>
  );
}
