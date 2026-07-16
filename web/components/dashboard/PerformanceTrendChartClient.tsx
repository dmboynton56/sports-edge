"use client";

import {
  Bar,
  CartesianGrid,
  ComposedChart,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { formatMaybePctMetric, formatNumber, formatPct } from "@/lib/format";

export type PerformanceTrendPoint = {
  label: string;
  hitRate?: number | null;
  units?: number | null;
  metric?: number | null;
};

export function PerformanceTrendChartClient({
  data,
  mode,
  metricLabel = "Metric",
}: {
  data: PerformanceTrendPoint[];
  mode: "weekly" | "metric";
  metricLabel?: string;
}) {
  if (!data.length) {
    return (
      <div className="grid h-64 place-items-center rounded-lg border border-dashed border-border text-sm text-muted-foreground">
        No chartable history is available.
      </div>
    );
  }

  return (
    <div className="h-64 w-full min-w-0">
      <ResponsiveContainer width="100%" height="100%" minWidth={1} minHeight={256}>
        <ComposedChart data={data} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
          <CartesianGrid stroke="hsl(var(--border))" vertical={false} />
          <XAxis
            dataKey="label"
            axisLine={false}
            tickLine={false}
            tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 11 }}
          />
          <YAxis
            yAxisId="rate"
            axisLine={false}
            tickLine={false}
            tickFormatter={(value) => formatMaybePctMetric(Number(value), 0)}
            tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 11 }}
          />
          {mode === "weekly" ? (
            <YAxis
              yAxisId="units"
              orientation="right"
              axisLine={false}
              tickLine={false}
              tickFormatter={(value) => formatNumber(Number(value), 1)}
              tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 11 }}
            />
          ) : null}
          <Tooltip
            contentStyle={{
              background: "hsl(var(--popover))",
              border: "1px solid hsl(var(--border))",
              borderRadius: 8,
              color: "hsl(var(--popover-foreground))",
            }}
            formatter={(value, name) => {
              const numeric = Number(value);
              if (name === "Hit rate") return [formatPct(numeric), name];
              if (name === "Units") return [formatNumber(numeric, 2), name];
              return [formatMaybePctMetric(numeric, 2), metricLabel];
            }}
          />
          {mode === "weekly" ? (
            <>
              <Bar yAxisId="units" dataKey="units" name="Units" fill="hsl(var(--secondary))" radius={[3, 3, 0, 0]} />
              <Line yAxisId="rate" dataKey="hitRate" name="Hit rate" stroke="hsl(var(--accent))" strokeWidth={2} dot={false} connectNulls />
            </>
          ) : (
            <Line yAxisId="rate" dataKey="metric" name={metricLabel} stroke="hsl(var(--accent))" strokeWidth={2} dot={{ r: 3 }} connectNulls />
          )}
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
