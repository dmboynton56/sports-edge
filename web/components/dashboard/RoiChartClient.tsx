"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { Performance } from "@/lib/data/types";
import { isFiniteNumber } from "@/lib/data/json";
import { formatPct } from "@/lib/format";
import { sportVar } from "@/lib/sports";

export function RoiChartClient({ records }: { records: Performance[] }) {
  const data = records.flatMap((record) =>
    isFiniteNumber(record.roi) ? [{ sport: record.sport, roi: record.roi }] : [],
  );

  // Every ROI is negative today. Pin zero into the domain so the bars hang from
  // a real baseline instead of collapsing into hairlines at the top of the plot.
  const values = data.map((entry) => entry.roi);
  const low = Math.min(0, ...values);
  const high = Math.max(0, ...values);
  const pad = (high - low) * 0.15 || 0.01;
  const domain: [number, number] = [low - pad, high + pad];

  if (!data.length) {
    return (
      <div className="grid h-64 place-items-center rounded-xl border border-dashed border-border text-sm text-muted-foreground">
        ROI history missing from current local artifacts.
      </div>
    );
  }

  return (
    <div className="h-64 w-full min-w-0">
      <ResponsiveContainer width="100%" height="100%" minWidth={1} minHeight={256}>
        <BarChart data={data} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
          <CartesianGrid stroke="hsl(var(--border))" vertical={false} />
          <XAxis
            dataKey="sport"
            axisLine={false}
            tickLine={false}
            tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 12 }}
          />
          <YAxis
            axisLine={false}
            tickLine={false}
            domain={domain}
            tickFormatter={(value) => `${Math.round(Number(value) * 100)}%`}
            tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 12 }}
          />
          <ReferenceLine y={0} stroke="hsl(var(--muted-foreground))" strokeOpacity={0.5} />
          <Tooltip
            cursor={{ fill: "hsl(var(--secondary))" }}
            contentStyle={{
              background: "hsl(var(--popover))",
              border: "1px solid hsl(var(--border))",
              borderRadius: 12,
              color: "hsl(var(--popover-foreground))",
            }}
            formatter={(value) => [formatPct(Number(value)), "ROI"]}
          />
          {/* Bars carry each league's color, the same one used in chips and tables. */}
          <Bar dataKey="roi" radius={[4, 4, 0, 0]} isAnimationActive={false}>
            {data.map((entry) => (
              <Cell key={entry.sport} fill={sportVar(entry.sport)} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
