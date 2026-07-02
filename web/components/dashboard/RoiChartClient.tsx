"use client";

import { useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { Performance } from "@/lib/data/types";
import { formatPct } from "@/lib/format";

export function RoiChartClient({ records }: { records: Performance[] }) {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  const data = records
    .filter((record) => typeof record.roi === "number")
    .map((record) => ({
      sport: record.sport,
      roi: record.roi,
    }));

  if (!data.length) {
    return (
      <div className="grid h-64 place-items-center rounded-lg border border-dashed border-border text-sm text-muted-foreground">
        ROI history missing from current local artifacts.
      </div>
    );
  }

  if (!mounted) {
    return <div className="h-64 w-full min-w-0" aria-hidden />;
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
            tickFormatter={(value) => `${Math.round(Number(value) * 100)}%`}
            tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 12 }}
          />
          <Tooltip
            cursor={{ fill: "hsl(var(--secondary))" }}
            contentStyle={{
              background: "hsl(var(--popover))",
              border: "1px solid hsl(var(--border))",
              borderRadius: 8,
              color: "hsl(var(--popover-foreground))",
            }}
            formatter={(value) => [formatPct(Number(value)), "ROI"]}
          />
          <Bar dataKey="roi" fill="hsl(var(--accent))" radius={[4, 4, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
