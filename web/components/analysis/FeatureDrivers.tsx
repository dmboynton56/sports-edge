"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { FeatureDriver } from "@/lib/data/explanations";

export function FeatureDrivers({ features }: { features: FeatureDriver[] }) {
  const chartData = features
    .slice(0, 10)
    .map((feature) => ({
      name: feature.feature,
      impact: feature.impact,
      value: feature.value,
    }))
    // The .map() above already produced a fresh array, so sorting in place is safe
    // here and keeps the chart rendering on browsers without Array#toSorted.
    .sort((a, b) => Math.abs(b.impact) - Math.abs(a.impact));

  if (!chartData.length) {
    return <p className="text-sm text-muted-foreground">No feature drivers available.</p>;
  }

  return (
    <div className="h-80 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={chartData} layout="vertical" margin={{ left: 24, right: 16 }}>
          <CartesianGrid strokeDasharray="3 3" horizontal={false} />
          <XAxis type="number" />
          <YAxis type="category" dataKey="name" width={160} tick={{ fontSize: 11 }} />
          <Tooltip
            formatter={(value, _name, item) => [
              `${Number(value).toFixed(3)} (value ${Number(item.payload.value).toFixed(3)})`,
              "impact",
            ]}
          />
          <Bar dataKey="impact" fill="hsl(var(--accent))" radius={[0, 4, 4, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
