"use client";

import dynamic from "next/dynamic";

import { Skeleton } from "@/components/ui/skeleton";
import type { PerformanceTrendPoint } from "@/components/dashboard/PerformanceTrendChartClient";

const PerformanceTrendChartClient = dynamic(
  () => import("@/components/dashboard/PerformanceTrendChartClient").then(
    (mod) => mod.PerformanceTrendChartClient,
  ),
  {
    ssr: false,
    loading: () => <Skeleton className="h-64 w-full" />,
  },
);

export function PerformanceTrendChart({
  data,
  mode,
  metricLabel,
}: {
  data: PerformanceTrendPoint[];
  mode: "weekly" | "metric";
  metricLabel?: string;
}) {
  return <PerformanceTrendChartClient data={data} mode={mode} metricLabel={metricLabel} />;
}
