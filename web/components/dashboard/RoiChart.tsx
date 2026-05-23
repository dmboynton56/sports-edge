"use client";

import dynamic from "next/dynamic";

import { Skeleton } from "@/components/ui/skeleton";
import type { Performance } from "@/lib/data/types";

const RoiChartClient = dynamic(
  () => import("@/components/dashboard/RoiChartClient").then((mod) => mod.RoiChartClient),
  {
    ssr: false,
    loading: () => <Skeleton className="h-64 w-full" />,
  },
);

export function RoiChart({ records }: { records: Performance[] }) {
  return <RoiChartClient records={records} />;
}
