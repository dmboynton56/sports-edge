import type { LucideIcon } from "lucide-react";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";

export function MetricCard({
  title,
  value,
  detail,
  icon: Icon,
  tone = "default",
}: {
  title: string;
  value: string;
  detail?: string;
  icon?: LucideIcon;
  tone?: "default" | "accent" | "warning";
}) {
  return (
    <Card className="min-h-32">
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-3">
        <CardTitle className="text-sm text-muted-foreground">{title}</CardTitle>
        {Icon ? (
          <Icon
            className={cn(
              "size-4 text-muted-foreground",
              tone === "accent" && "text-accent",
              tone === "warning" && "text-destructive",
            )}
          />
        ) : null}
      </CardHeader>
      <CardContent>
        <div className="text-2xl font-semibold tracking-normal">{value}</div>
        {detail ? <p className="mt-2 text-xs text-muted-foreground">{detail}</p> : null}
      </CardContent>
    </Card>
  );
}
