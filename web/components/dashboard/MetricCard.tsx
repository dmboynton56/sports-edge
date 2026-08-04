import type { LucideIcon } from "lucide-react";

import { Card } from "@/components/ui/card";
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
    <Card className="flex min-h-32 flex-col p-5">
      <div className="flex items-start justify-between gap-3">
        <span className="text-[13px] font-semibold text-muted-foreground">{title}</span>
        {Icon ? (
          <Icon
            className={cn(
              "size-4 shrink-0 text-muted-foreground",
              tone === "accent" && "text-accent",
              tone === "warning" && "text-destructive",
            )}
          />
        ) : null}
      </div>
      <div
        className={cn(
          "figure mt-2 text-[32px] leading-tight",
          tone === "warning" && "text-destructive",
        )}
      >
        {value}
      </div>
      {detail ? (
        <p className="mt-2 text-xs leading-relaxed text-muted-foreground">{detail}</p>
      ) : null}
    </Card>
  );
}
