import Link from "next/link";
import type { ReactNode } from "react";
import { ArrowRight } from "lucide-react";

import { SportChip } from "@/components/dashboard/SportChip";
import type { StatTone } from "@/components/dashboard/StatTile";
import { sportColor } from "@/lib/sports";
import { cn } from "@/lib/utils";

const TONE = {
  default: "",
  up: "text-positive",
  down: "text-destructive",
} satisfies Record<StatTone, string>;

export type ChannelChip = { sport: string; label: string; muted?: boolean };
export type ChannelFigure = { value: string; label: string; tone?: StatTone };

/**
 * The overview's primary navigation unit: a surface you can enter, with just
 * enough live data to tell you whether it's worth entering.
 */
export function ChannelCard({
  href,
  title,
  description,
  cta,
  chips,
  figures,
  sport,
  badge,
  className,
}: {
  href: string;
  title: string;
  description: string;
  cta: string;
  chips?: ChannelChip[];
  figures?: ChannelFigure[];
  /** Anchors the card to a league, matching chips and table swatches. */
  sport?: string;
  badge?: ReactNode;
  className?: string;
}) {
  return (
    <Link
      href={href}
      className={cn(
        "group flex flex-col rounded-xl border border-border bg-card p-6 shadow-soft transition-all duration-200",
        "hover:-translate-y-0.5 hover:border-muted-foreground/25 hover:shadow-lift",
        className,
      )}
    >
      <div className="flex items-center gap-3">
        <h3 className="font-display text-[22px] font-bold tracking-tight">{title}</h3>
        {badge ? <span className="ml-auto">{badge}</span> : null}
      </div>
      <p className="mt-2 flex-1 text-sm leading-relaxed text-muted-foreground">
        {description}
      </p>

      {chips?.length ? (
        <div className="mt-4 flex flex-wrap gap-1.5">
          {chips.map((chip) => (
            <SportChip
              key={`${chip.sport}-${chip.label}`}
              sport={chip.sport}
              label={chip.label}
              muted={chip.muted}
            />
          ))}
        </div>
      ) : null}

      {figures?.length ? (
        <div className="mt-4 flex gap-7">
          {figures.map((figure) => (
            <div key={figure.label}>
              <div className={cn("figure text-[26px] leading-tight", TONE[figure.tone ?? "default"])}>
                {figure.value}
              </div>
              <div className="text-[11px] font-medium text-muted-foreground">
                {figure.label}
              </div>
            </div>
          ))}
        </div>
      ) : null}

      <span className="mt-5 inline-flex items-center gap-2 text-sm font-semibold text-secondary-foreground transition-colors group-hover:text-accent">
        {cta}
        <ArrowRight className="size-4 transition-transform group-hover:translate-x-1" />
      </span>
    </Link>
  );
}
