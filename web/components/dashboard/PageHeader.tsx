import Link from "next/link";
import type { ReactNode } from "react";

import { formatDateTime } from "@/lib/format";

export function PageHeader({
  title,
  description,
  meta,
  actions,
}: {
  title: string;
  description: string;
  meta?: string | null;
  actions?: ReactNode;
}) {
  // `meta` can be a composed string ("<iso> · model · source"); only the leading
  // timestamp is formattable, and an unparseable one shouldn't render at all.
  const [stamp, ...rest] = (meta ?? "").split(" · ");
  const when = formatDateTime(stamp);
  const suffix = rest.join(" · ");

  return (
    <div className="mb-7 flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
      <div>
        <h1 className="font-display text-3xl font-bold tracking-tight sm:text-4xl">
          {title}
        </h1>
        <p className="mt-3 max-w-2xl text-[15px] leading-relaxed text-muted-foreground">
          {description}
        </p>
      </div>
      <div className="flex flex-wrap items-center gap-2">
        {actions}
        {when !== "n/a" ? (
          <span className="inline-flex items-center gap-2 rounded-full border border-border bg-card px-3 py-1.5 text-xs font-semibold text-muted-foreground shadow-soft">
            <span className="size-1.5 rounded-full bg-positive" />
            Updated {when}
            {suffix ? <span className="font-normal opacity-70">· {suffix}</span> : null}
          </span>
        ) : null}
      </div>
    </div>
  );
}

/** Divides a page into named stretches without adding another box. */
export function SectionHeading({
  title,
  note,
  action,
}: {
  title: string;
  note?: string;
  action?: { label: string; href: string };
}) {
  return (
    <div className="mb-4 mt-12 flex items-baseline gap-3 first:mt-0">
      <h2 className="font-display text-xl font-bold tracking-tight">{title}</h2>
      {note ? <span className="text-[13px] text-muted-foreground">{note}</span> : null}
      {action ? (
        <Link
          href={action.href}
          className="ml-auto text-[13px] font-semibold text-accent hover:underline"
        >
          {action.label} →
        </Link>
      ) : null}
    </div>
  );
}
