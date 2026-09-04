"use client";

import Link from "next/link";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { ArrowDown, ArrowUp, ChevronLeft, ChevronRight } from "lucide-react";
import { useMemo } from "react";

import { EmptyState } from "@/components/dashboard/EmptyState";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import type { Prediction } from "@/lib/data/types";
import { isFiniteNumber } from "@/lib/data/json";
import { formatAmericanPrice, formatDateTime, formatNumber, formatPct } from "@/lib/format";
import {
  filterAndSortMarketRows,
  MARKET_TABLE_PAGE_SIZE,
  paginateMarketRows,
  readMarketTableState,
  updateMarketTableQuery,
  type MarketTableSortKey,
} from "@/lib/markets-table";
import { sportColor } from "@/lib/sports";
import { cn } from "@/lib/utils";

const SORT_LABELS = {
  subject: "Pick",
  eventTime: "Start",
  market: "Market",
  price: "Price",
  modelProbability: "Probability",
  edge: "Edge",
  ev: "EV",
  marketStatus: "Status",
} satisfies Record<MarketTableSortKey, string>;

const COLUMN_KEYS = [
  "subject",
  "eventTime",
  "market",
  "price",
  "modelProbability",
  "edge",
  "ev",
  "marketStatus",
] satisfies MarketTableSortKey[];

const STATUS_LABELS = {
  supported: "Supported",
  research: "Research",
  model_only: "Model only",
} satisfies Record<Prediction["marketStatus"], string>;

function valuesFor(
  predictions: Prediction[],
  key: "sport" | "market" | "book",
  selected: string,
) {
  const values = predictions.map((prediction) => prediction[key]).filter(Boolean);
  if (selected !== "all") values.push(selected);
  return Array.from(new Set(values)).sort();
}

function statusVariant(status: Prediction["marketStatus"]) {
  if (status === "supported") return "positive";
  if (status === "research") return "warning";
  return "outline";
}

function WarningsDisclosure({ warnings }: { warnings: string[] }) {
  const uniqueWarnings = Array.from(new Set(warnings));
  return (
    <details className="group rounded-xl border border-border bg-secondary/25 px-4 py-3">
      <summary className="cursor-pointer list-none text-sm font-semibold text-secondary-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring [&::-webkit-details-marker]:hidden">
        <span className="inline-flex items-center gap-2">
          <span className="text-muted-foreground transition-transform group-open:rotate-90">›</span>
          Warnings ({uniqueWarnings.length})
        </span>
      </summary>
      {uniqueWarnings.length ? (
        <ul className="mt-3 space-y-2 border-t border-border pt-3 text-sm leading-relaxed text-muted-foreground">
          {uniqueWarnings.map((warning) => <li key={warning}>{warning}</li>)}
        </ul>
      ) : (
        <p className="mt-3 border-t border-border pt-3 text-sm text-muted-foreground">
          No feed gaps or withheld rows were reported.
        </p>
      )}
    </details>
  );
}

export function MarketsTable({
  initialPredictions,
  initialGaps,
  defaultSortKey = "ev",
  defaultSortDir = "desc",
  emptyTitle = "Nothing on the board",
  emptyDescription = "No upcoming publication-eligible predictions are available right now.",
}: {
  initialPredictions: Prediction[];
  initialGaps: string[];
  defaultSortKey?: MarketTableSortKey;
  defaultSortDir?: "asc" | "desc";
  emptyTitle?: string;
  emptyDescription?: string;
}) {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const searchString = searchParams.toString();
  const state = useMemo(() => {
    const parsed = readMarketTableState(new URLSearchParams(searchString));
    return {
      ...parsed,
      sort: searchParams.has("sort") ? parsed.sort : defaultSortKey,
      dir: searchParams.has("dir") ? parsed.dir : defaultSortDir,
    };
  }, [defaultSortDir, defaultSortKey, searchParams, searchString]);
  const filtered = useMemo(
    () => filterAndSortMarketRows(initialPredictions, state),
    [initialPredictions, state],
  );
  const pageCount = Math.max(1, Math.ceil(filtered.length / MARKET_TABLE_PAGE_SIZE));
  const page = Math.min(state.page, pageCount);
  const pageStart = (page - 1) * MARKET_TABLE_PAGE_SIZE;
  const visiblePredictions = paginateMarketRows(filtered, page);

  function updateQuery(
    updates: Parameters<typeof updateMarketTableQuery>[1],
    resetPage = false,
  ) {
    const query = updateMarketTableQuery(new URLSearchParams(searchString), updates, resetPage);
    router.replace(query ? `${pathname}?${query}` : pathname, { scroll: false });
  }

  function toggleSort(key: MarketTableSortKey) {
    const direction = state.sort === key && state.dir === "desc" ? "asc" : "desc";
    updateQuery({ sort: key, dir: direction }, true);
  }

  return (
    <div className="space-y-4">
      <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-5">
        <Select value={state.sport} onValueChange={(value) => updateQuery({ sport: value }, true)}>
          <SelectTrigger aria-label="Sport filter"><SelectValue placeholder="Sport" /></SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All sports</SelectItem>
            {valuesFor(initialPredictions, "sport", state.sport).map((value) => (
              <SelectItem value={value} key={value}>{value}</SelectItem>
            ))}
          </SelectContent>
        </Select>
        <Select value={state.market} onValueChange={(value) => updateQuery({ market: value }, true)}>
          <SelectTrigger aria-label="Market filter"><SelectValue placeholder="Market" /></SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All markets</SelectItem>
            {valuesFor(initialPredictions, "market", state.market).map((value) => (
              <SelectItem value={value} key={value}>{value}</SelectItem>
            ))}
          </SelectContent>
        </Select>
        <Select value={state.book} onValueChange={(value) => updateQuery({ book: value }, true)}>
          <SelectTrigger aria-label="Book filter"><SelectValue placeholder="Book" /></SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All books</SelectItem>
            {valuesFor(initialPredictions, "book", state.book).map((value) => (
              <SelectItem value={value} key={value}>{value}</SelectItem>
            ))}
          </SelectContent>
        </Select>
        <Select value={state.probability} onValueChange={(value) => updateQuery({ probability: value }, true)}>
          <SelectTrigger aria-label="Minimum probability filter"><SelectValue placeholder="Minimum probability" /></SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All probabilities</SelectItem>
            <SelectItem value="10">10%+</SelectItem>
            <SelectItem value="25">25%+</SelectItem>
            <SelectItem value="50">50%+</SelectItem>
            <SelectItem value="75">75%+</SelectItem>
          </SelectContent>
        </Select>
        <Select value={state.status} onValueChange={(value) => updateQuery({ status: value }, true)}>
          <SelectTrigger aria-label="Status filter"><SelectValue placeholder="Status" /></SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All statuses</SelectItem>
            <SelectItem value="supported">Supported</SelectItem>
            <SelectItem value="research">Research</SelectItem>
            <SelectItem value="model_only">Model only</SelectItem>
          </SelectContent>
        </Select>
      </div>

      <WarningsDisclosure warnings={initialGaps} />

      {filtered.length === 0 ? (
        <EmptyState
          title={initialPredictions.length ? "Nothing matches those filters" : emptyTitle}
          description={initialPredictions.length ? "Widen a filter to bring predictions back." : emptyDescription}
        />
      ) : (
        <>
          <Table className="min-w-[1060px]">
            <TableHeader>
              <TableRow>
                {COLUMN_KEYS.map((key) => (
                  <TableHead
                    key={key}
                    aria-sort={state.sort === key ? (state.dir === "asc" ? "ascending" : "descending") : "none"}
                  >
                    <Button
                      variant="ghost"
                      size="sm"
                      className="h-7 px-1"
                      onClick={() => toggleSort(key)}
                    >
                      {SORT_LABELS[key]}
                      {state.sort === key
                        ? state.dir === "asc" ? <ArrowUp /> : <ArrowDown />
                        : null}
                    </Button>
                  </TableHead>
                ))}
              </TableRow>
            </TableHeader>
            <TableBody>
              {visiblePredictions.map((prediction) => (
                <TableRow key={prediction.id}>
                  <TableCell className="min-w-64">
                    <div className="flex items-center gap-2">
                      <span className={cn("size-1.5 shrink-0 rounded-[2px]", sportColor(prediction.sport).fill)} />
                      {prediction.detailHref ? (
                        <Link href={prediction.detailHref} className="font-semibold text-foreground hover:underline">
                          {prediction.subject}
                        </Link>
                      ) : (
                        <span className="font-semibold text-foreground">{prediction.subject}</span>
                      )}
                    </div>
                    <div className="mt-1 max-w-64 truncate text-xs text-muted-foreground">
                      {prediction.sport} · {prediction.modelVersion}
                    </div>
                    {prediction.source ? (
                      <div className="max-w-64 truncate text-[11px] text-muted-foreground/80" title={prediction.source}>
                        {prediction.source}
                      </div>
                    ) : null}
                  </TableCell>
                  <TableCell className="whitespace-nowrap">{formatDateTime(prediction.eventTime)}</TableCell>
                  <TableCell>{prediction.market.replaceAll("_", " ")}</TableCell>
                  <TableCell>
                    <div className="font-medium text-foreground">{prediction.book}</div>
                    <div className="whitespace-nowrap text-xs text-muted-foreground">
                      {isFiniteNumber(prediction.line) ? `line ${formatNumber(prediction.line, 1)} · ` : ""}
                      {formatAmericanPrice(prediction.price)}
                    </div>
                  </TableCell>
                  <TableCell>
                    <div className="figure text-base text-foreground">{formatPct(prediction.modelProbability)}</div>
                    <div className="text-xs text-muted-foreground">market {formatPct(prediction.impliedProbability)}</div>
                  </TableCell>
                  <TableCell className={cn("figure", isFiniteNumber(prediction.edge) && prediction.edge < 0 ? "text-destructive" : "text-foreground")}>
                    {formatPct(prediction.edge)}
                  </TableCell>
                  <TableCell className={cn("figure", isFiniteNumber(prediction.ev) && prediction.ev < 0 ? "text-destructive" : "text-foreground")}>
                    {formatPct(prediction.ev)}
                  </TableCell>
                  <TableCell>
                    <Badge variant={statusVariant(prediction.marketStatus)}>
                      {STATUS_LABELS[prediction.marketStatus]}
                    </Badge>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>

          <div className="flex flex-col gap-3 border-t border-border pt-4 text-sm text-muted-foreground sm:flex-row sm:items-center sm:justify-between">
            <span>
              Showing {formatNumber(pageStart + 1)}–{formatNumber(pageStart + visiblePredictions.length)} of {formatNumber(filtered.length)} filtered rows · {formatNumber(initialPredictions.length)} total
            </span>
            <div className="flex items-center gap-2">
              <Button
                variant="outline"
                size="sm"
                aria-label="Previous page"
                disabled={page <= 1}
                onClick={() => updateQuery({ page: page - 1 })}
              >
                <ChevronLeft />
                Previous
              </Button>
              <span className="min-w-20 text-center text-xs">Page {page} of {pageCount}</span>
              <Button
                variant="outline"
                size="sm"
                aria-label="Next page"
                disabled={page >= pageCount}
                onClick={() => updateQuery({ page: page + 1 })}
              >
                Next
                <ChevronRight />
              </Button>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
