"use client";

import { useEffect, useMemo, useState } from "react";
import { ArrowUpDown, SlidersHorizontal } from "lucide-react";

import { EmptyState } from "@/components/dashboard/EmptyState";
import { Notice } from "@/components/dashboard/Notice";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import type { Prediction } from "@/lib/data/types";
import { formatDateTime, formatMaybePctMetric, formatNumber, formatPct } from "@/lib/format";
import { sportColor } from "@/lib/sports";
import { cn } from "@/lib/utils";

type SortKey =
  | "sport"
  | "market"
  | "book"
  | "edge"
  | "ev"
  | "confidence"
  | "eventTime"
  | "modelVersion";

const sortLabels: Record<SortKey, string> = {
  sport: "Sport",
  market: "Market",
  book: "Book",
  edge: "Edge",
  ev: "EV",
  confidence: "Confidence",
  eventTime: "Start",
  modelVersion: "Model",
};

function valuesFor(predictions: Prediction[], key: keyof Prediction) {
  return Array.from(new Set(predictions.map((prediction) => prediction[key]).filter(Boolean))).sort() as string[];
}

function compare(a: Prediction, b: Prediction, key: SortKey, dir: "asc" | "desc") {
  const sign = dir === "asc" ? 1 : -1;
  const av = a[key as keyof Prediction];
  const bv = b[key as keyof Prediction];

  if (key === "eventTime") {
    const at = a.eventTime ? new Date(a.eventTime).getTime() : Number.POSITIVE_INFINITY;
    const bt = b.eventTime ? new Date(b.eventTime).getTime() : Number.POSITIVE_INFINITY;
    return (at - bt) * sign;
  }

  if (typeof av === "number" || typeof bv === "number") {
    return (((av as number | null) ?? -Infinity) - ((bv as number | null) ?? -Infinity)) * sign;
  }

  return String(av ?? "").localeCompare(String(bv ?? "")) * sign;
}

export function MarketsTable({
  initialPredictions,
  initialGaps,
}: {
  initialPredictions: Prediction[];
  initialGaps: string[];
}) {
  const [predictions, setPredictions] = useState(initialPredictions);
  const [gaps, setGaps] = useState(initialGaps);
  const [loading, setLoading] = useState(initialPredictions.length === 0);
  const [error, setError] = useState<string | null>(null);
  const [sport, setSport] = useState("all");
  const [market, setMarket] = useState("all");
  const [book, setBook] = useState("all");
  const [confidence, setConfidence] = useState("all");
  const [modelVersion, setModelVersion] = useState("all");
  const [sortKey, setSortKey] = useState<SortKey>("edge");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("desc");

  useEffect(() => {
    if (initialPredictions.length > 0) {
      return;
    }
    let active = true;
    fetch("/data/predictions.json", { cache: "no-store" })
      .then((response) => {
        if (!response.ok) throw new Error(`Prediction feed returned ${response.status}`);
        return response.json();
      })
      .then((payload) => {
        if (!active) return;
        const rows = Array.isArray(payload.predictions) ? payload.predictions : [];
        setPredictions(rows);
        setGaps(Array.isArray(payload.gaps) ? payload.gaps : []);
        setError(null);
      })
      .catch((err) => {
        if (!active) return;
        setError(err instanceof Error ? err.message : "Unable to load predictions");
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, [initialPredictions.length]);

  const filtered = useMemo(() => {
    return predictions
      .filter((prediction) => sport === "all" || prediction.sport === sport)
      .filter((prediction) => market === "all" || prediction.market === market)
      .filter((prediction) => book === "all" || prediction.book === book)
      .filter((prediction) => modelVersion === "all" || prediction.modelVersion === modelVersion)
      .filter((prediction) => {
        if (confidence === "all") return true;
        if (prediction.confidence == null) return false;
        if (confidence === "high") return prediction.confidence >= 0.7;
        if (confidence === "medium") return prediction.confidence >= 0.4 && prediction.confidence < 0.7;
        return prediction.confidence < 0.4;
      })
      .sort((a, b) => compare(a, b, sortKey, sortDir));
  }, [book, confidence, market, modelVersion, predictions, sortDir, sortKey, sport]);

  // The feed repeats the same caveat once per contributing source; show it once.
  const uniqueGaps = useMemo(() => Array.from(new Set(gaps)), [gaps]);

  function toggleSort(key: SortKey) {
    if (sortKey === key) {
      setSortDir((dir) => (dir === "asc" ? "desc" : "asc"));
    } else {
      setSortKey(key);
      setSortDir(key === "eventTime" ? "asc" : "desc");
    }
  }

  if (loading) {
    return (
      <div className="space-y-3">
        <div className="grid gap-3 md:grid-cols-5">
          {Array.from({ length: 5 }).map((_, index) => (
            <Skeleton key={index} className="h-9" />
          ))}
        </div>
        <Skeleton className="h-80" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="rounded-xl border border-destructive/20 bg-destructive-soft p-5 text-sm">
        <div className="font-bold text-destructive">The prediction feed didn&apos;t load</div>
        <p className="mt-2 text-destructive/90">{error}</p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
        <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-5">
          <Select value={sport} onValueChange={setSport}>
            <SelectTrigger><SelectValue placeholder="Sport" /></SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All sports</SelectItem>
              {valuesFor(predictions, "sport").map((value) => (
                <SelectItem value={value} key={value}>{value}</SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Select value={market} onValueChange={setMarket}>
            <SelectTrigger><SelectValue placeholder="Market" /></SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All markets</SelectItem>
              {valuesFor(predictions, "market").map((value) => (
                <SelectItem value={value} key={value}>{value}</SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Select value={book} onValueChange={setBook}>
            <SelectTrigger><SelectValue placeholder="Book" /></SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All books</SelectItem>
              {valuesFor(predictions, "book").map((value) => (
                <SelectItem value={value} key={value}>{value}</SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Select value={confidence} onValueChange={setConfidence}>
            <SelectTrigger><SelectValue placeholder="Confidence" /></SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All confidence</SelectItem>
              <SelectItem value="high">High</SelectItem>
              <SelectItem value="medium">Medium</SelectItem>
              <SelectItem value="low">Low</SelectItem>
            </SelectContent>
          </Select>
          <Select value={modelVersion} onValueChange={setModelVersion}>
            <SelectTrigger><SelectValue placeholder="Model" /></SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All models</SelectItem>
              {valuesFor(predictions, "modelVersion").map((value) => (
                <SelectItem value={value} key={value}>{value}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <Dialog>
          <DialogTrigger asChild>
            <Button variant="outline" size="sm">
              <SlidersHorizontal />
              Contract
            </Button>
          </DialogTrigger>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Prediction Feed Contract</DialogTitle>
              <DialogDescription>
                Rows map to sport, league, gameId, eventTime, market, book, line,
                price, modelProbability, impliedProbability, edge, EV, Kelly,
                confidence, and modelVersion. Missing fields render as n/a.
              </DialogDescription>
            </DialogHeader>
          </DialogContent>
        </Dialog>
      </div>

      <Notice
        title={`${uniqueGaps.length} ${uniqueGaps.length === 1 ? "caveat" : "caveats"} on this board`}
        items={uniqueGaps}
      />

      {filtered.length === 0 ? (
        <EmptyState
          title={predictions.length ? "Nothing matches those filters" : "Nothing on the board"}
          description={
            predictions.length
              ? "Widen a filter to bring rows back. Sport and market are the two that cut the most."
              : "No predictions have been published for today's slate. Boards fill in once the models run against a posted schedule."
          }
        />
      ) : (
        <Table className="table-fixed">
          <TableHeader>
            <TableRow>
              {(["sport", "eventTime", "market", "book", "edge", "ev", "confidence", "modelVersion"] as SortKey[]).map((key) => (
                <TableHead key={key}>
                  <Button variant="ghost" size="sm" className="h-7 px-1" onClick={() => toggleSort(key)}>
                    {sortLabels[key]}
                    <ArrowUpDown className="size-3" />
                  </Button>
                </TableHead>
              ))}
            </TableRow>
          </TableHeader>
          <TableBody>
            {filtered.map((prediction) => (
              <TableRow key={prediction.id}>
                <TableCell>
                  <div className="flex items-center gap-2">
                    <span
                      className={cn(
                        "size-1.5 shrink-0 rounded-[2px]",
                        sportColor(prediction.sport).fill,
                      )}
                    />
                    <span className="font-semibold text-foreground">{prediction.sport}</span>
                  </div>
                  <div className="mt-0.5 truncate text-xs text-muted-foreground">
                    {prediction.subject}
                  </div>
                </TableCell>
                <TableCell>{formatDateTime(prediction.eventTime)}</TableCell>
                <TableCell>{prediction.market}</TableCell>
                <TableCell>{prediction.book}</TableCell>
                <TableCell>{formatMaybePctMetric(prediction.edge)}</TableCell>
                <TableCell>{formatMaybePctMetric(prediction.ev)}</TableCell>
                <TableCell>{formatPct(prediction.confidence)}</TableCell>
                <TableCell>
                  <div>{prediction.modelVersion}</div>
                  <div className="text-xs text-muted-foreground">
                    p {formatPct(prediction.modelProbability)} / imp {formatPct(prediction.impliedProbability)}
                  </div>
                  <div className="text-xs text-muted-foreground">
                    line {formatNumber(prediction.line, 1)} · {formatNumber(prediction.price)}
                  </div>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      )}
    </div>
  );
}
