"use client";

import { useEffect, useMemo, useState } from "react";
import { ArrowDownUp, Check, ChevronDown, ChevronUp, RotateCcw, Settings2, Sparkles, Users, X } from "lucide-react";

import {
  DEFAULT_FANTASY_ROSTER,
  DEFAULT_FANTASY_SCORING,
  HALF_PPR_SCORING,
  STANDARD_SCORING,
  type FantasyFeed,
  type FantasyPosition,
  type FantasyProjection,
  type FantasyRoster,
  type FantasyScoring,
  rescoreProjection,
} from "@/lib/data/fantasy";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { cn } from "@/lib/utils";

type View = "rankings" | "weekly" | "draft" | "lineup";
type SortKey = "points" | "ppg" | "floor" | "adp" | "position";

const POSITIONS: Array<"ALL" | FantasyPosition> = ["ALL", "QB", "RB", "WR", "TE", "K", "DST"];
const SCORING_FIELDS: Array<{ key: keyof FantasyScoring; label: string; step: string }> = [
  { key: "reception", label: "Reception", step: "0.5" },
  { key: "passing_yards", label: "Pass yard", step: "0.01" },
  { key: "passing_td", label: "Pass TD", step: "0.5" },
  { key: "interception", label: "Interception", step: "0.5" },
  { key: "rushing_yards", label: "Rush yard", step: "0.01" },
  { key: "rushing_td", label: "Rush TD", step: "0.5" },
  { key: "receiving_yards", label: "Rec yard", step: "0.01" },
  { key: "receiving_td", label: "Rec TD", step: "0.5" },
  { key: "fumble_lost", label: "Fumble lost", step: "0.5" },
  { key: "two_point_conversion", label: "2-point", step: "0.5" },
];

type DisplayProjection = FantasyProjection & { displayPoints: number; displayFloor: number; displayCeiling: number };

function positionValue(position: FantasyPosition) {
  return position === "QB" ? 0 : position === "RB" ? 1 : position === "WR" ? 2 : position === "TE" ? 3 : position === "K" ? 4 : 5;
}

function replacementRank(position: FantasyPosition, roster: FantasyRoster) {
  const demand = position === "QB" ? roster.quarterback : position === "RB" ? roster.running_back + roster.flex : position === "WR" ? roster.wide_receiver + roster.flex : position === "TE" ? roster.tight_end + roster.flex : position === "K" ? roster.kicker : roster.defense;
  return Math.max(1, roster.teams * demand + 1);
}

function draftRecommendation(rows: DisplayProjection[], drafted: Set<string>, roster: FantasyRoster) {
  const available = rows.filter((row) => !drafted.has(row.player_id));
  const best = available
    .map((row) => ({
      row,
      value: row.displayPoints - (rows.filter((candidate) => candidate.position === row.position).sort((a, b) => b.displayPoints - a.displayPoints)[replacementRank(row.position, roster) - 1]?.displayPoints ?? 0),
    }))
    .sort((a, b) => b.value - a.value || a.row.displayPoints - b.row.displayPoints)[0];
  return best?.row ?? null;
}

function slotEligible(position: FantasyPosition, slot: string) {
  return slot === "FLEX" ? ["RB", "WR", "TE"].includes(position) : position === slot || (slot === "DST" && position === "DST");
}

function optimizeLineup(rows: DisplayProjection[], selected: Set<string>, roster: FantasyRoster) {
  const available = rows.filter((row) => selected.has(row.player_id) && row.availability !== "out" && row.availability !== "bye");
  const lineup: Array<{ slot: string; row: DisplayProjection }> = [];
  const used = new Set<string>();
  const fixedSlots = [
    ...Array(roster.quarterback).fill("QB"),
    ...Array(roster.running_back).fill("RB"),
    ...Array(roster.wide_receiver).fill("WR"),
    ...Array(roster.tight_end).fill("TE"),
    ...Array(roster.kicker).fill("K"),
    ...Array(roster.defense).fill("DST"),
  ];
  for (const slot of fixedSlots) {
    const candidate = available.filter((row) => !used.has(row.player_id) && slotEligible(row.position, slot)).sort((a, b) => b.displayPoints - a.displayPoints)[0];
    if (candidate) {
      lineup.push({ slot, row: candidate });
      used.add(candidate.player_id);
    }
  }
  for (let index = 0; index < roster.flex; index += 1) {
    const candidate = available.filter((row) => !used.has(row.player_id) && slotEligible(row.position, "FLEX")).sort((a, b) => b.displayPoints - a.displayPoints)[0];
    if (candidate) {
      lineup.push({ slot: "FLEX", row: candidate });
      used.add(candidate.player_id);
    }
  }
  return { lineup, total: lineup.reduce((sum, item) => sum + item.row.displayPoints, 0) };
}

function readPlannerState() {
  try {
    const raw = window.localStorage.getItem("sports-edge-fantasy-planner-v1");
    return raw ? JSON.parse(raw) as { scoring?: FantasyScoring; roster?: FantasyRoster; drafted?: string[]; mine?: string[] } : {};
  } catch {
    return {};
  }
}

export function FantasyBoard({ feed: initialFeed }: { feed: FantasyFeed }) {
  const [feed, setFeed] = useState<FantasyFeed>(initialFeed);
  const [view, setView] = useState<View>("rankings");
  const [week, setWeek] = useState(1);
  const [position, setPosition] = useState<"ALL" | FantasyPosition>("ALL");
  const [search, setSearch] = useState("");
  const [sortKey, setSortKey] = useState<SortKey>("points");
  const [ascending, setAscending] = useState(false);
  const [scoring, setScoring] = useState<FantasyScoring>(DEFAULT_FANTASY_SCORING);
  const [roster, setRoster] = useState<FantasyRoster>(DEFAULT_FANTASY_ROSTER);
  const [drafted, setDrafted] = useState<Set<string>>(new Set());
  const [mine, setMine] = useState<Set<string>>(new Set());
  const [showSettings, setShowSettings] = useState(false);
  const [hydrated, setHydrated] = useState(false);
  const [weeklyLoading, setWeeklyLoading] = useState(false);
  const [weeklyError, setWeeklyError] = useState<string | null>(null);

  useEffect(() => {
    const frame = window.requestAnimationFrame(() => {
      const state = readPlannerState();
      if (state.scoring) setScoring({ ...DEFAULT_FANTASY_SCORING, ...state.scoring });
      if (state.roster) setRoster({ ...DEFAULT_FANTASY_ROSTER, ...state.roster });
      if (state.drafted) setDrafted(new Set(state.drafted));
      if (state.mine) setMine(new Set(state.mine));
      setHydrated(true);
    });
    return () => window.cancelAnimationFrame(frame);
  }, []);

  useEffect(() => {
    if (!hydrated) return;
    window.localStorage.setItem("sports-edge-fantasy-planner-v1", JSON.stringify({ scoring, roster, drafted: [...drafted], mine: [...mine] }));
  }, [drafted, hydrated, mine, roster, scoring]);

  useEffect(() => {
    if (view !== "weekly" && view !== "lineup") return;
    const weekKey = String(week);
    if (feed.weekly[weekKey]?.length) return;
    const controller = new AbortController();
    const frame = window.requestAnimationFrame(() => {
      setWeeklyLoading(true);
      setWeeklyError(null);
      fetch(`/api/fantasy/projections?scope=week&week=${week}`, { signal: controller.signal, cache: "no-store" })
        .then(async (response) => {
          if (!response.ok) throw new Error(`Weekly feed returned ${response.status}.`);
          return response.json() as Promise<{ projections?: FantasyProjection[]; gaps?: string[] }>;
        })
        .then((payload) => {
          setFeed((current) => ({
            ...current,
            weekly: { ...current.weekly, [weekKey]: payload.projections ?? [] },
            gaps: [...new Set([...current.gaps, ...(payload.gaps ?? [])])],
          }));
        })
        .catch((error: unknown) => {
          if (error instanceof DOMException && error.name === "AbortError") return;
          setWeeklyError(error instanceof Error ? error.message : "Unable to load weekly projections.");
        })
        .finally(() => setWeeklyLoading(false));
    });
    return () => {
      window.cancelAnimationFrame(frame);
      controller.abort();
    };
  }, [feed.weekly, view, week]);

  const sourceRows = useMemo<FantasyProjection[]>(() => {
    if (view === "rankings" || view === "draft") return feed.projections;
    const preseasonById = new Map(feed.projections.map((row) => [row.player_id, row]));
    const weeklyRows = feed.weekly[String(week)] ?? [];
    return weeklyRows.map((weeklyRow) => {
      const base = preseasonById.get(weeklyRow.player_id);
      if (!base || Object.keys(weeklyRow.statline ?? {}).length) return weeklyRow;
      const multiplier = 1 / Math.max(1, base.projected_games);
      const scale = (line: Record<string, number> | undefined) => Object.fromEntries(Object.entries(line ?? {}).map(([key, value]) => [key, Number(value) * multiplier]));
      return {
        ...base,
        ...weeklyRow,
        statline: scale(base.statline),
        statline_low: scale(base.statline_low),
        statline_high: scale(base.statline_high),
        projected_games: weeklyRow.projected_games ?? 1,
      };
    });
  }, [feed, view, week]);
  const rows = useMemo<DisplayProjection[]>(() => sourceRows.map((row) => {
    const scored = rescoreProjection(row, scoring);
    return { ...row, displayPoints: scored.median, displayFloor: scored.floor, displayCeiling: scored.ceiling };
  }), [scoring, sourceRows]);
  const filtered = useMemo(() => rows.filter((row) => (position === "ALL" || row.position === position) && row.player_name.toLowerCase().includes(search.toLowerCase())).sort((a, b) => {
    const aValue = sortKey === "ppg" ? a.displayPoints / Math.max(1, a.projected_games) : sortKey === "floor" ? a.displayFloor : sortKey === "adp" ? (a.adp ?? Infinity) : sortKey === "position" ? positionValue(a.position) : a.displayPoints;
    const bValue = sortKey === "ppg" ? b.displayPoints / Math.max(1, b.projected_games) : sortKey === "floor" ? b.displayFloor : sortKey === "adp" ? (b.adp ?? Infinity) : sortKey === "position" ? positionValue(b.position) : b.displayPoints;
    return (ascending ? 1 : -1) * (aValue - bValue) || a.player_name.localeCompare(b.player_name);
  }), [position, rows, search, sortKey, ascending]);
  const recommendation = useMemo(() => draftRecommendation(rows, drafted, roster), [drafted, roster, rows]);
  const lineup = useMemo(() => optimizeLineup(rows, mine, roster), [mine, roster, rows]);
  const topByPosition = useMemo(() => POSITIONS.slice(1).map((item) => ({ position: item, row: rows.filter((row) => row.position === item).sort((a, b) => b.displayPoints - a.displayPoints)[0] })).filter((item) => item.row), [rows]);
  const validation = useMemo(() => {
    const rawTargets = (feed.metrics as { targets?: unknown }).targets;
    if (!rawTargets || typeof rawTargets !== "object") return null;
    const checks = Object.values(rawTargets as Record<string, Record<string, { beats_baseline?: boolean }>>).flatMap((group) => Object.values(group));
    return {
      holdout: Number((feed.metrics as { holdout_season?: unknown }).holdout_season ?? 0),
      beats: checks.filter((check) => check.beats_baseline).length,
      total: checks.length,
    };
  }, [feed.metrics]);

  function setPreset(next: FantasyScoring) {
    setScoring({ ...next });
  }

  function toggleDraft(row: DisplayProjection, own: boolean) {
    setDrafted((current) => {
      const next = new Set(current);
      if (next.has(row.player_id) && (!own || mine.has(row.player_id))) next.delete(row.player_id); else next.add(row.player_id);
      return next;
    });
    if (own) setMine((current) => {
      const next = new Set(current);
      if (next.has(row.player_id)) next.delete(row.player_id); else next.add(row.player_id);
      return next;
    });
  }

  function resetPlanner() {
    setScoring(DEFAULT_FANTASY_SCORING);
    setRoster(DEFAULT_FANTASY_ROSTER);
    setDrafted(new Set());
    setMine(new Set());
    window.localStorage.removeItem("sports-edge-fantasy-planner-v1");
  }

  return (
    <div className="space-y-4">
      <Card className="overflow-hidden border-accent/30 bg-gradient-to-br from-accent/10 via-card to-card">
        <CardContent className="p-5">
          <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
            <div>
              <div className="flex flex-wrap items-center gap-2">
                <Badge variant="accent">{feed.season} season</Badge>
                <Badge variant="outline">{scoring.name}</Badge>
                <Badge variant={feed.dataSource === "unavailable" ? "missing" : "outline"}>{feed.dataSource.replace("_", " ")}</Badge>
              </div>
              <h2 className="mt-3 text-xl font-semibold tracking-tight sm:text-2xl">Find the points before the draft finds you.</h2>
              <p className="mt-2 max-w-2xl text-sm text-muted-foreground">Original component projections, transparent uncertainty, and a local draft board. Change scoring and the rankings recalculate immediately.</p>
              {validation?.total ? <p className="mt-2 text-xs text-muted-foreground">Validation: {validation.beats}/{validation.total} stat components beat the prior-season baseline on the {validation.holdout} out-of-time holdout.</p> : null}
            </div>
            <div className="flex flex-wrap gap-2">
              <Button variant={view === "rankings" ? "default" : "outline"} onClick={() => setView("rankings")}><Sparkles /> Rankings</Button>
              <Button variant={view === "weekly" ? "default" : "outline"} onClick={() => setView("weekly")}><ArrowDownUp /> Weekly</Button>
              <Button variant={view === "draft" ? "default" : "outline"} onClick={() => setView("draft")}><Users /> Draft room</Button>
              <Button variant={view === "lineup" ? "default" : "outline"} onClick={() => setView("lineup")}><Check /> Lineup</Button>
              <Button variant="outline" size="icon" aria-label="Open scoring settings" onClick={() => setShowSettings((value) => !value)}><Settings2 /></Button>
            </div>
          </div>
        </CardContent>
      </Card>

      {showSettings ? (
        <Card className="anim-slide-up">
          <CardHeader className="flex flex-row items-center justify-between space-y-0"><CardTitle>Scoring and roster</CardTitle><Button variant="ghost" size="icon" onClick={() => setShowSettings(false)} aria-label="Close settings"><X /></Button></CardHeader>
          <CardContent className="space-y-4">
            <div className="flex flex-wrap gap-2"><Button size="sm" variant="outline" onClick={() => setPreset(STANDARD_SCORING)}>Standard</Button><Button size="sm" variant="outline" onClick={() => setPreset(HALF_PPR_SCORING)}>Half PPR</Button><Button size="sm" variant="outline" onClick={() => setPreset(DEFAULT_FANTASY_SCORING)}>Full PPR</Button><Button size="sm" variant="ghost" onClick={resetPlanner}><RotateCcw /> Reset local planner</Button></div>
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-5">{SCORING_FIELDS.map((field) => <label key={field.key} className="space-y-1 text-xs text-muted-foreground">{field.label}<input className="h-9 w-full rounded-md border border-input bg-background px-3 text-sm text-foreground" type="number" step={field.step} value={Number(scoring[field.key] ?? 0)} onChange={(event) => setScoring((current) => ({ ...current, name: "Custom", [field.key]: Number(event.target.value) }))} /></label>)}</div>
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">{(["teams", "quarterback", "running_back", "wide_receiver", "tight_end", "flex", "kicker", "defense", "bench"] as const).map((key) => <label key={key} className="space-y-1 text-xs capitalize text-muted-foreground">{key.replaceAll("_", " ")}<input className="h-9 w-full rounded-md border border-input bg-background px-3 text-sm text-foreground" type="number" min={0} max={20} value={roster[key]} onChange={(event) => setRoster((current) => ({ ...current, [key]: Math.max(0, Number(event.target.value)) }))} /></label>)}</div>
            <p className="text-xs text-muted-foreground">Common QB/RB/WR/TE/FLEX/K/DST redraft settings are supported. Custom settings are saved only in this browser.</p>
          </CardContent>
        </Card>
      ) : null}

      {view === "lineup" ? (
        <Card>
          <CardHeader className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between"><div><CardTitle>Weekly lineup planner</CardTitle><p className="mt-1 text-sm text-muted-foreground">Add players from the board, then let the optimizer fill eligible slots.</p></div><label className="flex items-center gap-2 text-sm">Week<select className="h-9 rounded-md border border-input bg-background px-2" value={week} onChange={(event) => setWeek(Number(event.target.value))}>{Array.from({ length: 18 }, (_, index) => <option key={index + 1} value={index + 1}>{index + 1}</option>)}</select></label></CardHeader>
          <CardContent className="space-y-4">
            <div className="grid gap-3 md:grid-cols-2"><div className="rounded-lg border border-accent/30 bg-accent/5 p-4"><p className="text-xs uppercase tracking-wide text-muted-foreground">Optimized starters</p><p className="mt-1 text-3xl font-semibold text-accent">{lineup.total.toFixed(1)}</p><p className="mt-1 text-xs text-muted-foreground">{lineup.lineup.length} of {roster.quarterback + roster.running_back + roster.wide_receiver + roster.tight_end + roster.flex + roster.kicker + roster.defense} slots filled</p></div><div className="rounded-lg border border-border p-4"><p className="text-xs uppercase tracking-wide text-muted-foreground">Roster pool</p><p className="mt-1 text-3xl font-semibold">{mine.size}</p><p className="mt-1 text-xs text-muted-foreground">Use “Add to roster” below to build a manual player pool.</p></div></div>
            <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">{lineup.lineup.map((item) => <div key={`${item.slot}-${item.row.player_id}`} className="flex items-center justify-between rounded-md border border-border bg-background/60 p-3"><div><p className="text-xs text-muted-foreground">{item.slot}</p><p className="font-medium">{item.row.player_name}</p></div><span className="font-semibold text-accent">{item.row.displayPoints.toFixed(1)}</span></div>)}</div>
            {!mine.size ? <div className="rounded-lg border border-dashed border-border p-6 text-center text-sm text-muted-foreground">Your local roster is empty. Filter the board and add players to get a lineup.</div> : null}
          </CardContent>
        </Card>
      ) : null}

      {view === "draft" ? (
        <Card className="border-accent/30"><CardHeader><CardTitle>Live snake draft assistant</CardTitle><p className="mt-1 text-sm text-muted-foreground">Mark players as drafted. “Mine” adds a player to your local roster and removes them from the available pool.</p></CardHeader><CardContent><div className="grid gap-3 md:grid-cols-3"><div className="rounded-lg border border-border p-3"><p className="text-xs uppercase tracking-wide text-muted-foreground">Drafted</p><p className="mt-1 text-2xl font-semibold">{drafted.size}</p></div><div className="rounded-lg border border-border p-3"><p className="text-xs uppercase tracking-wide text-muted-foreground">My roster</p><p className="mt-1 text-2xl font-semibold">{mine.size}</p></div><div className="rounded-lg border border-accent/30 bg-accent/5 p-3"><p className="text-xs uppercase tracking-wide text-muted-foreground">Best value available</p><p className="mt-1 truncate text-lg font-semibold text-accent">{recommendation?.player_name ?? "Draft board empty"}</p>{recommendation ? <p className="text-xs text-muted-foreground">{recommendation.position} · {recommendation.displayPoints.toFixed(1)} projected PPR points · {recommendation.adp ? `ADP ${recommendation.adp.toFixed(1)}` : "no ADP"}</p> : null}</div></div></CardContent></Card>
      ) : null}

      <Card>
        <CardHeader className="space-y-4"><div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between"><div><CardTitle>{view === "draft" ? "Available player board" : view === "lineup" ? "Roster player pool" : view === "weekly" ? `Week ${week} projections` : "Preseason player projections"}</CardTitle><p className="mt-1 text-sm text-muted-foreground">{filtered.length.toLocaleString()} players · median projection with a calibrated range · model {feed.modelVersion}</p></div><div className="flex flex-wrap gap-2"><input aria-label="Search players" placeholder="Search player" value={search} onChange={(event) => setSearch(event.target.value)} className="h-9 rounded-md border border-input bg-background px-3 text-sm" /><Button variant="outline" size="sm" onClick={() => { setSortKey(sortKey === "points" ? "ppg" : "points"); setAscending(false); }}><ArrowDownUp /> Sort</Button></div></div><div className="flex flex-wrap items-center gap-2">{POSITIONS.map((item) => <Button key={item} size="sm" variant={position === item ? "secondary" : "ghost"} onClick={() => setPosition(item)}>{item}</Button>)}{view === "weekly" || view === "lineup" ? <label className="ml-auto flex items-center gap-2 text-sm text-muted-foreground">Week<select className="h-9 rounded-md border border-input bg-background px-2 text-foreground" value={week} onChange={(event) => setWeek(Number(event.target.value))}>{Array.from({ length: 18 }, (_, index) => <option key={index + 1} value={index + 1}>{index + 1}</option>)}</select></label> : null}</div></CardHeader>
        <CardContent>
          {weeklyLoading ? <p className="mb-3 text-sm text-muted-foreground">Loading week {week} projections…</p> : null}
          {weeklyError ? <p className="mb-3 text-sm text-destructive">{weeklyError}</p> : null}
          <Table className="min-w-[920px]"><TableHeader><TableRow><TableHead className="w-10">#</TableHead><TableHead>Player</TableHead><TableHead>Pos</TableHead><TableHead><button onClick={() => { setSortKey("points"); setAscending(!ascending); }}>Points {sortKey === "points" ? ascending ? <ChevronUp className="inline size-3" /> : <ChevronDown className="inline size-3" /> : null}</button></TableHead><TableHead>Range</TableHead><TableHead>PPG</TableHead><TableHead>Model rank</TableHead><TableHead>ADP</TableHead><TableHead>Action</TableHead></TableRow></TableHeader><TableBody>{filtered.slice(0, 250).map((row, index) => <TableRow key={row.player_id} className={cn(drafted.has(row.player_id) && "opacity-45")}><TableCell className="text-muted-foreground">{index + 1}</TableCell><TableCell><div className="font-medium">{row.player_name}</div><div className="text-xs text-muted-foreground">{row.team ?? "FA"} · {row.confidence} confidence</div></TableCell><TableCell><Badge variant="outline">{row.position}</Badge></TableCell><TableCell className="font-semibold text-accent">{row.displayPoints.toFixed(1)}</TableCell><TableCell className="text-xs text-muted-foreground">{row.displayFloor.toFixed(1)}–{row.displayCeiling.toFixed(1)}</TableCell><TableCell>{(row.displayPoints / Math.max(1, row.projected_games)).toFixed(1)}</TableCell><TableCell>{row.position_rank ? `${row.position}${row.position_rank}` : "—"}</TableCell><TableCell>{row.adp ? <span title="FantasyPros market ADP">{row.adp.toFixed(1)}</span> : <span className="text-muted-foreground">—</span>}</TableCell><TableCell>{view === "draft" ? <div className="flex gap-1"><Button size="sm" variant="outline" disabled={drafted.has(row.player_id)} onClick={() => toggleDraft(row, false)}>{drafted.has(row.player_id) ? "Drafted" : "Other"}</Button><Button size="sm" variant={mine.has(row.player_id) ? "secondary" : "default"} onClick={() => toggleDraft(row, true)}>{mine.has(row.player_id) ? "Mine" : "Draft me"}</Button></div> : view === "lineup" ? <Button size="sm" variant={mine.has(row.player_id) ? "secondary" : "outline"} onClick={() => toggleDraft(row, true)}>{mine.has(row.player_id) ? "Remove" : "Add to roster"}</Button> : <Button size="sm" variant="ghost" onClick={() => setView("lineup")}>Plan</Button>}</TableCell></TableRow>)}</TableBody></Table>
          {filtered.length > 250 ? <p className="mt-3 text-xs text-muted-foreground">Showing the first 250 matches. Narrow the position or search filters to inspect the full board.</p> : null}
        </CardContent>
      </Card>

      {view === "rankings" ? <div className="grid gap-3 md:grid-cols-3 lg:grid-cols-6">{topByPosition.map(({ position: item, row }) => <Card key={item} className="bg-card/80"><CardContent className="p-4"><div className="flex items-center justify-between"><span className="text-xs font-medium uppercase tracking-wide text-muted-foreground">{item}</span><Badge variant="accent">{row.position_rank ? `#${row.position_rank}` : "top"}</Badge></div><p className="mt-3 truncate font-medium">{row.player_name}</p><p className="mt-1 text-2xl font-semibold text-accent">{row.displayPoints.toFixed(1)}</p><p className="text-xs text-muted-foreground">{row.team} · {row.projected_games.toFixed(1)} projected games</p></CardContent></Card>)}</div> : null}
      {feed.gaps.length ? <div className="rounded-lg border border-dashed border-border p-3 text-xs text-muted-foreground">{feed.gaps.join(" ")}</div> : null}
    </div>
  );
}
