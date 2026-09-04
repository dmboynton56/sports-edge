import type { FantasyPosition, FantasyProjection, FantasyRoster } from "@/lib/data/fantasy";

export type DraftPickOwner = "mine" | "other";

export type DraftPick = {
  playerId: string;
  owner: DraftPickOwner;
};

export type DraftTurn = {
  pick: number;
  round: number;
  slot: number;
  pickInRound: number;
};

export type DraftRecommendation<T extends FantasyProjection = FantasyProjection> = {
  row: T;
  score: number;
  valueOverReplacement: number;
  reason: string;
};

const POSITIONS: FantasyPosition[] = ["QB", "RB", "WR", "TE", "K", "DST"];
const FLEX_WEIGHTS = {
  QB: 0,
  RB: 0.4,
  WR: 0.5,
  TE: 0.1,
  K: 0,
  DST: 0,
} satisfies Record<FantasyPosition, number>;

export function totalRosterSlots(roster: FantasyRoster) {
  return roster.quarterback + roster.running_back + roster.wide_receiver + roster.tight_end
    + roster.flex + roster.kicker + roster.defense + roster.bench;
}

export function snakeDraftTurn(pick: number, teams: number): DraftTurn {
  const safeTeams = Math.max(1, Math.floor(teams));
  const safePick = Math.max(1, Math.floor(pick));
  const round = Math.floor((safePick - 1) / safeTeams) + 1;
  const pickInRound = ((safePick - 1) % safeTeams) + 1;
  const slot = round % 2 === 1 ? pickInRound : safeTeams - pickInRound + 1;
  return { pick: safePick, round, slot, pickInRound };
}

export function nextPickForSlot(currentPick: number, teams: number, draftPosition: number) {
  const safePosition = Math.min(Math.max(1, Math.floor(draftPosition)), Math.max(1, Math.floor(teams)));
  const searchLimit = currentPick + Math.max(1, teams) * 2;
  for (let pick = Math.max(1, currentPick); pick <= searchLimit; pick += 1) {
    if (snakeDraftTurn(pick, teams).slot === safePosition) return pick;
  }
  return currentPick;
}

export function recommendDraftPicks<T extends FantasyProjection & { displayPoints: number }>(
  rows: T[],
  drafted: Set<string>,
  mine: Set<string>,
  roster: FantasyRoster,
  currentPick: number,
  draftPosition: number,
  limit = 3,
): DraftRecommendation<T>[] {
  if (mine.size >= totalRosterSlots(roster)) return [];

  const available = rows.filter((row) => !drafted.has(row.player_id) && isDraftable(row));
  const myRows = rows.filter((row) => mine.has(row.player_id));
  const counts = positionCounts(myRows);
  const round = snakeDraftTurn(currentPick, roster.teams).round;
  const nextUserPick = nextPickForSlot(currentPick + 1, roster.teams, draftPosition);
  const picksUntilNextTurn = Math.max(0, nextUserPick - currentPick - 1);
  const availableRank = new Map(
    [...available]
      .sort((a, b) => b.displayPoints - a.displayPoints)
      .map((row, index) => [row.player_id, index + 1]),
  );

  // SAFETY: POSITIONS contains every FantasyPosition exactly once, so Object.fromEntries produces a complete record.
  const baselines = Object.fromEntries(POSITIONS.map((position) => {
    const positional = rows
      .filter((row) => row.position === position)
      .sort((a, b) => b.displayPoints - a.displayPoints);
    const rank = replacementRank(position, roster);
    return [position, positional[Math.min(rank - 1, Math.max(0, positional.length - 1))]?.displayPoints ?? 0];
  })) as Record<FantasyPosition, number>;

  const scored = available.map((row): DraftRecommendation<T> => {
    const valueOverReplacement = row.displayPoints - baselines[row.position];
    const need = starterNeed(row.position, counts, roster);
    const rank = availableRank.get(row.player_id) ?? available.length;
    const likelyGone = row.adp != null
      ? row.adp <= currentPick + picksUntilNextTurn
      : rank <= picksUntilNextTurn + 1;
    let score = valueOverReplacement + (need ? Math.max(18, row.displayPoints * 0.08) : 0);

    if (!need && row.position === "QB") score -= 45;
    if (!need && row.position === "TE") score -= 18;
    if (["K", "DST"].includes(row.position) && round < Math.max(9, totalRosterSlots(roster) - 3)) score -= 1_000;
    if (row.availability.toLowerCase() === "questionable") score -= Math.max(5, row.displayPoints * 0.05);
    if (row.availability.toLowerCase() === "doubtful") score -= Math.max(20, row.displayPoints * 0.2);
    if (likelyGone) score += Math.min(20, Math.max(6, valueOverReplacement * 0.18));

    const reason = need
      ? `${row.position} fills an open starter or flex need${likelyGone ? " and may not make it back" : ""}.`
      : `${row.position} offers ${Math.max(0, valueOverReplacement).toFixed(1)} points over replacement${likelyGone ? " and is unlikely to survive the turn" : ""}.`;
    return { row, score, valueOverReplacement, reason };
  });

  return scored
    .sort((a, b) => b.score - a.score || b.row.displayPoints - a.row.displayPoints)
    .slice(0, Math.max(1, limit));
}

function replacementRank(position: FantasyPosition, roster: FantasyRoster) {
  const starters = {
    QB: roster.quarterback,
    RB: roster.running_back,
    WR: roster.wide_receiver,
    TE: roster.tight_end,
    K: roster.kicker,
    DST: roster.defense,
  } satisfies Record<FantasyPosition, number>;
  const flexShare = (FLEX_WEIGHTS[position] ?? 0) * roster.flex;
  return Math.max(1, Math.ceil(roster.teams * (starters[position] + flexShare)) + 1);
}

function starterNeed(position: FantasyPosition, counts: Record<FantasyPosition, number>, roster: FantasyRoster) {
  const baseNeeds = {
    QB: roster.quarterback,
    RB: roster.running_back,
    WR: roster.wide_receiver,
    TE: roster.tight_end,
    K: roster.kicker,
    DST: roster.defense,
  } satisfies Record<FantasyPosition, number>;
  if (counts[position] < baseNeeds[position]) return true;
  if (!["RB", "WR", "TE"].includes(position)) return false;
  const flexUsed = Math.max(0, counts.RB - roster.running_back)
    + Math.max(0, counts.WR - roster.wide_receiver)
    + Math.max(0, counts.TE - roster.tight_end);
  return flexUsed < roster.flex;
}

function positionCounts(rows: FantasyProjection[]) {
  // SAFETY: POSITIONS contains every FantasyPosition exactly once, initialized to zero.
  const counts = Object.fromEntries(POSITIONS.map((position) => [position, 0])) as Record<FantasyPosition, number>;
  for (const row of rows) counts[row.position] += 1;
  return counts;
}

function isDraftable(row: FantasyProjection) {
  return !["out", "inactive", "ir", "pup", "suspended", "bye"].includes(row.availability.toLowerCase());
}
