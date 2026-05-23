export type TeamInfo = {
  team: string;
  teamId: number;
  seed: number;
  region: string;
};

export type MatchupSlot = {
  slotId: string;
  region: string;
  round: number;
  index: number;
  teamA: TeamInfo | null;
  teamB: TeamInfo | null;
  winner: TeamInfo | null;
};

export const REGION_NAMES: Record<string, string> = {
  E: "East",
  W: "West",
  S: "South",
  M: "Midwest",
  F4: "Final Four",
};

export const ROUND_NAMES: Record<number, string> = {
  0: "Round of 64",
  1: "Round of 32",
  2: "Sweet 16",
  3: "Elite 8",
  4: "Final Four",
  5: "Championship",
};

const SEED_MATCHUPS: [number, number][] = [
  [1, 16],
  [8, 9],
  [5, 12],
  [4, 13],
  [6, 11],
  [3, 14],
  [7, 10],
  [2, 15],
];

function pickedWinner(
  slotId: string,
  userPicks: Record<string, number>,
  teamById: Map<number, TeamInfo>,
  teamA: TeamInfo | null,
  teamB: TeamInfo | null,
) {
  const picked = userPicks[slotId];
  if (picked == null) return null;
  if (picked !== teamA?.teamId && picked !== teamB?.teamId) return null;
  return teamById.get(picked) ?? null;
}

export function computeRegionBracket(
  teams: TeamInfo[],
  region: string,
  userPicks: Record<string, number>,
  teamById: Map<number, TeamInfo>,
): MatchupSlot[][] {
  const regionTeams = teams.filter((team) => team.region === region);
  const bySeed = new Map(regionTeams.map((team) => [team.seed, team]));
  const rounds: MatchupSlot[][] = [];

  rounds[0] = SEED_MATCHUPS.map(([seedA, seedB], index) => {
    const teamA = bySeed.get(seedA) ?? null;
    const teamB = bySeed.get(seedB) ?? null;
    const slotId = `${region}-0-${index}`;
    return {
      slotId,
      region,
      round: 0,
      index,
      teamA,
      teamB,
      winner: pickedWinner(slotId, userPicks, teamById, teamA, teamB),
    };
  });

  for (let round = 1; round <= 3; round++) {
    const previous = rounds[round - 1];
    rounds[round] = [];
    for (let index = 0; index < previous.length / 2; index++) {
      const teamA = previous[index * 2]?.winner ?? null;
      const teamB = previous[index * 2 + 1]?.winner ?? null;
      const slotId = `${region}-${round}-${index}`;
      rounds[round].push({
        slotId,
        region,
        round,
        index,
        teamA,
        teamB,
        winner: pickedWinner(slotId, userPicks, teamById, teamA, teamB),
      });
    }
  }

  return rounds;
}

function regionWinner(regionBrackets: Record<string, MatchupSlot[][]>, region: string) {
  return regionBrackets[region]?.[3]?.[0]?.winner ?? null;
}

export function computeFinalFour(
  regionBrackets: Record<string, MatchupSlot[][]>,
  userPicks: Record<string, number>,
  teamById: Map<number, TeamInfo>,
): MatchupSlot[][] {
  const semiTeams: [TeamInfo | null, TeamInfo | null][] = [
    [regionWinner(regionBrackets, "E"), regionWinner(regionBrackets, "W")],
    [regionWinner(regionBrackets, "S"), regionWinner(regionBrackets, "M")],
  ];

  const semifinals = semiTeams.map(([teamA, teamB], index) => {
    const slotId = `F4-4-${index}`;
    return {
      slotId,
      region: "F4",
      round: 4,
      index,
      teamA,
      teamB,
      winner: pickedWinner(slotId, userPicks, teamById, teamA, teamB),
    };
  });

  const teamA = semifinals[0]?.winner ?? null;
  const teamB = semifinals[1]?.winner ?? null;
  const championshipSlotId = "F4-5-0";
  const championship: MatchupSlot = {
    slotId: championshipSlotId,
    region: "F4",
    round: 5,
    index: 0,
    teamA,
    teamB,
    winner: pickedWinner(championshipSlotId, userPicks, teamById, teamA, teamB),
  };

  return [semifinals, [championship]];
}

export function resetDownstream(
  picks: Record<string, number>,
  slotId: string,
  oldWinner?: number,
) {
  const [region, roundRaw] = slotId.split("-");
  const round = Number(roundRaw);
  const next = { ...picks };

  for (const [key, value] of Object.entries(picks)) {
    const [keyRegion, keyRoundRaw] = key.split("-");
    const keyRound = Number(keyRoundRaw);
    const sameRegionDownstream = keyRegion === region && keyRound > round;
    const finalFourDownstream = keyRegion === "F4" && (region !== "F4" || keyRound > round);
    const referencesOldWinner = oldWinner != null && value === oldWinner;

    if (sameRegionDownstream || finalFourDownstream || referencesOldWinner) {
      delete next[key];
    }
  }

  return next;
}
