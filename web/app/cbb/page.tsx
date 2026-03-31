'use client';

import { useState, useMemo, useCallback, useEffect } from 'react';
import { BracketView, type BracketTeam, type TeamProbabilities } from '@/components/BracketView';
import { SimulationControls, type SimConstraint } from '@/components/SimulationControls';
import { TeamCard } from '@/components/TeamCard';
import { MiniFullBracket } from '@/components/MiniFullBracket';
import { RegionDetail } from '@/components/RegionDetail';
import { MatchupCard } from '@/components/MatchupCard';
import {
  type TeamInfo,
  type MatchupSlot,
  REGION_NAMES,
  computeRegionBracket,
  computeFinalFour,
  resetDownstream,
} from '@/lib/bracketUtils';

// ---------------------------------------------------------------------------
// 2026 bracket data
// ---------------------------------------------------------------------------
const BRACKET_2026_RAW: Omit<BracketTeam, 'probabilities'>[] = [
  { team: "Duke", teamId: 1181, seed: 1, region: "E" },
  { team: "UConn", teamId: 1163, seed: 2, region: "E" },
  { team: "Michigan State", teamId: 1277, seed: 3, region: "E" },
  { team: "Kansas", teamId: 1242, seed: 4, region: "E" },
  { team: "St. John's", teamId: 1385, seed: 5, region: "E" },
  { team: "Louisville", teamId: 1257, seed: 6, region: "E" },
  { team: "UCLA", teamId: 1417, seed: 7, region: "E" },
  { team: "Ohio State", teamId: 1326, seed: 8, region: "E" },
  { team: "TCU", teamId: 1395, seed: 9, region: "E" },
  { team: "UCF", teamId: 1416, seed: 10, region: "E" },
  { team: "South Florida", teamId: 1378, seed: 11, region: "E" },
  { team: "Northern Iowa", teamId: 1320, seed: 12, region: "E" },
  { team: "Cal Baptist", teamId: 1465, seed: 13, region: "E" },
  { team: "North Dakota State", teamId: 1295, seed: 14, region: "E" },
  { team: "Furman", teamId: 1202, seed: 15, region: "E" },
  { team: "Siena", teamId: 1373, seed: 16, region: "E" },
  { team: "Arizona", teamId: 1112, seed: 1, region: "W" },
  { team: "Purdue", teamId: 1345, seed: 2, region: "W" },
  { team: "Gonzaga", teamId: 1211, seed: 3, region: "W" },
  { team: "Arkansas", teamId: 1116, seed: 4, region: "W" },
  { team: "Wisconsin", teamId: 1458, seed: 5, region: "W" },
  { team: "BYU", teamId: 1140, seed: 6, region: "W" },
  { team: "Miami (FL)", teamId: 1274, seed: 7, region: "W" },
  { team: "Villanova", teamId: 1437, seed: 8, region: "W" },
  { team: "Utah State", teamId: 1429, seed: 9, region: "W" },
  { team: "Missouri", teamId: 1281, seed: 10, region: "W" },
  { team: "NC State", teamId: 1301, seed: 11, region: "W" },
  { team: "High Point", teamId: 1219, seed: 12, region: "W" },
  { team: "Hawaii", teamId: 1218, seed: 13, region: "W" },
  { team: "Kennesaw State", teamId: 1244, seed: 14, region: "W" },
  { team: "Queens (NC)", teamId: 1474, seed: 15, region: "W" },
  { team: "Long Island", teamId: 1254, seed: 16, region: "W" },
  { team: "Florida", teamId: 1196, seed: 1, region: "S" },
  { team: "Houston", teamId: 1222, seed: 2, region: "S" },
  { team: "Illinois", teamId: 1228, seed: 3, region: "S" },
  { team: "Nebraska", teamId: 1304, seed: 4, region: "S" },
  { team: "Vanderbilt", teamId: 1435, seed: 5, region: "S" },
  { team: "North Carolina", teamId: 1314, seed: 6, region: "S" },
  { team: "Saint Mary's", teamId: 1388, seed: 7, region: "S" },
  { team: "Clemson", teamId: 1155, seed: 8, region: "S" },
  { team: "Iowa", teamId: 1234, seed: 9, region: "S" },
  { team: "Texas A&M", teamId: 1401, seed: 10, region: "S" },
  { team: "VCU", teamId: 1433, seed: 11, region: "S" },
  { team: "McNeese", teamId: 1270, seed: 12, region: "S" },
  { team: "Troy", teamId: 1407, seed: 13, region: "S" },
  { team: "Penn", teamId: 1335, seed: 14, region: "S" },
  { team: "Idaho", teamId: 1225, seed: 15, region: "S" },
  { team: "Lehigh", teamId: 1250, seed: 16, region: "S" },
  { team: "Michigan", teamId: 1276, seed: 1, region: "M" },
  { team: "Iowa State", teamId: 1235, seed: 2, region: "M" },
  { team: "Virginia", teamId: 1438, seed: 3, region: "M" },
  { team: "Alabama", teamId: 1104, seed: 4, region: "M" },
  { team: "Texas Tech", teamId: 1403, seed: 5, region: "M" },
  { team: "Tennessee", teamId: 1397, seed: 6, region: "M" },
  { team: "Kentucky", teamId: 1246, seed: 7, region: "M" },
  { team: "Georgia", teamId: 1208, seed: 8, region: "M" },
  { team: "Saint Louis", teamId: 1387, seed: 9, region: "M" },
  { team: "Santa Clara", teamId: 1365, seed: 10, region: "M" },
  { team: "SMU", teamId: 1374, seed: 11, region: "M" },
  { team: "Akron", teamId: 1103, seed: 12, region: "M" },
  { team: "Hofstra", teamId: 1220, seed: 13, region: "M" },
  { team: "Wright State", teamId: 1460, seed: 14, region: "M" },
  { team: "Tennessee State", teamId: 1398, seed: 15, region: "M" },
  { team: "UMBC", teamId: 1420, seed: 16, region: "M" },
];

const TEAMS_AS_INFO: TeamInfo[] = BRACKET_2026_RAW.map(t => ({
  team: t.team, teamId: t.teamId, seed: t.seed, region: t.region,
}));

// ---------------------------------------------------------------------------
// Probability helpers
// ---------------------------------------------------------------------------
const SEED_MATCHUPS: [number, number][] = [
  [1, 16], [8, 9], [5, 12], [4, 13], [6, 11], [3, 14], [7, 10], [2, 15],
];

const HIST_SEED_WIN_RATE: Record<string, number> = {
  '1-16': 0.993, '2-15': 0.94, '3-14': 0.85, '4-13': 0.79,
  '5-12': 0.65, '6-11': 0.63, '7-10': 0.61, '8-9': 0.52,
};

function calibratedSeedProb(seedA: number, seedB: number): number {
  const lo = Math.min(seedA, seedB);
  const hi = Math.max(seedA, seedB);
  const key = `${lo}-${hi}`;
  if (HIST_SEED_WIN_RATE[key] !== undefined) {
    return seedA < seedB ? HIST_SEED_WIN_RATE[key] : 1 - HIST_SEED_WIN_RATE[key];
  }
  return 1 / (1 + Math.exp(-0.175 * (seedB - seedA)));
}

function buildSeedProbMatrix(teams: Omit<BracketTeam, 'probabilities'>[]): number[][] {
  const n = teams.length;
  const matrix: number[][] = Array.from({ length: n }, () => new Array(n).fill(0.5));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (i === j) { matrix[i][j] = 0; continue; }
      matrix[i][j] = calibratedSeedProb(teams[i].seed, teams[j].seed);
    }
  }
  return matrix;
}

interface ProbMatrixJson {
  teams: { teamId: number; team: string; seed: number; region: string }[];
  probMatrix: number[][];
  rawModelMatrix?: number[][];
}

async function loadProbMatrices(): Promise<{
  calibrated: number[][] | null;
  raw: number[][] | null;
}> {
  try {
    const res = await fetch('/data/cbb_prob_matrix.json');
    if (!res.ok) return { calibrated: null, raw: null };
    const data: ProbMatrixJson = await res.json();
    const cal = data.probMatrix?.length === 64 ? data.probMatrix : null;
    const raw = data.rawModelMatrix?.length === 64 ? data.rawModelMatrix : null;
    return { calibrated: cal, raw: raw };
  } catch {
    return { calibrated: null, raw: null };
  }
}

// ---------------------------------------------------------------------------
// Monte Carlo simulation (unchanged)
// ---------------------------------------------------------------------------
interface SimResult {
  teams: BracketTeam[];
  simCount: number;
}

function runClientSimulation(
  rawTeams: Omit<BracketTeam, 'probabilities'>[],
  probMatrix: number[][],
  nSims: number,
  constraints: SimConstraint[],
): SimResult {
  const idToIdx = new Map<number, number>();
  rawTeams.forEach((t, i) => idToIdx.set(t.teamId, i));

  const regionMap: Record<string, Omit<BracketTeam, 'probabilities'>[]> = {};
  for (const t of rawTeams) {
    if (!regionMap[t.region]) regionMap[t.region] = [];
    regionMap[t.region].push(t);
  }

  const counts: Record<string, Record<string, number>> = {};
  for (const t of rawTeams) {
    counts[t.team] = { R64: nSims, R32: 0, S16: 0, E8: 0, F4: 0, Championship: 0, Winner: 0 };
  }

  const getProb = (a: number, b: number) => probMatrix[idToIdx.get(a)!]?.[idToIdx.get(b)!] ?? 0.5;

  const applyConstraints = (
    tA: Omit<BracketTeam, 'probabilities'>,
    tB: Omit<BracketTeam, 'probabilities'>,
    roundNum: number,
    aWins: boolean,
  ): boolean => {
    let result = aWins;
    for (const c of constraints) {
      if (c.action === 'eliminate' && c.round === roundNum) {
        if (c.team === tA.team) result = false;
        else if (c.team === tB.team) result = true;
      }
      if (c.action === 'advance_to' && c.round > roundNum) {
        if (c.team === tA.team) result = true;
        else if (c.team === tB.team) result = false;
      }
    }
    return result;
  };

  for (let sim = 0; sim < nSims; sim++) {
    const regionWinners: Record<string, Omit<BracketTeam, 'probabilities'>> = {};

    for (const region of ['E', 'W', 'S', 'M']) {
      const teams = regionMap[region] || [];
      const seedMap = new Map<number, Omit<BracketTeam, 'probabilities'>>();
      teams.forEach(t => seedMap.set(t.seed, t));

      let currentTeams: Omit<BracketTeam, 'probabilities'>[] = [];
      for (const [s1, s2] of SEED_MATCHUPS) {
        const t1 = seedMap.get(s1);
        const t2 = seedMap.get(s2);
        if (!t1 || !t2) continue;
        const prob = getProb(t1.teamId, t2.teamId);
        let aWins = Math.random() < prob;
        aWins = applyConstraints(t1, t2, 0, aWins);
        const winner = aWins ? t1 : t2;
        counts[winner.team].R32++;
        currentTeams.push(winner);
      }

      const roundLabels: (keyof typeof counts[string])[] = ['S16', 'E8', 'F4'];
      for (let r = 0; r < roundLabels.length; r++) {
        const nextTeams: typeof currentTeams = [];
        for (let i = 0; i < currentTeams.length; i += 2) {
          if (i + 1 >= currentTeams.length) { nextTeams.push(currentTeams[i]); continue; }
          const t1 = currentTeams[i];
          const t2 = currentTeams[i + 1];
          const prob = getProb(t1.teamId, t2.teamId);
          let aWins = Math.random() < prob;
          aWins = applyConstraints(t1, t2, r + 1, aWins);
          const winner = aWins ? t1 : t2;
          counts[winner.team][roundLabels[r]]++;
          nextTeams.push(winner);
        }
        currentTeams = nextTeams;
      }

      if (currentTeams.length > 0) {
        regionWinners[region] = currentTeams[0];
      }
    }

    const ffPairs: [string, string][] = [['E', 'W'], ['S', 'M']];
    const champContenders: Omit<BracketTeam, 'probabilities'>[] = [];

    for (const [r1, r2] of ffPairs) {
      const t1 = regionWinners[r1];
      const t2 = regionWinners[r2];
      if (!t1 || !t2) { champContenders.push(t1 || t2); continue; }
      const prob = getProb(t1.teamId, t2.teamId);
      let aWins = Math.random() < prob;
      aWins = applyConstraints(t1, t2, 4, aWins);
      const winner = aWins ? t1 : t2;
      counts[winner.team].Championship++;
      champContenders.push(winner);
    }

    if (champContenders.length >= 2 && champContenders[0] && champContenders[1]) {
      const t1 = champContenders[0];
      const t2 = champContenders[1];
      const prob = getProb(t1.teamId, t2.teamId);
      let aWins = Math.random() < prob;
      aWins = applyConstraints(t1, t2, 5, aWins);
      const winner = aWins ? t1 : t2;
      counts[winner.team].Winner++;
    } else if (champContenders[0]) {
      counts[champContenders[0].team].Winner++;
    }
  }

  const teamsWithProbs: BracketTeam[] = rawTeams.map(t => ({
    ...t,
    probabilities: {
      R64: 1,
      R32: (counts[t.team]?.R32 || 0) / nSims,
      S16: (counts[t.team]?.S16 || 0) / nSims,
      E8: (counts[t.team]?.E8 || 0) / nSims,
      F4: (counts[t.team]?.F4 || 0) / nSims,
      Championship: (counts[t.team]?.Championship || 0) / nSims,
      Winner: (counts[t.team]?.Winner || 0) / nSims,
    },
  }));

  return { teams: teamsWithProbs, simCount: nSims };
}

// ---------------------------------------------------------------------------
// Page Component
// ---------------------------------------------------------------------------
type ActiveTab = 'bracket' | 'simulation';

export default function CBBPage() {
  // --- Shared state ---
  const [usingModel, setUsingModel] = useState(false);
  const seedMatrix = useMemo(() => buildSeedProbMatrix(BRACKET_2026_RAW), []);
  const [probMatrix, setProbMatrix] = useState<number[][]>(seedMatrix);
  const [rawMatrix, setRawMatrix] = useState<number[][] | null>(null);
  const [activeTab, setActiveTab] = useState<ActiveTab>('bracket');

  const idToIdx = useMemo(() => {
    const map = new Map<number, number>();
    BRACKET_2026_RAW.forEach((t, i) => map.set(t.teamId, i));
    return map;
  }, []);

  const teamById = useMemo(() => {
    const map = new Map<number, TeamInfo>();
    TEAMS_AS_INFO.forEach(t => map.set(t.teamId, t));
    return map;
  }, []);

  useEffect(() => {
    loadProbMatrices().then(({ calibrated, raw }) => {
      if (calibrated) {
        setProbMatrix(calibrated);
        setUsingModel(true);
      }
      if (raw) setRawMatrix(raw);
    });
  }, []);

  const getProb = useCallback(
    (aId: number, bId: number, mode: 'tournament' | 'neutral' = 'tournament') => {
      const matrix = mode === 'neutral' && rawMatrix ? rawMatrix : probMatrix;
      const ai = idToIdx.get(aId);
      const bi = idToIdx.get(bId);
      if (ai === undefined || bi === undefined) return 0.5;
      return matrix[ai]?.[bi] ?? 0.5;
    },
    [probMatrix, rawMatrix, idToIdx],
  );

  // --- Bracket picker state ---
  const [userPicks, setUserPicks] = useState<Record<string, number>>({});
  const [selectedRegion, setSelectedRegion] = useState<string | null>('E');
  const [selectedMatchup, setSelectedMatchup] = useState<{
    slot: MatchupSlot;
  } | null>(null);

  const regionBrackets = useMemo(() => {
    const result: Record<string, MatchupSlot[][]> = {};
    for (const r of ['E', 'W', 'S', 'M']) {
      result[r] = computeRegionBracket(TEAMS_AS_INFO, r, userPicks, teamById);
    }
    return result;
  }, [userPicks, teamById]);

  const finalFour = useMemo(
    () => computeFinalFour(regionBrackets, userPicks, teamById),
    [regionBrackets, userPicks, teamById],
  );

  const handlePickWinner = useCallback((slot: MatchupSlot, winner: TeamInfo) => {
    setUserPicks(prev => {
      if (prev[slot.slotId] === winner.teamId) {
        const cleaned = resetDownstream(prev, slot.slotId, winner.teamId);
        delete cleaned[slot.slotId];
        return cleaned;
      }
      const oldWinner = prev[slot.slotId];
      let next = { ...prev, [slot.slotId]: winner.teamId };
      if (oldWinner !== undefined && oldWinner !== winner.teamId) {
        next = resetDownstream(next, slot.slotId, oldWinner);
        next[slot.slotId] = winner.teamId;
      }
      return next;
    });
  }, []);

  const handleMatchupClick = useCallback((slot: MatchupSlot) => {
    if (slot.teamA && slot.teamB) {
      setSelectedMatchup({ slot });
    }
  }, []);

  const handleResetBracket = useCallback(() => {
    setUserPicks({});
    setSelectedMatchup(null);
  }, []);

  const handleAutoFill = useCallback(() => {
    const picks: Record<string, number> = {};

    const fillSlot = (slot: MatchupSlot) => {
      if (slot.teamA && slot.teamB && picks[slot.slotId] === undefined) {
        const prob = getProb(slot.teamA.teamId, slot.teamB.teamId);
        picks[slot.slotId] = prob >= 0.5 ? slot.teamA.teamId : slot.teamB.teamId;
      }
    };

    // Fill each region round by round so picks propagate
    for (const r of ['E', 'W', 'S', 'M']) {
      for (let round = 0; round <= 3; round++) {
        const brackets = computeRegionBracket(TEAMS_AS_INFO, r, picks, teamById);
        for (const slot of brackets[round]) fillSlot(slot);
      }
    }

    // Fill final four rounds
    for (let pass = 0; pass < 2; pass++) {
      const regBrackets: Record<string, MatchupSlot[][]> = {};
      for (const r of ['E', 'W', 'S', 'M']) {
        regBrackets[r] = computeRegionBracket(TEAMS_AS_INFO, r, picks, teamById);
      }
      const ff = computeFinalFour(regBrackets, picks, teamById);
      for (const round of ff) {
        for (const slot of round) fillSlot(slot);
      }
    }

    setUserPicks(picks);
  }, [getProb, teamById]);

  const pickCount = Object.keys(userPicks).length;
  const totalSlots = 63; // 32+16+8+4+2+1 matchups

  // --- Sim tab state ---
  const [constraints, setConstraints] = useState<SimConstraint[]>([]);
  const [simCount, setSimCount] = useState(10000);
  const [isRunning, setIsRunning] = useState(false);
  const [selectedTeam, setSelectedTeam] = useState<BracketTeam | null>(null);

  const baselineResult = useMemo(
    () => runClientSimulation(BRACKET_2026_RAW, probMatrix, 50000, []),
    [probMatrix],
  );

  const [currentResult, setCurrentResult] = useState<SimResult | null>(null);

  useEffect(() => {
    setCurrentResult(baselineResult);
  }, [baselineResult]);

  const handleRunSimulation = useCallback(() => {
    setIsRunning(true);
    setTimeout(() => {
      const result = runClientSimulation(BRACKET_2026_RAW, probMatrix, simCount, constraints);
      setCurrentResult(result);
      setIsRunning(false);
      if (selectedTeam) {
        const updated = result.teams.find(t => t.team === selectedTeam.team);
        if (updated) setSelectedTeam(updated);
      }
    }, 50);
  }, [probMatrix, simCount, constraints, selectedTeam]);

  const handleTeamClick = useCallback((team: BracketTeam) => {
    setSelectedTeam(prev => prev?.team === team.team ? null : team);
  }, []);

  const topContenders = useMemo(() => {
    if (!currentResult) return [];
    return [...currentResult.teams]
      .sort((a, b) => b.probabilities.Winner - a.probabilities.Winner)
      .slice(0, 8);
  }, [currentResult]);

  const baselineMap = useMemo(() => {
    const map: Record<string, TeamProbabilities> = {};
    for (const t of baselineResult.teams) map[t.team] = t.probabilities;
    return map;
  }, [baselineResult]);

  // --- Current region bracket for detail view ---
  const currentRegionRounds = selectedRegion && selectedRegion !== 'F4'
    ? regionBrackets[selectedRegion]
    : null;

  return (
    <div className="container mx-auto px-4 py-8 max-w-[1400px]">
      {/* Page header */}
      <div className="mb-6">
        <h1 className="text-3xl font-bold tracking-tight">
          March Madness <span className="text-accent">2026</span>
        </h1>
        <p className="text-sm text-muted-foreground mt-1">
          Interactive bracket picker & Monte Carlo simulation
          {usingModel && (
            <span className="ml-2 inline-flex items-center px-2 py-0.5 rounded text-[10px] font-medium bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200">
              ML Model
            </span>
          )}
          {!usingModel && (
            <span className="ml-2 inline-flex items-center px-2 py-0.5 rounded text-[10px] font-medium bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200">
              Seed-Only Fallback
            </span>
          )}
        </p>
      </div>

      {/* Tab navigation */}
      <div className="flex gap-1 mb-6 border-b border-border">
        <button
          onClick={() => setActiveTab('bracket')}
          className={`px-4 py-2.5 text-sm font-bold border-b-2 transition-colors -mb-px ${
            activeTab === 'bracket'
              ? 'border-accent text-accent'
              : 'border-transparent text-muted-foreground hover:text-foreground'
          }`}
        >
          Pick Bracket
        </button>
        <button
          onClick={() => setActiveTab('simulation')}
          className={`px-4 py-2.5 text-sm font-bold border-b-2 transition-colors -mb-px ${
            activeTab === 'simulation'
              ? 'border-accent text-accent'
              : 'border-transparent text-muted-foreground hover:text-foreground'
          }`}
        >
          Monte Carlo Sim
        </button>
      </div>

      {/* ========================= BRACKET TAB ========================= */}
      {activeTab === 'bracket' && (
        <div className="space-y-6">
          {/* Actions bar */}
          <div className="flex items-center gap-3 flex-wrap">
            <button
              onClick={handleAutoFill}
              className="px-3 py-1.5 rounded-md text-xs font-bold bg-accent text-accent-foreground hover:bg-accent/80 transition-colors"
            >
              Auto-Fill (Chalk)
            </button>
            <button
              onClick={handleResetBracket}
              className="px-3 py-1.5 rounded-md text-xs font-bold bg-secondary text-foreground border border-border hover:bg-secondary/80 transition-colors"
            >
              Reset Bracket
            </button>
            <span className="text-xs text-muted-foreground">
              {pickCount}/{totalSlots} picks made
            </span>
          </div>

          {/* Mini bracket overview */}
          <MiniFullBracket
            regionBrackets={regionBrackets}
            finalFour={finalFour}
            selectedRegion={selectedRegion}
            onRegionClick={setSelectedRegion}
          />

          {/* Region detail */}
          {currentRegionRounds && (
            <RegionDetail
              region={selectedRegion!}
              rounds={currentRegionRounds}
              onPickWinner={handlePickWinner}
              onMatchupClick={handleMatchupClick}
              getProb={(a, b) => getProb(a, b, 'tournament')}
            />
          )}

          {/* Final Four detail */}
          {selectedRegion === 'F4' && (
            <RegionDetail
              region="F4"
              rounds={finalFour}
              onPickWinner={handlePickWinner}
              onMatchupClick={handleMatchupClick}
              getProb={(a, b) => getProb(a, b, 'tournament')}
            />
          )}

          {/* Matchup detail modal */}
          {selectedMatchup && selectedMatchup.slot.teamA && selectedMatchup.slot.teamB && (
            <MatchupCard
              teamA={selectedMatchup.slot.teamA}
              teamB={selectedMatchup.slot.teamB}
              round={selectedMatchup.slot.round}
              region={selectedMatchup.slot.region}
              probTournament={getProb(
                selectedMatchup.slot.teamA.teamId,
                selectedMatchup.slot.teamB.teamId,
                'tournament',
              )}
              probNeutral={getProb(
                selectedMatchup.slot.teamA.teamId,
                selectedMatchup.slot.teamB.teamId,
                'neutral',
              )}
              winner={selectedMatchup.slot.winner}
              onAdvance={(team) => {
                handlePickWinner(selectedMatchup.slot, team);
                setSelectedMatchup(null);
              }}
              onClose={() => setSelectedMatchup(null)}
            />
          )}
        </div>
      )}

      {/* ======================== SIMULATION TAB ======================== */}
      {activeTab === 'simulation' && (
        <>
          {!currentResult ? (
            <div className="text-center py-20 text-muted-foreground">Loading simulation...</div>
          ) : (
            <>
              {/* Dashboard stats */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-8">
                {[
                  { label: 'Teams', value: '64', sub: '4 regions' },
                  { label: 'Simulations', value: currentResult.simCount.toLocaleString(), sub: 'Monte Carlo runs' },
                  { label: 'Constraints', value: constraints.length.toString(), sub: 'active scenarios' },
                  {
                    label: 'Top Pick',
                    value: topContenders[0]?.team || '-',
                    sub: topContenders[0] ? `${(topContenders[0].probabilities.Winner * 100).toFixed(1)}% to win` : '',
                  },
                ].map(stat => (
                  <div key={stat.label} className="p-3 rounded-lg border border-border bg-card">
                    <p className="text-[10px] text-muted-foreground uppercase tracking-wider">{stat.label}</p>
                    <p className="text-lg font-bold mt-0.5">{stat.value}</p>
                    <p className="text-[10px] text-muted-foreground">{stat.sub}</p>
                  </div>
                ))}
              </div>

              {/* Top contenders bar */}
              <div className="mb-6 p-4 rounded-lg border border-border bg-card">
                <h2 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-3">
                  Top Contenders
                </h2>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                  {topContenders.map(team => {
                    const baseWin = baselineMap[team.team]?.Winner || 0;
                    const diff = team.probabilities.Winner - baseWin;
                    const hasDiff = Math.abs(diff) > 0.001;
                    return (
                      <button
                        key={team.teamId}
                        onClick={() => handleTeamClick(team)}
                        className={`p-2.5 rounded-md border transition-all text-left
                          hover:border-accent/50 cursor-pointer
                          ${selectedTeam?.team === team.team ? 'border-accent bg-accent/5' : 'border-border bg-secondary/30'}
                        `}
                      >
                        <div className="flex items-center gap-1.5 mb-1">
                          <span className="text-[10px] font-mono text-muted-foreground">#{team.seed}</span>
                          <span className="text-xs font-bold truncate">{team.team}</span>
                        </div>
                        <div className="flex items-baseline gap-2">
                          <span className="text-lg font-bold text-accent">
                            {(team.probabilities.Winner * 100).toFixed(1)}%
                          </span>
                          {hasDiff && (
                            <span className={`text-[10px] font-medium ${diff > 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                              {diff > 0 ? '+' : ''}{(diff * 100).toFixed(1)}
                            </span>
                          )}
                        </div>
                        <div className="text-[10px] text-muted-foreground">
                          F4: {(team.probabilities.F4 * 100).toFixed(0)}%
                        </div>
                      </button>
                    );
                  })}
                </div>
              </div>

              {/* Main content: Bracket + Sidebar */}
              <div className="grid grid-cols-1 xl:grid-cols-[1fr_320px] gap-6">
                <div className="space-y-6">
                  <div className="p-4 rounded-lg border border-border bg-card">
                    <h2 className="text-sm font-bold mb-4">Tournament Bracket</h2>
                    <BracketView
                      teams={currentResult.teams}
                      onTeamClick={handleTeamClick}
                      selectedTeam={selectedTeam?.team}
                      constraints={constraints.map(c => ({ team: c.team, action: c.action, round: c.round }))}
                    />
                  </div>
                </div>

                <div className="space-y-4">
                  <div className="p-4 rounded-lg border border-border bg-card">
                    <h2 className="text-sm font-bold mb-3">Simulation Controls</h2>
                    <SimulationControls
                      teams={currentResult.teams}
                      constraints={constraints}
                      onConstraintsChange={setConstraints}
                      onRunSimulation={handleRunSimulation}
                      isRunning={isRunning}
                      simCount={simCount}
                      onSimCountChange={setSimCount}
                      selectedTeam={selectedTeam}
                    />
                  </div>

                  {selectedTeam && (
                    <TeamCard
                      team={selectedTeam}
                      baselineProbabilities={baselineMap[selectedTeam.team]}
                      onClose={() => setSelectedTeam(null)}
                    />
                  )}
                </div>
              </div>
            </>
          )}
        </>
      )}
    </div>
  );
}
