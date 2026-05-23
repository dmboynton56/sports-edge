export type NullableNumber = number | null;

export type Prediction = {
  id: string;
  sport: string;
  league: string;
  gameId: string;
  eventTime: string | null;
  subject: string;
  homeTeam?: string | null;
  awayTeam?: string | null;
  player?: string | null;
  market: string;
  book: string;
  line: NullableNumber;
  price: NullableNumber;
  modelProbability: NullableNumber;
  impliedProbability: NullableNumber;
  edge: NullableNumber;
  ev: NullableNumber;
  kelly: NullableNumber;
  confidence: NullableNumber;
  modelVersion: string;
  source?: string;
  updatedAt?: string | null;
};

export type PerformanceMetrics = {
  accuracy?: NullableNumber;
  auc?: NullableNumber;
  brier?: NullableNumber;
  logLoss?: NullableNumber;
  mae?: NullableNumber;
  roi?: NullableNumber;
  units?: NullableNumber;
  [key: string]: string | number | null | undefined;
};

export type Performance = {
  sport: string;
  modelVersion: string;
  season: string;
  market: string;
  sampleSize: number | null;
  metrics: PerformanceMetrics;
  roi: NullableNumber;
  units: NullableNumber;
  bets: number | null;
  wins: number | null;
  losses: number | null;
  pushes: number | null;
  oddsStatus: string;
  dataSource?: string;
  sample: Record<string, number | string | null>;
  thresholdPerformance?: Record<string, string | number | null>[];
  modePerformance?: Record<string, string | number | null>[];
  artifactRefs: string[];
  gaps: string[];
};

export type PerformanceHistory = {
  generatedAt: string | null;
  oddspapi?: Record<string, string | number | null>;
  records: Performance[];
  gaps: string[];
};

export type DataQuality = {
  source: string;
  sport?: string;
  coveragePct: number | null;
  missingRows: number | null;
  lastUpdated: string | null;
  blockingGaps: string[];
  status: "ok" | "warning" | "blocked" | "missing";
  notes?: string;
};
