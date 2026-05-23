import type { DataQuality, PerformanceHistory } from "@/lib/data/types";

function coverageFromSample(sample: Record<string, number | string | null>) {
  const joined =
    typeof sample.odds_joined_games === "number"
      ? sample.odds_joined_games
      : typeof sample.odds_rows === "number"
        ? sample.odds_rows
        : null;
  const denominator =
    typeof sample.completed_games === "number"
      ? sample.completed_games
      : typeof sample.test_games === "number"
        ? sample.test_games
        : typeof sample.bigquery_scored_games === "number"
          ? sample.bigquery_scored_games
          : null;

  if (!joined || !denominator) {
    return { coveragePct: null, missingRows: null };
  }

  return {
    coveragePct: Math.min(100, Math.max(0, (joined / denominator) * 100)),
    missingRows: Math.max(0, denominator - joined),
  };
}

export function deriveDataQuality(history: PerformanceHistory): DataQuality[] {
  const rows: DataQuality[] = history.records.map((record) => {
    const coverage = coverageFromSample(record.sample);
    const status: DataQuality["status"] =
      record.oddsStatus.includes("no_") || record.oddsStatus.includes("missing")
        ? "missing"
        : record.gaps.length > 0
          ? "warning"
          : "ok";

    return {
      source: `${record.sport} ${record.market} odds/model coverage`,
      sport: record.sport,
      coveragePct: coverage.coveragePct,
      missingRows: coverage.missingRows,
      lastUpdated: history.generatedAt,
      blockingGaps: record.gaps,
      status,
      notes: record.oddsStatus,
    };
  });

  if (history.oddspapi) {
    rows.unshift({
      source: "OddsPapi validation",
      coveragePct:
        typeof history.oddspapi.validation_match_rate === "number"
          ? history.oddspapi.validation_match_rate * 100
          : null,
      missingRows: null,
      lastUpdated: history.generatedAt,
      blockingGaps:
        history.oddspapi.validation_status === "ok"
          ? []
          : [`Validation status: ${history.oddspapi.validation_status ?? "n/a"}`],
      status: history.oddspapi.validation_status === "ok" ? "ok" : "warning",
      notes:
        typeof history.oddspapi.cumulative_api_requests === "number"
          ? `${history.oddspapi.cumulative_api_requests} cumulative API requests recorded`
          : undefined,
    });
  }

  return rows;
}
