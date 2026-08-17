import { isFiniteNumber } from "@/lib/data/json";
import type { DataQuality, PerformanceHistory } from "@/lib/data/types";

function coverageFromSample(sample: Record<string, number | string | null>) {
  const joined =
    isFiniteNumber(sample.odds_joined_games)
      ? sample.odds_joined_games
      : isFiniteNumber(sample.odds_rows)
        ? sample.odds_rows
        : null;
  const denominator =
    isFiniteNumber(sample.completed_games)
      ? sample.completed_games
      : isFiniteNumber(sample.test_games)
        ? sample.test_games
        : isFiniteNumber(sample.bigquery_scored_games)
          ? sample.bigquery_scored_games
          : null;

  if (joined == null || denominator == null || denominator <= 0) {
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
        isFiniteNumber(history.oddspapi.validation_match_rate)
          ? history.oddspapi.validation_match_rate * 100
          : null,
      missingRows: null,
      lastUpdated: history.generatedAt,
      blockingGaps:
        ["ok", "pass"].includes(String(history.oddspapi.validation_status).toLowerCase())
          ? []
          : [`Validation status: ${history.oddspapi.validation_status ?? "n/a"}`],
      status: ["ok", "pass"].includes(String(history.oddspapi.validation_status).toLowerCase()) ? "ok" : "warning",
      notes:
        isFiniteNumber(history.oddspapi.cumulative_api_requests)
          ? `${history.oddspapi.cumulative_api_requests} cumulative API requests recorded`
          : undefined,
    });
  }

  return rows;
}
