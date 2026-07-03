export function formatNumber(value: number | null | undefined, digits = 0) {
  if (typeof value !== "number" || !Number.isFinite(value)) return "n/a";
  return new Intl.NumberFormat("en-US", {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  }).format(value);
}

export function formatPct(value: number | null | undefined, digits = 1) {
  if (typeof value !== "number" || !Number.isFinite(value)) return "n/a";
  return `${(value * 100).toFixed(digits)}%`;
}

export function formatPctFromWhole(value: number | null | undefined, digits = 1) {
  if (typeof value !== "number" || !Number.isFinite(value)) return "n/a";
  return `${value.toFixed(digits)}%`;
}

export function formatMaybePctMetric(value: number | null | undefined, digits = 1) {
  if (typeof value !== "number" || !Number.isFinite(value)) return "n/a";
  return Math.abs(value) <= 1 ? formatPct(value, digits) : formatNumber(value, digits);
}

export function formatDateTime(value: string | null | undefined) {
  if (!value) return "n/a";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "n/a";
  return new Intl.DateTimeFormat("en-US", {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  }).format(date);
}

export function formatDate(value: string | null | undefined) {
  if (!value) return "n/a";
  const date = new Date(`${value.slice(0, 10)}T00:00:00`);
  if (Number.isNaN(date.getTime())) return "n/a";
  return new Intl.DateTimeFormat("en-US", {
    month: "short",
    day: "numeric",
  }).format(date);
}

export function formatGamesSinceLastHr(
  value: number | null | undefined,
  qualityFlags?: string[] | null,
) {
  if (typeof value === "number" && Number.isFinite(value)) {
    return `${Math.round(value)}g`;
  }
  if (qualityFlags?.includes("no_hr_in_history_window")) {
    return "No HR";
  }
  return "—";
}
