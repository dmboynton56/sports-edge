export function denverDate(now = new Date()): string {
  return now.toLocaleDateString("en-CA", { timeZone: "America/Denver" });
}

export function nflWeekWindow(today: string): { start: string; end: string } {
  const date = new Date(`${today}T12:00:00Z`);
  date.setUTCDate(date.getUTCDate() - (date.getUTCDay() + 5) % 7);
  const start = date.toISOString().slice(0, 10);
  date.setUTCDate(date.getUTCDate() + 6);
  return { start, end: date.toISOString().slice(0, 10) };
}

export function nflPredictionIsCurrent(timestamp: string, now = new Date()): boolean {
  const captured = new Date(timestamp);
  if (!Number.isFinite(captured.getTime()) || captured > now) return false;
  return denverDate(captured) >= nflWeekWindow(denverDate(now)).start;
}
