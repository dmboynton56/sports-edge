import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.dirname(fileURLToPath(import.meta.url));

function loadFixture(name) {
  return JSON.parse(readFileSync(path.join(root, "fixtures", name), "utf8"));
}

function cite(payload) {
  return [
    `as-of ${payload.as_of ?? "missing"}`,
    `model_version ${payload.model_version ?? "n/a"}`,
    `status ${payload.status}`,
    `freshness ${payload.freshness?.status ?? "unknown"}`,
  ].join("; ");
}

function gaps(payload) {
  const items = payload.gaps ?? [];
  return items.length ? `Gaps: ${items.join(" ")}` : "No published gaps.";
}

function answerFromQuestion(question) {
  const q = question.toLowerCase();
  if (/bet|lock|parlay|unit size|put money|should i take|hammer/.test(q)) {
    const status = loadFixture("get_status_stale.json");
    return [
      "I will not recommend a bet or imply a lock.",
      "Sports Edge is a versioned prediction app, not a betting system.",
      cite(status),
      "Fetch live numbers with get_status / list_edges if you want the current board.",
    ].join("\n");
  }
  if (/stale|fresh right now|what is fresh/.test(q)) {
    const payload = loadFixture("get_status_stale.json");
    return [
      `NFL feed is ${payload.freshness.status} as of ${payload.as_of} (model ${payload.model_version}).`,
      "Do not quote this as current.",
      cite(payload),
      gaps(payload),
    ].join("\n");
  }
  if (/empty|no mlb slate/.test(q)) {
    const payload = loadFixture("get_slate_empty.json");
    return [
      `MLB slate is empty for ${payload.windowStart}. Nothing is published for that window.`,
      cite(payload),
      gaps(payload),
    ].join("\n");
  }
  if (/last sunday|graded|w-l|won-loss/.test(q)) {
    const payload = loadFixture("get_results_nfl_last_sunday.json");
    const spread = payload.summaries[0];
    return [
      `Last Sunday NFL graded results (window ${payload.window}, not invented).`,
      `Spread W-L ${spread.wins}-${spread.losses}, units ${spread.units}, sample ${spread.sample}.`,
      `Games: DEN @ KC loss ATS; HOU @ BUF win ATS. Dates ${payload.games.map((g) => g.gameDate).join(", ")}.`,
      cite(payload),
      gaps(payload),
    ].join("\n");
  }
  if (/why|explanation|like hou|like houston/.test(q)) {
    const payload = loadFixture("get_game_hou.json");
    const features = payload.game.topFeatures.map((f) => `${f.feature} impact ${f.impact}`).join("; ");
    return [
      `${payload.game.title}: book line ${payload.game.line}, home win prob ${payload.game.homeWinProb}.`,
      `Top explanation features: ${features}. Injury adjusted: ${payload.game.injuryAdjusted}.`,
      cite(payload),
      gaps(payload),
    ].join("\n");
  }
  if (/ev sort|expected value|sort=ev|by ev/.test(q)) {
    const payload = loadFixture("list_edges_ev.json");
    return [
      `Ranked with sort=${payload.sort}.`,
      `${payload.rows[0].title} EV ${payload.rows[0].ev} (${payload.rows[0].status}).`,
      cite(payload),
    ].join("\n");
  }
  if (/home.run|hr model|player market.*mlb|mlb home/.test(q)) {
    const payload = loadFixture("get_player_markets_mlb.json");
    return [
      `${payload.rows[0].player} HR model probability ${payload.rows[0].modelProbability}, fair odds ${payload.rows[0].fairOdds} (model-derived, not a sportsbook).`,
      "No book EV is calculated.",
      cite(payload),
      gaps(payload),
    ].join("\n");
  }
  if (/anytime.?td|touchdown/.test(q)) {
    const payload = loadFixture("get_player_markets_nfl.json");
    return [
      `${payload.rows[0].player} anytime TD model probability ${payload.rows[0].modelProbability} (${payload.status}).`,
      "Not a sportsbook price; no book EV.",
      cite(payload),
    ].join("\n");
  }
  if (/roi|performance|gate/.test(q)) {
    const payload = loadFixture("get_model_performance.json");
    return [
      `NBA ${payload.records[0].modelVersion} ROI ${payload.records[0].roi} (${payload.records[0].productionStatus}).`,
      payload.note,
      cite(payload),
      gaps(payload),
    ].join("\n");
  }
  if (/edge|biggest|top nba|point_edge/.test(q)) {
    const payload = loadFixture("list_edges_point.json");
    return [
      `Ranked with sort=${payload.sort}.`,
      `Biggest edge: ${payload.rows[0].title} point_edge ${payload.rows[0].pointEdge} (line ${payload.rows[0].line}).`,
      `Second: ${payload.rows[1].title} point_edge ${payload.rows[1].pointEdge}.`,
      cite(payload),
      "Sports Edge model output, not a bet recommendation.",
    ].join("\n");
  }
  const status = loadFixture("get_status_stale.json");
  return `Need a live Sports Edge tool call. ${cite(status)}`;
}

export default class SportsEdgeFixtureProvider {
  id() {
    return "sports-edge-fixture";
  }

  async callApi(prompt) {
    return { output: answerFromQuestion(String(prompt ?? "")) };
  }
}
