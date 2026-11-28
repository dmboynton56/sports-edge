import { Client, StreamableHTTPClientTransport } from "@modelcontextprotocol/client";

const origin = process.argv.slice(2).find((argument) => argument !== "--") || "http://localhost:3000";
const endpoint = new URL("/api/mcp", `${origin.replace(/\/$/, "")}/`);

const calls = [
  { name: "get_status", arguments: {} },
  { name: "list_edges", arguments: { sort: "point_edge", limit: 5 } },
  { name: "list_edges", arguments: { league: "NBA", sort: "ev", limit: 5 } },
  { name: "get_slate", arguments: { league: "NBA" } },
  { name: "get_slate", arguments: { league: "NFL" } },
  { name: "get_slate", arguments: { league: "MLB" } },
  { name: "get_slate", arguments: { league: "PGA" } },
  { name: "get_game", arguments: { league: "NBA", team: "HOU" } },
  { name: "get_results", arguments: { league: "NFL", window: "last_sunday" } },
  { name: "get_model_performance", arguments: { league: "NBA" } },
  { name: "get_player_markets", arguments: { league: "MLB" } },
  { name: "get_player_markets", arguments: { league: "NFL" } },
];

function preview(result) {
  const text = result.content?.map((block) => block.text).join("\n") ?? JSON.stringify(result);
  return text.length > 1200 ? `${text.slice(0, 1200)}\n…` : text;
}

async function main() {
  const client = new Client({ name: "sports-edge-mcp-smoke", version: "1.0.0" });
  const transport = new StreamableHTTPClientTransport(endpoint);
  console.log("Connecting", endpoint.toString());
  await client.connect(transport);
  const { tools } = await client.listTools();
  console.log("tools", tools.map((tool) => tool.name).join(", "));
  for (const call of calls) {
    console.log(`\n=== ${call.name} ${JSON.stringify(call.arguments)} ===`);
    const result = await client.callTool(call);
    console.log(preview(result));
  }
  await client.close();
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
