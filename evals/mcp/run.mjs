import { spawnSync } from "node:child_process";
import { mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { parse } from "yaml";

const root = path.dirname(fileURLToPath(import.meta.url));
const sourceConfig = path.join(root, "promptfooconfig.yaml");
const hasJudge = Boolean(process.env.OPENAI_API_KEY || process.env.ANTHROPIC_API_KEY);

const config = parse(readFileSync(sourceConfig, "utf8"));
if (!hasJudge) {
  for (const test of config.tests ?? []) {
    test.assert = (test.assert ?? []).filter((assertion) => assertion.type !== "llm-rubric");
  }
}
config.providers = [{ id: `file://${path.join(root, "provider.mjs")}` }];

const dir = mkdtempSync(path.join(tmpdir(), "sports-edge-mcp-eval-"));
const configPath = path.join(dir, "promptfooconfig.json");
writeFileSync(configPath, JSON.stringify(config, null, 2));

const promptfoo = path.join(root, "node_modules", ".bin", "promptfoo");
const result = spawnSync(promptfoo, ["eval", "-c", configPath, "--no-write"], {
  cwd: root,
  stdio: "inherit",
  env: {
    ...process.env,
    PROMPTFOO_DISABLE_TELEMETRY: "1",
  },
});

if (result.error) {
  console.error(result.error);
  process.exit(1);
}
process.exit(result.status ?? 1);
