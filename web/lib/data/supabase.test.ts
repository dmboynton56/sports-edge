import { afterEach, expect, it, vi } from "vitest";
import { supabaseRest } from "./supabase";

afterEach(() => {
  vi.unstubAllEnvs();
  vi.unstubAllGlobals();
});

it("preserves the default and live-market cache lifetimes", async () => {
  vi.stubEnv("NEXT_PUBLIC_SUPABASE_URL", "https://example.supabase.co/");
  vi.stubEnv("NEXT_PUBLIC_SUPABASE_ANON_KEY", "public-test-key");
  const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => [{ id: 1 }] });
  vi.stubGlobal("fetch", fetchMock);
  expect(await supabaseRest("games?select=id")).toEqual([{ id: 1 }]);
  await supabaseRest("games?select=id", 60);
  expect(fetchMock.mock.calls[0]).toEqual([
    "https://example.supabase.co/rest/v1/games?select=id",
    { headers: { apikey: "public-test-key", Authorization: "Bearer public-test-key" }, next: { revalidate: 300 }, signal: expect.any(AbortSignal) },
  ]);
  expect(fetchMock.mock.calls[1][1].next.revalidate).toBe(60);
});

it("returns unavailable for failed requests and malformed responses", async () => {
  vi.stubEnv("NEXT_PUBLIC_SUPABASE_URL", "https://example.supabase.co");
  vi.stubEnv("NEXT_PUBLIC_SUPABASE_ANON_KEY", "public-test-key");
  vi.stubGlobal("fetch", vi.fn()
    .mockRejectedValueOnce(new Error("offline"))
    .mockResolvedValueOnce({ ok: false })
    .mockResolvedValueOnce({ ok: true, json: async () => ({ message: "unexpected object" }) })
    .mockResolvedValueOnce({ ok: true, json: async () => { throw new SyntaxError("invalid JSON"); } }));
  for (let i = 0; i < 4; i++) expect(await supabaseRest("games")).toBeNull();
});

it("cancels a hung data request and returns unavailable", async () => {
  vi.stubEnv("NEXT_PUBLIC_SUPABASE_URL", "https://example.supabase.co");
  vi.stubEnv("NEXT_PUBLIC_SUPABASE_ANON_KEY", "test-key");
  const controller = new AbortController();
  const timeout = vi.spyOn(AbortSignal, "timeout").mockReturnValue(controller.signal);
  vi.stubGlobal("fetch", vi.fn((_url, { signal }) => new Promise((_resolve, reject) => {
    signal.addEventListener("abort", () => reject(signal.reason), { once: true });
  })));
  try {
    const pending = supabaseRest("games");
    controller.abort(new DOMException("Timed out", "TimeoutError"));
    expect(await pending).toBeNull();
    expect(timeout).toHaveBeenCalledWith(8_000);
  } finally {
    timeout.mockRestore();
  }
});

it("keeps unaffected sports available when one source fails", async () => {
  const { getCfbMarketFeed } = await import("./cfb-markets");
  const { getNflAnytimeTdFeed } = await import("./nfl-anytime-td");
  vi.stubEnv("NEXT_PUBLIC_SUPABASE_URL", "https://example.supabase.co");
  vi.stubEnv("NEXT_PUBLIC_SUPABASE_ANON_KEY", "test-key");
  vi.stubGlobal("fetch", vi.fn(async (url: string) => {
    if (url.includes("cfb_")) throw new Error("CFB upstream offline");
    return { ok: true, json: async () => [] };
  }));
  const [cfb, nfl] = await Promise.all([getCfbMarketFeed(), getNflAnytimeTdFeed()]);
  expect(cfb.predictions).toEqual([]);
  expect(cfb.gaps.join(" ")).toContain("could not be reached");
  expect(nfl.predictions).toEqual([]);
});
