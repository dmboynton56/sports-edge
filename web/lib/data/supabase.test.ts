import { describe, expect, it } from "vitest";

import { asRestRows } from "@/lib/data/supabase";

describe("asRestRows", () => {
  it("keeps arrays and rejects null or object payloads", () => {
    expect(asRestRows([{ id: "1" }])).toEqual([{ id: "1" }]);
    expect(asRestRows(null)).toBeNull();
    expect(asRestRows({ message: "not a list" })).toBeNull();
  });
});
