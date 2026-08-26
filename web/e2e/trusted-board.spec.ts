import { expect, test } from "@playwright/test";

test("MLB HR board fails closed without a live Supabase run", async ({ page }) => {
  await page.goto("/markets/mlb/home-runs");
  await expect(page.getByRole("heading", { name: "MLB Home Runs" })).toBeVisible();
  await expect(page.getByText(/Board unavailable|Board updating|No MLB games/).first()).toBeVisible();
  await expect(page.getByText(/lines priced/i)).toHaveCount(0);
});

test("markets index renders HR board with switcher", async ({ page }) => {
  await page.goto("/markets");
  await expect(page.getByRole("heading", { name: "MLB Home Runs" })).toBeVisible();
  await expect(page.getByRole("link", { name: "Home runs" })).toBeVisible();
});

test("mobile navigation exposes Record", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/markets");
  await page.getByRole("button", { name: /Open navigation/ }).click();
  await expect(page.getByRole("link", { name: "Record" })).toBeVisible();
});
