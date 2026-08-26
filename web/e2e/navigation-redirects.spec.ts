import { expect, test } from "@playwright/test";

test("home redirects to MLB HR board", async ({ page }) => {
  await page.goto("/");
  await expect(page).toHaveURL(/\/markets\/mlb\/home-runs/);
  await expect(page.getByRole("heading", { name: "MLB Home Runs" })).toBeVisible();
});

test("results redirects to record", async ({ page }) => {
  await page.goto("/results");
  await expect(page).toHaveURL(/\/record/);
  await expect(page.getByRole("heading", { name: "Record" })).toBeVisible();
});

test("performance redirects to record", async ({ page }) => {
  await page.goto("/performance");
  await expect(page).toHaveURL(/\/record/);
  await expect(page.getByRole("heading", { name: "Record" })).toBeVisible();
});

test("performance with sport redirects to record", async ({ page }) => {
  await page.goto("/performance/mlb");
  await expect(page).toHaveURL(/\/record/);
  await expect(page.getByRole("heading", { name: "Record" })).toBeVisible();
});

test("insights redirects to record", async ({ page }) => {
  await page.goto("/insights");
  await expect(page).toHaveURL(/\/record/);
  await expect(page.getByRole("heading", { name: "Record" })).toBeVisible();
});

test("markets is active when on HR board", async ({ page }) => {
  await page.goto("/markets/mlb/home-runs");
  const marketsLink = page.getByRole("link", { name: "Markets" }).first();
  await expect(marketsLink).toHaveAttribute("aria-current", "page");
});

test("markets index also shows HR board", async ({ page }) => {
  await page.goto("/markets");
  await expect(page.getByRole("heading", { name: "MLB Home Runs" })).toBeVisible();
  const marketsLink = page.getByRole("link", { name: "Markets" }).first();
  await expect(marketsLink).toHaveAttribute("aria-current", "page");
});

test("record is active when on record page", async ({ page }) => {
  await page.goto("/record");
  const recordLink = page.getByRole("link", { name: "Record" }).first();
  await expect(recordLink).toHaveAttribute("aria-current", "page");
});

test("mobile navigation shows 4 nav items", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/markets");
  await page.getByRole("button", { name: /Open navigation/ }).click();
  await expect(page.getByRole("link", { name: "Markets" })).toBeVisible();
  await expect(page.getByRole("link", { name: "Models" })).toBeVisible();
  await expect(page.getByRole("link", { name: "Fantasy" })).toBeVisible();
  await expect(page.getByRole("link", { name: "Record" })).toBeVisible();
  await expect(page.getByRole("link", { name: "Overview" })).toHaveCount(0);
  await expect(page.getByRole("link", { name: "Results" })).toHaveCount(0);
  await expect(page.getByRole("link", { name: "Performance" })).toHaveCount(0);
});
