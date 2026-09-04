import { expect, test } from "@playwright/test";

const primaryDestinations = ["Overview", "Markets", "Models", "Fantasy"];

test("desktop and mobile navigation expose exactly four primary destinations", async ({ page }) => {
  await page.goto("/");
  const desktopNav = page.locator("header nav");
  await expect(desktopNav.getByRole("link")).toHaveCount(4);
  expect(await desktopNav.getByRole("link").allTextContents()).toEqual(primaryDestinations);

  await page.setViewportSize({ width: 390, height: 844 });
  await page.getByRole("button", { name: "Open navigation" }).click();
  const mobileNav = page.locator('[role="dialog"] nav');
  await expect(mobileNav.getByRole("link")).toHaveCount(4);
  expect(await mobileNav.getByRole("link").allTextContents()).toEqual(primaryDestinations);
});

test("models stays active across nested accountability routes", async ({ page }) => {
  await page.goto("/models/results");
  await expect(page.locator("header").getByRole("link", { name: "Models" })).toHaveAttribute("aria-current", "page");
  await expect(page.getByRole("navigation", { name: "Models" }).getByRole("link")).toHaveCount(6);
  await expect(page.getByRole("navigation", { name: "Models" }).getByRole("link", { name: "Results" })).toHaveAttribute("aria-current", "page");
});

test("markets uses URL-backed filters and an initially collapsed warnings disclosure", async ({ page }) => {
  await page.goto("/markets?sport=NFL&market=spread&probability=50&status=research&sort=eventTime&dir=asc");
  await expect(page.getByRole("heading", { name: "Markets", exact: true })).toBeVisible();
  await expect(page.getByRole("combobox", { name: "Sport filter" })).toContainText("NFL");
  await expect(page.getByRole("combobox", { name: "Market filter" })).toContainText("spread");
  await expect(page.getByRole("combobox", { name: "Minimum probability filter" })).toContainText("50%+");
  await expect(page.getByRole("combobox", { name: "Status filter" })).toContainText("Research");
  const warnings = page.locator("details");
  await expect(warnings).not.toHaveAttribute("open", "");
  await expect(warnings.locator("summary")).toContainText(/Warnings \(\d+\)/);
});

test("legacy routes redirect to canonical markets and models URLs", async ({ page }) => {
  const redirects = [
    ["/nba", "/markets?sport=NBA&market=spread"],
    ["/nfl", "/markets?sport=NFL&market=spread"],
    ["/performance", "/models/performance"],
    ["/results", "/models/results"],
    ["/insights", "/models/insights"],
    ["/data-quality", "/models/data-quality"],
  ] as const;

  for (const [legacy, canonical] of redirects) {
    await page.goto(legacy);
    await expect(page).toHaveURL(new RegExp(`${canonical.replace(/[?]/g, "\\?")}$`));
  }

  await page.goto("/nba/example-game");
  await expect(page).toHaveURL(/\/markets\/nba\/example-game$/);
  await page.goto("/nfl/example-game");
  await expect(page).toHaveURL(/\/markets\/nfl\/example-game$/);
});

test("results tables scroll instead of colliding in split cards", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/models/results");
  await expect(page.getByRole("heading", { name: "Results", exact: true })).toBeVisible();
  await expect(page.locator("main")).toHaveCSS("overflow-x", "visible");
  const tables = page.locator("table");
  if (await tables.count()) {
    const wrapper = tables.first().locator("..");
    await expect(wrapper).toHaveCSS("overflow-x", "auto");
  }
});
