import { test, expect } from "@playwright/test";

test("webgpu readback returns expected values", async ({ page }) => {
  await page.goto("/?values=1.25,-2.5,3.75,0");
  const hasGpu = await page.evaluate(() => !!navigator.gpu);
  test.skip(!hasGpu, "WebGPU is not available in this browser");

  await page.waitForFunction(
    () => window.__E2E_RESULT__ && window.__E2E_RESULT__.done === true,
    null,
    { timeout: 30000 },
  );
  const result = await page.evaluate(() => window.__E2E_RESULT__);
  expect(result.ok, JSON.stringify(result)).toBe(true);
  expect(result.actual).toEqual(result.expected);
});

test("webgpu readback supports repeat", async ({ page }) => {
  await page.goto("/?values=0.5,-1&repeat=3");
  const hasGpu = await page.evaluate(() => !!navigator.gpu);
  test.skip(!hasGpu, "WebGPU is not available in this browser");

  await page.waitForFunction(
    () => window.__E2E_RESULT__ && window.__E2E_RESULT__.done === true,
    null,
    { timeout: 30000 },
  );
  const result = await page.evaluate(() => window.__E2E_RESULT__);
  expect(result.ok, JSON.stringify(result)).toBe(true);
  expect(result.actual).toEqual(result.expected);
  expect(result.actual.length).toBe(6);
});

test("webgpu readback supports seed and offset", async ({ page }) => {
  await page.goto("/?seed=7&count=4&offset=10");
  const hasGpu = await page.evaluate(() => !!navigator.gpu);
  test.skip(!hasGpu, "WebGPU is not available in this browser");

  await page.waitForFunction(
    () => window.__E2E_RESULT__ && window.__E2E_RESULT__.done === true,
    null,
    { timeout: 30000 },
  );
  const result = await page.evaluate(() => window.__E2E_RESULT__);
  expect(result.ok, JSON.stringify(result)).toBe(true);
  expect(result.actual).toEqual(result.expected);
  expect(result.actual.length).toBe(4);
});

test("webgpu mlp loss matches cpu eval", async ({ page }) => {
  await page.goto("/?mode=loss");
  const hasGpu = await page.evaluate(() => !!navigator.gpu);
  test.skip(!hasGpu, "WebGPU is not available in this browser");

  await page.waitForFunction(
    () => window.__E2E_RESULT__ && window.__E2E_RESULT__.done === true,
    null,
    { timeout: 30000 },
  );
  const result = await page.evaluate(() => window.__E2E_RESULT__);
  expect(result.ok, JSON.stringify(result)).toBe(true);
  expect(result.actual.length).toBe(9);
  expect(result.expected.length).toBe(9);
});

test("webgpu mlp train step matches cpu update", async ({ page }) => {
  await page.goto("/?mode=train");
  const hasGpu = await page.evaluate(() => !!navigator.gpu);
  test.skip(!hasGpu, "WebGPU is not available in this browser");

  await page.waitForFunction(
    () => window.__E2E_RESULT__ && window.__E2E_RESULT__.done === true,
    null,
    { timeout: 30000 },
  );
  const result = await page.evaluate(() => window.__E2E_RESULT__);
  expect(result.ok, JSON.stringify(result)).toBe(true);
  expect(result.actual.length).toBe(23);
  expect(result.expected.length).toBe(23);
});
