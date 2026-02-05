import { test, expect } from "@playwright/test";

const shouldBench = process.env.BENCH === "1";

const readInt = (name: string, fallback: number) => {
  const raw = process.env[name];
  const value = raw === undefined ? NaN : Number(raw);
  if (!Number.isFinite(value)) return fallback;
  return Math.floor(value);
};

test("webgpu loss benchmark", async ({ page }) => {
  test.skip(!shouldBench, "BENCH=1 not set");
  test.setTimeout(120_000);

  const input = readInt("BENCH_INPUT", 784);
  const hidden = readInt("BENCH_HIDDEN", 128);
  const output = readInt("BENCH_OUTPUT", 10);
  const batch = readInt("BENCH_BATCH", 128);
  const warmup = readInt("BENCH_WARMUP", 20);
  const iters = readInt("BENCH_ITERS", 200);
  const seed = readInt("BENCH_SEED", 0);

  const params = new URLSearchParams({
    mode: "loss",
    bench: "1",
    input: String(input),
    hidden: String(hidden),
    output: String(output),
    batch: String(batch),
    warmup: String(warmup),
    iters: String(iters),
    seed: String(seed),
  });

  await page.goto(`/?${params.toString()}`);
  const hasGpu = await page.evaluate(() => !!navigator.gpu);
  test.skip(!hasGpu, "WebGPU is not available in this browser");

  await page.waitForFunction(
    () => window.__BENCH_RESULT__ && window.__BENCH_RESULT__.done === true,
    null,
    { timeout: 120_000 },
  );

  const result = await page.evaluate(() => window.__BENCH_RESULT__);
  expect(result.ok, JSON.stringify(result)).toBe(true);
  expect(result.spec.input).toBe(input);
  expect(result.spec.hidden).toBe(hidden);
  expect(result.spec.output).toBe(output);
  expect(result.spec.batch).toBe(batch);
  expect(result.spec.warmup).toBe(warmup);
  expect(result.spec.iters).toBe(iters);
  expect(typeof result.stats.avg).toBe("number");
  expect(typeof result.stats.p50).toBe("number");
  expect(typeof result.stats.p95).toBe("number");
  console.log(JSON.stringify(result));
});
