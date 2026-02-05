import { execFileSync } from "node:child_process";
import { copyFileSync } from "node:fs";
import { resolve } from "node:path";

const root = process.cwd();
const out = execFileSync(
  "moon",
  ["run", "--target", "js", "--build-only", "src/browser"],
  { cwd: root, encoding: "utf8" },
);

const lines = out.trim().split(/\r?\n/);
const jsonLine = [...lines].reverse().find((line) => line.trim().startsWith("{"));
if (!jsonLine) {
  throw new Error(`Failed to find JSON output from moon: ${out}`);
}
let json;
try {
  json = JSON.parse(jsonLine);
} catch (err) {
  throw new Error(`Failed to parse moon output: ${jsonLine}`);
}

const artifact = json?.artifacts_path?.[0];
if (!artifact) {
  throw new Error(`Missing artifacts_path in moon output: ${out}`);
}

const dest = resolve(root, "e2e", "moonbit.js");
copyFileSync(artifact, dest);
