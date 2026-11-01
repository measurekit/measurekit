import { readdirSync } from "node:fs";
import { basename, dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { afterEach, describe, expect, it, vi } from "vitest";

/** The folder of the examples, which has one folder for each package. */
const EXAMPLES = join(dirname(fileURLToPath(import.meta.url)), "..");

/** The runnable examples of each package: every file of its folder. */
const examples = readdirSync(EXAMPLES, { withFileTypes: true })
  .filter((entry) => entry.isDirectory() && entry.name !== "test" && entry.name !== "node_modules")
  .flatMap((folder) =>
    readdirSync(join(EXAMPLES, folder.name))
      .filter((file) => file.endsWith(".ts"))
      .map((file) => ({ package: folder.name, file })),
  );

describe("examples", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("has an example for each package", () => {
    expect([...new Set(examples.map(({ package: name }) => name))].sort()).toEqual([
      "core",
      "knowledge-tracing",
      "psychometrics",
    ]);
  });

  it.each(examples)("runs $package/$file to its end and prints what it computes", async ({ package: name, file }) => {
    const log = vi.spyOn(console, "log").mockImplementation(() => {});
    await import(pathToFileURL(join(EXAMPLES, name, file)).href);
    expect(log.mock.calls.length, basename(file)).toBeGreaterThan(0);
    for (const call of log.mock.calls) expect(call.join(" ")).not.toMatch(/NaN|undefined/);
  });
});
