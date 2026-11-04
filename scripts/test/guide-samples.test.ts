import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, describe, expect, it } from "vitest";
import { chainsOf, instrument, resultsShown, runSample } from "../guide-samples.js";

/** Instruments a sample whose lines are named by their numbers. */
const instrumented = (code: string) => instrument(code, (line) => `line ${line}`);

/** The folder of the guides and the samples of these tests. */
const FOLDER = mkdtempSync(join(tmpdir(), "measurekit-guide-samples-"));

/** Writes a guide with the samples given, each in a fence of its own, and returns its runs of samples. */
function guide(...fences: readonly (readonly [fence: string, code: string])[]) {
  writeFileSync(join(FOLDER, "guide.md"), fences.map(([fence, code]) => `${fence}\n${code}\n\`\`\`\n`).join("\n"));
  return chainsOf(FOLDER, "guide.md");
}

/** The number of samples run, which names each one's module: a module is loaded once for its name. */
let modules = 0;

/** Runs the samples of a guide, as the test of the guides does. */
async function run(...fences: readonly (readonly [fence: string, code: string])[]): Promise<number> {
  let results = 0;
  for (const chain of guide(...fences)) {
    expect(chain.problems).toEqual([]);
    modules += 1;
    await runSample(join(FOLDER, "samples"), `sample-${modules}`, chain.source);
    results += chain.results;
  }
  return results;
}

afterAll(() => {
  rmSync(FOLDER, { recursive: true, force: true });
});

describe("the runner of the samples of the guides", () => {
  it("checks the result shown on the line of a statement and on the line under it", async () => {
    expect(await run(["```ts", "Math.max(2, 4); // => 4\nMath.min(2, 4);\n// => 2"])).toBe(2);
  });

  it("checks a declaration of one name by the value it declares, a destructured one included", async () => {
    expect(
      await run(["```ts", "const twice = [1, 2].map((n) => n * 2); // => [2, 4]\nconst [first] = twice; // => 2"]),
    ).toBe(2);
  });

  it("joins the lines of a result until its brackets close, and reads the explanation after it", async () => {
    const code =
      "const row = { a: 0.681, b: [1, 2] };\nrow;\n// => { a: 0.68,\n//      b: [1, 2] }: a note\n// A comment.";
    expect(await run(["```ts", code])).toBe(1);
  });

  it("refuses a result that differs, with its place in the guide", async () => {
    await expect(run(["```ts", "Math.max(2, 4); // => 5"])).rejects.toThrow(
      'guide.md:2: the result "5" is not what the code gives: the value is 4, not 5.',
    );
  });

  it("runs a sample that continues after the one before it, with its variables", async () => {
    expect(await run(["```ts", "const base = 3;"], ["```ts continues", "base * 2; // => 6"])).toBe(1);
  });

  it("waits for a statement that awaits, and runs the types of TypeScript", async () => {
    expect(await run(["```ts", "const value: number = await Promise.resolve(7); // => 7"])).toBe(1);
  });

  it("does not run a sample marked no-run, and counts the results it shows", () => {
    expect(guide(["```ts no-run", "Math.max(2, 4); // => 4"])).toEqual([]);
    expect(resultsShown("```ts no-run\nMath.max(2, 4); // => 4\n```\n")).toBe(1);
  });

  it("names a value shown without the mark of a result, and a result that follows no expression or declaration", () => {
    expect(instrumented("Math.max(2, 4); // 4\nMath.min(2, 4); // the smaller").problems).toEqual([
      'line 1: the comment "4" states a value; show it as "// => 4".',
    ]);
    expect(instrumented("if (true) {\n} // => 1").problems).toEqual([
      "line 1: a result follows a statement that gives none; show it after an expression or a declaration of one name.",
    ]);
    expect(instrumented("Math.max(2, 4); // => four").problems).toHaveLength(1);
  });

  it("checks only the statements of the top level, so that a result in a block is counted as shown and not checked", () => {
    const code = "for (const n of [1]) {\n  n; // => 2\n}";
    expect(instrumented(code).results).toBe(0);
    expect(resultsShown(code)).toBe(1);
  });
});
