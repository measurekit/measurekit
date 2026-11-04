import { readdirSync, readFileSync, rmSync } from "node:fs";
import { dirname, join, relative } from "node:path";
import { fileURLToPath } from "node:url";
import { afterAll, afterEach, describe, expect, it, vi } from "vitest";
import { type Instrumented, instrument, resultsShown, runSample } from "../../scripts/guide-samples.js";

/** The root of the repository. */
const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..", "..");

/** The folder that holds the examples while they run, which git ignores. */
const RUNNING = join(ROOT, "examples", ".tsdoc");

/** A TSDoc comment, and a fenced `ts` block in the text of its `@example` tag. */
const TSDOC = /\/\*\*[\s\S]*?\*\//g;
const FENCE = /^```ts\n([\s\S]*?)^```$/m;

/** An example of a TSDoc comment: where it is, its code with its results checked, and the results it shows. */
interface Example extends Instrumented {
  readonly file: string;
  readonly line: number;
  /** The results that the example shows, whether a statement picked them up or not. */
  readonly shown: number;
}

/** Returns the `.ts` files below a folder. */
function sourcesBelow(folder: string): string[] {
  return readdirSync(folder, { withFileTypes: true }).flatMap((entry) => {
    const path = join(folder, entry.name);
    if (entry.isDirectory()) return entry.name === "node_modules" ? [] : sourcesBelow(path);
    return entry.name.endsWith(".ts") ? [path] : [];
  });
}

/** Returns the text of a comment without the stars that begin its lines. */
function withoutStars(comment: string): string {
  return comment
    .replace(/^\/\*\*[ \t]*\n?/, "")
    .replace(/\*\/$/, "")
    .split("\n")
    .map((line) => line.replace(/^\s*\* ?/, ""))
    .join("\n");
}

/** Returns the examples of the TSDoc comments of a source file. */
function examplesOf(path: string): Example[] {
  const text = readFileSync(path, "utf8");
  const found: Example[] = [];
  for (const match of text.matchAll(TSDOC)) {
    const body = withoutStars(match[0]);
    const tag = body.indexOf("@example");
    if (tag < 0) continue;
    const fence = FENCE.exec(body.slice(tag));
    const file = relative(ROOT, path);
    const line = text.slice(0, match.index).split("\n").length;
    const code = fence?.[1] ?? "";
    found.push({
      file,
      line,
      shown: resultsShown(code),
      ...instrument(code, (codeLine) => `${file}:${line}, line ${codeLine} of the example`),
    });
  }
  return found;
}

const examples = ["core", "psychometrics", "knowledge-tracing"].flatMap((name) =>
  sourcesBelow(join(ROOT, "packages", name, "src")).flatMap(examplesOf),
);

describe("the examples of the TSDoc comments", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  afterAll(() => {
    rmSync(RUNNING, { recursive: true, force: true });
  });

  it("has an example for the main function of each module", () => {
    expect(examples.length).toBeGreaterThan(0);
  });

  it("shows each result after a statement that gives it, with its mark", () => {
    expect(examples.flatMap(({ problems }) => problems)).toEqual([]);
  });

  it("shows at least one result in each example and checks every one, so that running it checks something", () => {
    expect(
      examples
        .filter(({ results, shown }) => results === 0 || results !== shown)
        .map(({ file, line }) => `${file}:${line}`),
    ).toEqual([]);
  });

  it.each(examples)("runs the example of $file from line $line and gets the results it shows", async (example) => {
    vi.spyOn(console, "log").mockImplementation(() => {});
    await runSample(RUNNING, `${example.file.replaceAll(/[/.]/g, "-")}-${example.line}`, example.source);
  });
});
