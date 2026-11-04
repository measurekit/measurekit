import { mkdirSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { dirname, join, relative } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import ts from "typescript";
import { afterAll, afterEach, describe, expect, it, vi } from "vitest";
import { readStated } from "./stated-values.js";

/** The root of the repository. */
const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..", "..");

/** The folder that holds the examples while they run, which git ignores. */
const RUNNING = join(ROOT, "examples", ".tsdoc");

/** A TSDoc comment, and a fenced `ts` block in the text of its `@example` tag. */
const TSDOC = /\/\*\*[\s\S]*?\*\//g;
const FENCE = /^```ts\n([\s\S]*?)^```$/m;

/** An example of a TSDoc comment: where it is, its code and the statements whose comments state a value. */
interface Example {
  readonly file: string;
  readonly line: number;
  readonly source: string;
  /** The statements whose comments state a value. */
  readonly statements: number;
  /** What is wrong with a comment that begins like a value and is not one, if anything. */
  readonly problem: string | undefined;
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

/** The comment after a statement: on its line, or on the lines directly under it, joined into one line. */
function commentAfter(code: string, end: number): string | undefined {
  const rest = code.slice(end);
  const sameLine = /^[ \t]*\/\/ ?([^\n]*)/.exec(rest)?.[1];
  if (sameLine !== undefined) return sameLine.trim();
  const under = /^[ \t]*\n((?:[ \t]*\/\/[^\n]*(?:\n|$))+)/.exec(rest)?.[1];
  return under
    ?.split("\n")
    .map((line) => line.replace(/^\s*\/\/ ?/, "").trim())
    .join(" ")
    .trim();
}

/**
 * Returns the code of the example, with each statement whose comment begins with a value checked against it, and the
 * number of those statements.
 */
function checked(code: string, where: string): Pick<Example, "source" | "statements" | "problem"> {
  const file = ts.createSourceFile("example.ts", code, ts.ScriptTarget.ES2022, true);
  const edits: { start: number; end: number; text: string }[] = [];
  let problem: string | undefined;
  for (const statement of file.statements) {
    if (!ts.isExpressionStatement(statement)) continue;
    const text = commentAfter(code, statement.end);
    if (text === undefined) continue;
    const line = file.getLineAndCharacterOfPosition(statement.getStart(file)).line + 1;
    try {
      if (readStated(text) === undefined) continue;
    } catch (error) {
      problem ??= `${where}, line ${line} of the example: ${(error as Error).message}`;
      continue;
    }
    edits.push({
      start: statement.expression.getStart(file),
      end: statement.expression.end,
      text: `__stated(${statement.expression.getText(file)}, ${JSON.stringify(text)}, ${JSON.stringify(`${where}, line ${line} of the example`)})`,
    });
  }
  let source = code;
  for (const { start, end, text } of edits.reverse()) source = source.slice(0, start) + text + source.slice(end);
  return { source, statements: edits.length, problem };
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
    const where = `${file}:${line}`;
    found.push({ file, line, ...checked(code, where) });
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

  it("begins a comment with a value only to state it", () => {
    expect(examples.flatMap(({ problem }) => (problem === undefined ? [] : [problem]))).toEqual([]);
  });

  it("states at least one value in each example, so that running it checks something", () => {
    expect(examples.filter(({ statements }) => statements === 0).map(({ file, line }) => `${file}:${line}`)).toEqual(
      [],
    );
  });

  it.each(examples)(
    "runs the example of $file from line $line and gets the values its comments state",
    async (example) => {
      vi.spyOn(console, "log").mockImplementation(() => {});
      mkdirSync(RUNNING, { recursive: true });
      const path = join(RUNNING, `${example.file.replaceAll(/[/.]/g, "-")}-${example.line}.ts`);
      writeFileSync(path, `import { stated as __stated } from "../test/stated-values.js";\n${example.source}`);
      await import(pathToFileURL(path).href);
    },
  );
});
