import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join, relative } from "node:path";
import { pathToFileURL } from "node:url";
import ts from "typescript";
import { readStated } from "./stated-values.ts";

/**
 * The runner of the samples of the guides and READMEs, and of the examples of the TSDoc. A statement followed by
 * `// => value`, on its line or on the line under it, must give that value, compared as `stated-values.ts` says; a
 * declaration of one name, `const [cuts] = ...` included, is checked by the value it declares. A value that spans lines goes on in the comment lines under
 * it until its brackets close. The samples are TypeScript and run as modules, so that they may hold types.
 */

/** A sample of a guide: `ts continues` runs it after the sample before it, and `ts no-run` shows code that does not run. */
const FENCE = /^```ts( continues| no-run)?\n([\s\S]*?)^```$/gm;

/** One import of named bindings, the only form the samples use. */
const IMPORT = /^import\s+(?:type\s+)?\{([^}]*)\}\s+from\s+"([^"]+)";?\s*$/gm;

/** The mark of a result, which the count of the results of a document reads. */
const RESULT_MARK = /\/\/ => /g;

/** The result after a statement: `// => ` on its line or on the line under it. */
const RESULT_AFTER = /^[ \t]*(?:\n[ \t]*)?\/\/ => ([^\n]*)/;

/** A comment on the line of a statement, which states no result unless it is marked. */
const COMMENT_AFTER = /^[ \t]*\/\/ ?([^\n]*)/;

/** The name of the function that compares a result, as the code of a sample calls it. */
const CHECK = "__stated";

/** A sample's code with each statement that shows a result checked. */
export interface Instrumented {
  readonly source: string;
  /** The number of results that the code checks. */
  readonly results: number;
  /** What is wrong with a result that cannot be checked, or a value shown without its mark, with its place. */
  readonly problems: readonly string[];
}

/** Whether the brackets of a value close, outside its strings. */
function closes(text: string): boolean {
  let depth = 0;
  for (const [token] of text.matchAll(/"(?:[^"\\]|\\.)*"|[[{]|[\]}]/g)) {
    if (token === "[" || token === "{") depth += 1;
    else if (token === "]" || token === "}") depth -= 1;
  }
  return depth <= 0;
}

/** The result shown after the end of a statement, its lines joined, or `undefined` when none is shown. */
function resultAfter(code: string, end: number): string | undefined {
  const rest = code.slice(end);
  const first = RESULT_AFTER.exec(rest);
  if (first === null) return undefined;
  let text = first[1] ?? "";
  let position = first[0].length;
  for (let next = /^\n[ \t]*\/\/ ?([^\n]*)/.exec(rest.slice(position)); !closes(text) && next !== null; ) {
    text += ` ${(next[1] ?? "").trim()}`;
    position += next[0].length;
    next = /^\n[ \t]*\/\/ ?([^\n]*)/.exec(rest.slice(position));
  }
  return text.trim();
}

/** The names that a declaration binds, `const [cuts] = ...` and `const { omega } = ...` included. */
function boundNames(name: ts.BindingName): string[] {
  if (ts.isIdentifier(name)) return [name.text];
  return name.elements.flatMap((element) => (ts.isOmittedExpression(element) ? [] : boundNames(element.name)));
}

/** Whether a comment would state a value with the mark of a result before it: a sample that forgot the mark checks nothing. */
function looksLikeResult(comment: string): boolean {
  try {
    readStated(comment);
    return true;
  } catch {
    return false;
  }
}

/** A replacement of the code of a sample. */
interface Edit {
  readonly start: number;
  readonly end: number;
  readonly text: string;
}

/** Returns the edit that checks the result shown after a statement, or what is wrong with the result. */
function checkOf(file: ts.SourceFile, statement: ts.Statement, result: string, where: string): Edit | string {
  try {
    readStated(result);
  } catch (error) {
    return `${where}: ${(error as Error).message}`;
  }
  const check = (expression: string) => `${CHECK}(${expression}, ${JSON.stringify(result)}, ${JSON.stringify(where)})`;
  if (ts.isExpressionStatement(statement)) {
    const { expression } = statement;
    return { start: expression.getStart(file), end: expression.end, text: check(expression.getText(file)) };
  }
  const names = ts.isVariableStatement(statement)
    ? statement.declarationList.declarations.flatMap((declaration) => boundNames(declaration.name))
    : [];
  const [name] = names;
  return names.length === 1 && name !== undefined
    ? { start: statement.end, end: statement.end, text: ` ${check(name)};` }
    : `${where}: a result follows a statement that gives none; show it after an expression or a declaration of one name.`;
}

/**
 * Returns the code with each statement that shows a result wrapped in a check of it.
 *
 * @param code - The code of a sample.
 * @param place - Names the place of a line of the code, from 1, for the messages.
 */
export function instrument(code: string, place: (line: number) => string): Instrumented {
  const file = ts.createSourceFile("sample.ts", code, ts.ScriptTarget.ES2022, true);
  const edits: Edit[] = [];
  const problems: string[] = [];
  for (const statement of file.statements) {
    const where = place(file.getLineAndCharacterOfPosition(statement.getStart(file)).line + 1);
    const result = resultAfter(code, statement.end);
    const comment = COMMENT_AFTER.exec(code.slice(statement.end))?.[1]?.trim();
    if (result !== undefined) {
      const edit = checkOf(file, statement, result, where);
      if (typeof edit === "string") problems.push(edit);
      else edits.push(edit);
    } else if (comment !== undefined && looksLikeResult(comment)) {
      problems.push(`${where}: the comment "${comment}" states a value; show it as "// => ${comment}".`);
    }
  }
  let source = code;
  for (const { start, end, text } of edits.reverse()) source = source.slice(0, start) + text + source.slice(end);
  return { source, results: edits.length, problems };
}

/** Returns the number of results that a text shows, whether a statement picked them up or not. */
export function resultsShown(text: string): number {
  return text.match(RESULT_MARK)?.length ?? 0;
}

/** A run of samples of a document that share their variables: a sample and the `ts continues` samples after it, as one module. */
export interface Chain extends Instrumented {
  readonly document: string;
  /** The line of the document where the first sample begins. */
  readonly line: number;
}

/** Returns the samples as one module: their imports merged, and each body in a block inside the block of the one before. */
function moduleOf(samples: readonly string[]): string {
  const modules = new Map<string, Set<string>>();
  const bodies = samples.map((code) =>
    code
      .replace(IMPORT, (statement: string, names: string, from: string) => {
        const bindings = modules.get(from) ?? new Set<string>();
        for (const name of names.split(",").map((part) => part.trim())) {
          if (name !== "" && !name.startsWith("type ") && !statement.startsWith("import type")) bindings.add(name);
        }
        modules.set(from, bindings);
        return "";
      })
      .replace(/^export /gm, ""),
  );
  const imports = [...modules].map(([from, bindings]) => `import { ${[...bindings].join(", ")} } from "${from}";`);
  return [...imports, ...bodies.map((body) => `{\n${body}`), "}".repeat(bodies.length), ""].join("\n");
}

/**
 * Returns the runs of samples of a document, each with its results checked.
 *
 * @param root - The root of the repository.
 * @param document - The path of the Markdown file from the root, which the messages name.
 */
export function chainsOf(root: string, document: string): Chain[] {
  const text = readFileSync(join(root, document), "utf8");
  const runs: { line: number; samples: Instrumented[] }[] = [];
  for (const { 1: mark, 2: code = "", index } of text.matchAll(FENCE)) {
    if (mark === " no-run") continue;
    // The code of a sample begins on the line after its fence.
    const first = text.slice(0, index).split("\n").length + 1;
    const sample = instrument(code, (line) => `${document}:${first + line - 1}`);
    const run = runs.at(-1);
    if (mark === " continues" && run !== undefined) run.samples.push(sample);
    else runs.push({ line: first - 1, samples: [sample] });
  }
  return runs.map(({ line, samples }) => ({
    document,
    line,
    source: moduleOf(samples.map(({ source }) => source)),
    results: samples.reduce((sum, { results }) => sum + results, 0),
    problems: samples.flatMap(({ problems }) => problems),
  }));
}

/**
 * Runs the code of a sample as a module of a folder, which imports the comparison of its results, and resolves when it ends.
 * Rejects with what the code throws, or with the first result that differs from what the code gives.
 *
 * @param directory - The folder of the module, below the folder whose dependencies the code imports.
 * @param name - The name of the module's file, without `.ts`.
 */
export async function runSample(directory: string, name: string, source: string): Promise<void> {
  mkdirSync(directory, { recursive: true });
  const comparison = relative(directory, join(import.meta.dirname, "stated-values.js")).replaceAll("\\", "/");
  const file = join(directory, `${name}.ts`);
  writeFileSync(file, `import { stated as ${CHECK} } from "${comparison}";\n${source}`);
  await import(pathToFileURL(file).href);
}
