import { describe, expect, it } from "vitest";
import {
  checkConsole,
  checkDiagrams,
  checkEntries,
  checkErrorNames,
  checkFamilies,
  checkFileNames,
  checkGuideSections,
  checkHistory,
  checkModuleExamples,
  checkNotIncluded,
  checkReadmeSections,
  checkReadonlyResults,
  checkSectionSigns,
  checkSecurityLimits,
  checkStandards,
  checkTestTitles,
  type Files,
} from "../check-standards.ts";
import { familyFiles } from "./family-repository.ts";

/** The files of a repository as the checks read them. */
function repository(files: Record<string, string>): Files {
  return new Map(Object.entries(files));
}

const DIAGRAM = "```mermaid\nflowchart LR\n  A --> B\n```\n";

describe("section signs", () => {
  it("reports a section sign in code and in docs", () => {
    const files = repository({
      "packages/a/src/x.ts": "// Rule §4.1 applies.\n",
      "docs/a/guide.md": "See §2.\n",
      "packages/a/src/y.ts": "// Section 4.1 applies.\n",
    });
    expect(checkSectionSigns(files).map(({ file, line }) => [file, line])).toStrictEqual([
      ["packages/a/src/x.ts", 1],
      ["docs/a/guide.md", 1],
    ]);
  });
});

describe("test titles", () => {
  it("reports a row code, a unit code and a clause number in a title", () => {
    const files = repository({
      "packages/a/test/x.test.ts": [
        'it("reads the score [T48]", () => {});',
        'describe("E19: the reader", () => {});',
        'it("reads the score of clause 3.4.3.1", () => {});',
        'it("answers with the media type (D9)", () => {});',
      ].join("\n"),
    });
    expect(checkTestTitles(files).map(({ line }) => line)).toStrictEqual([1, 2, 3, 4]);
  });

  it("accepts a plain sentence, a version number and code outside a title", () => {
    const files = repository({
      "packages/a/test/x.test.ts": [
        'it("reads QTI 3.0.1 documents", () => {});',
        'it("shares the places at a tied boundary equally", () => {});',
        "const code = [T48];",
      ].join("\n"),
    });
    expect(checkTestTitles(files)).toStrictEqual([]);
  });
});

describe("history", () => {
  it("reports history in prose and in comments, and ignores code and fences", () => {
    const files = repository({
      "docs/a/guide.md": [
        "The reader now accepts a URL.",
        '`previously` in code is skipped, and so is the shipped text "previously".',
        "```ts",
        "// previously",
        "```",
        "The option was renamed.",
      ].join("\n"),
      "packages/a/src/x.ts": "/** Returns the text; it formerly returned bytes. */\nconst previously = 1;\n",
      "packages/a/test/x.test.ts": "// We found that the order matters.\n",
    });
    expect(checkHistory(files).map(({ file, line }) => [file, line])).toStrictEqual([
      ["docs/a/guide.md", 1],
      ["docs/a/guide.md", 6],
      ["packages/a/src/x.ts", 1],
      ["packages/a/test/x.test.ts", 1],
    ]);
  });

  it("accepts a state that changes while the code runs", () => {
    const files = repository({ "docs/a/guide.md": "The tool is no longer open once the item is closed.\n" });
    expect(checkHistory(files)).toStrictEqual([]);
  });
});

describe("file names", () => {
  it("reports utils, helpers and misc files, whatever their folder or extension", () => {
    const files = repository({
      "packages/a/src/utils.ts": "",
      "packages/a/src/x/helpers.ts": "",
      "packages/a/test/misc.ts": "",
      "packages/a/src/utilities.ts": "",
    });
    expect(checkFileNames(files).map(({ file }) => file)).toStrictEqual([
      "packages/a/src/utils.ts",
      "packages/a/src/x/helpers.ts",
      "packages/a/test/misc.ts",
    ]);
  });
});

describe("diagrams", () => {
  it("reports a README, a contributing guide and a guide without a Mermaid diagram", () => {
    const files = repository({
      "README.md": "# A\n",
      "CONTRIBUTING.md": "# A\n",
      "packages/a/README.md": "# A\n",
      "docs/a/guide.md": "# Guide\n",
      "docs/a/diagram.md": `# Guide\n\n${DIAGRAM}`,
      "AGENTS.md": "# Agents\n",
    });
    expect(checkDiagrams(files).map(({ file }) => file)).toStrictEqual([
      "README.md",
      "CONTRIBUTING.md",
      "packages/a/README.md",
      "docs/a/guide.md",
    ]);
  });
});

describe("error names", () => {
  it("reports an error class that carries the name of a package", () => {
    const files = repository({
      "packages/psychometrics/src/x.ts":
        "export class PsychometricsError extends Error {}\nexport class FitError extends Error {}\n",
      "packages/knowledge-tracing/src/y.ts": "class KnowledgeTracingError extends Error {}\n",
      "packages/core/src/errors.ts": "export class MeasurekitError extends Error {}\n",
    });
    expect(checkErrorNames(files).map(({ file, line }) => [file, line])).toStrictEqual([
      ["packages/psychometrics/src/x.ts", 1],
      ["packages/knowledge-tracing/src/y.ts", 1],
    ]);
  });
});

describe("families", () => {
  it("reports an import of another family outside the examples", () => {
    const files = repository({
      "packages/psychometrics/src/x.ts": 'import { read } from "@assesskit/runtime";\n',
      "packages/psychometrics/test/x.test.ts": 'const m = await import("@interopkit/learning-analytics");\n',
      "examples/psychometrics/adapter.ts": 'import { read } from "@assesskit/runtime";\n',
      "packages/psychometrics/src/y.ts": 'import { DataError } from "@measurekit/core";\n',
    });
    expect(checkFamilies(files).map(({ file }) => file)).toStrictEqual([
      "packages/psychometrics/src/x.ts",
      "packages/psychometrics/test/x.test.ts",
    ]);
  });

  it("reports a library that imports another library of this family, and accepts core and the examples", () => {
    const files = repository({
      "packages/psychometrics/src/x.ts": 'import { bkt } from "@measurekit/knowledge-tracing";\n',
      "packages/knowledge-tracing/test/x.test.ts": 'export * from "@measurekit/psychometrics/irt";\n',
      "packages/knowledge-tracing/src/y.ts": 'import { cholesky } from "@measurekit/core";\n',
      "packages/psychometrics/src/z.ts": 'import { DataError } from "@measurekit/core/errors";\n',
      "examples/knowledge-tracing/run.ts": 'import { calibrate } from "@measurekit/psychometrics";\n',
    });
    expect(checkFamilies(files).map(({ file }) => file)).toStrictEqual([
      "packages/psychometrics/src/x.ts",
      "packages/knowledge-tracing/test/x.test.ts",
    ]);
  });
});

describe("module entries", () => {
  const files = repository({
    "packages/a/src/checks.ts": "",
    "packages/a/src/one/index.ts": "",
    "packages/a/src/one/internal.ts": "",
    "packages/a/src/one/read.ts": "",
    "packages/a/src/one/write.ts": 'import { read } from "./read.js";\n',
    "packages/a/src/one/nested/index.ts": "",
    "packages/a/src/one/nested/deep.ts": 'import { read } from "../read.js";\n',
    "packages/a/src/two/index.ts": "",
    "packages/a/src/two/use.ts": [
      'import { checks } from "../checks.js";',
      'import { one } from "../one/index.js";',
      'import { shared } from "../one/internal.js";',
      'import { read } from "../one/read.js";',
      'import { nested } from "../one/nested/deep.js";',
      'import { solve } from "../math/internal.js";',
      'import { invert } from "../math/invert.js";',
    ].join("\n"),
    "packages/a/src/math/internal.ts": 'export { invert } from "./invert.js";\n',
    "packages/a/src/math/invert.ts": "",
    "packages/a/test/one.test.ts": [
      'import { one } from "../src/one/index.js";',
      'import { read } from "../src/one/read.js";',
      'import { checks } from "../src/checks.js";',
    ].join("\n"),
  });

  it("reports an import that reaches a file of another module, also of one with only an internal.ts", () => {
    expect(checkEntries(files).map(({ file, line }) => [file, line])).toStrictEqual([
      ["packages/a/src/one/nested/deep.ts", 1],
      ["packages/a/src/two/use.ts", 4],
      ["packages/a/src/two/use.ts", 5],
      ["packages/a/src/two/use.ts", 7],
      ["packages/a/test/one.test.ts", 2],
    ]);
  });

  it("accepts the entries of a module, its own files and the files of the package root", () => {
    const accepted = [...files].filter(([file]) => file === "packages/a/src/one/write.ts");
    expect(checkEntries(new Map(accepted))).toStrictEqual([]);
  });
});

describe("console", () => {
  it("reports a console call in library code, and not in tests, examples or scripts", () => {
    const files = repository({
      "packages/a/src/x.ts": 'console.log("x");\n',
      "packages/a/test/x.test.ts": 'console.log("x");\n',
      "examples/a/run.ts": 'console.log("x");\n',
      "scripts/x.ts": 'console.log("x");\n',
    });
    expect(checkConsole(files).map(({ file }) => file)).toStrictEqual(["packages/a/src/x.ts"]);
  });
});

describe("readonly results", () => {
  const index =
    'export { rows, matrix, byId, maybe, table, sorted, pair } from "./x.js";\nexport type { Row } from "./x.js";\n';

  it("reports a public function that returns a mutable list, matrix or record, alone or in a union", () => {
    const files = repository({
      "packages/a/src/m/index.ts": index,
      "packages/a/src/m/x.ts": [
        "export function rows(): Row[] { return []; }",
        "export function matrix(): readonly number[][] { return []; }",
        "export function byId(): Record<string, number> { return {}; }",
        "export function maybe(): Array<number> | null { return null; }",
        "export function table(): Readonly<Record<string, string[]>> { return {}; }",
      ].join("\n"),
    });
    expect(
      checkReadonlyResults(files).map(({ file, line, message }) => [file, line, message.split(";")[0]]),
    ).toStrictEqual([
      ["packages/a/src/m/x.ts", 1, "rows returns Row[]"],
      ["packages/a/src/m/x.ts", 2, "matrix returns number[]"],
      ["packages/a/src/m/x.ts", 3, "byId returns Record<string, number>"],
      ["packages/a/src/m/x.ts", 4, "maybe returns Array<number>"],
      ["packages/a/src/m/x.ts", 5, "table returns string[]"],
    ]);
  });

  it("accepts readonly lists, matrices and records, tuples, promises of them, and a function the module keeps to itself", () => {
    const files = repository({
      "packages/a/src/m/index.ts": index,
      "packages/a/src/m/x.ts": [
        "export function rows(): readonly Row[] { return []; }",
        "export function matrix(): readonly (readonly number[])[] { return []; }",
        "export function byId(): Readonly<Record<string, number>> { return {}; }",
        "export function maybe(): ReadonlyArray<number> | null { return null; }",
        "export function table(): Readonly<Record<string, readonly string[]>> { return {}; }",
        "export function sorted(): Promise<readonly number[]> { return Promise.resolve([]); }",
        "export function hidden(): Row[] { return []; }",
        "export function pair(): [number, number] { return [0, 0]; }",
      ].join("\n"),
    });
    expect(checkReadonlyResults(files).map(({ message }) => message.split(" ")[0])).toStrictEqual([]);
  });
});

describe("module examples", () => {
  const documented =
    "/**\n * Adds.\n *\n * @example\n * ```ts\n * add(1, 2); // 3\n * ```\n */\nexport function add(): number { return 3; }\n";
  const plain = "/** Subtracts. */\nexport function subtract(): number { return 1; }\n";

  it("accepts an entry that exports a function with an example, named or through export *, and one level of re-exports", () => {
    const files = repository({
      "packages/a/src/index.ts": 'export * from "./m/index.js";\nexport * from "./n/index.js";\n',
      "packages/a/src/m/index.ts": 'export { add } from "./add.js";\nexport type { Sum } from "./add.js";\n',
      "packages/a/src/m/add.ts": documented,
      "packages/a/src/n/index.ts": 'export { add as plus } from "../m/add.js";\n',
    });
    expect(checkModuleExamples(files)).toStrictEqual([]);
  });

  it("follows a root's re-export of another package of the family to that package's entry", () => {
    const files = repository({
      "packages/a/src/index.ts": 'export { add, type Sum } from "@measurekit/core";\n',
      "packages/core/src/index.ts": 'export * from "./add.js";\n',
      "packages/core/src/add.ts": documented,
      "packages/b/src/index.ts": 'export { subtract } from "@measurekit/core";\n',
      "packages/c/src/index.ts": 'export { add } from "other";\n',
    });
    expect(checkModuleExamples(files).map(({ file }) => file)).toStrictEqual([
      "packages/b/src/index.ts",
      "packages/c/src/index.ts",
    ]);
  });

  it("reports an entry whose exported functions have no example, even when another function of the file has one", () => {
    const files = repository({
      "packages/a/src/m/index.ts": 'export { subtract } from "./ops.js";\n',
      "packages/a/src/m/ops.ts": `${plain}${documented}`,
      "packages/a/src/n/index.ts": 'export { subtract } from "./plain.js";\n',
      "packages/a/src/n/plain.ts": plain,
    });
    expect(checkModuleExamples(files).map(({ file }) => file)).toStrictEqual([
      "packages/a/src/m/index.ts",
      "packages/a/src/n/index.ts",
    ]);
  });
});

describe("guide sections", () => {
  const complete = [
    "# Guide",
    "## Quick start",
    "## Errors",
    "## Not included",
    "",
    "| Not included    | Why                                                     |",
    "| --------------- | ------------------------------------------------------- |",
    "| Kernel equating | Outside the sources: Kolen and Brennan give it no chapter. |",
    "",
    "## Differences from the references",
  ].join("\n");

  it("accepts the guide of a module that has every section, and the pages that are not guides of a module", () => {
    const files = repository({
      "docs/a/one.md": complete,
      "docs/a/architecture.md": "# Architecture\n",
      "docs/a/performance.md": "# Performance\n",
      "docs/a/storage.md": "# Storage\n",
      "docs/a/validation.md": "# Validation\n",
      "docs/a/one/extra.md": "# Extra\n",
      "README.md": "# A\n",
    });
    expect(checkGuideSections(files)).toStrictEqual([]);
  });

  it("names each section that the guide of a module lacks, and ignores a heading inside a fence or of another level", () => {
    const files = repository({
      "docs/a/one.md": ["# One", "## Quick start", "```md", "## Errors", "```", "### Not included"].join("\n"),
    });
    expect(checkGuideSections(files)).toStrictEqual(
      ["Errors", "Not included", "Differences from the references"].map((title) => ({
        file: "docs/a/one.md",
        line: 0,
        rule: "guide-sections",
        message: `Every guide of a module has a "## ${title}" section; add it, with "None" and the reason where there is nothing to say.`,
      })),
    );
  });

  it("is one of the checks of the repository", () => {
    const files = repository({ "docs/a/one.md": `# One\n\n${DIAGRAM}` });
    expect(checkStandards(files).filter(({ rule }) => rule === "guide-sections")).toHaveLength(4);
  });
});

describe("not included", () => {
  const table = (...rows: string[]): string =>
    ["# One", "", "## Not included", "", "| Not included | Why |", "| --- | --- |", ...rows, ""].join("\n");

  it("accepts a table of a method and a reason of the three with why on each row, before the next section", () => {
    const files = repository({
      "docs/a/one.md": `${table(
        "| Kernel equating | Outside the sources: Kolen and Brennan give it no chapter. |",
        "| Deep learning   | No fit: it needs a tensor runtime.                           |",
        "| A \\| B spline  | No reference: nothing runs it.                               |",
      )}\n## References\n\n- Anything.\n`,
    });
    expect(checkNotIncluded(files)).toStrictEqual([]);
  });

  it("reports a row without a method, with another reason, without why, of more cells, and prose in the table", () => {
    const files = repository({
      "docs/a/one.md": table(
        "|  | No fit: no method. |",
        "| Kernel equating | Little use: it is rare. |",
        "| Kernel equating | No fit: |",
        "| Kernel equating | No fit: it is rare. | More |",
        "Some prose.",
      ),
    });
    expect(checkNotIncluded(files).map(({ line, rule }) => [line, rule])).toStrictEqual([
      [7, "not-included"],
      [8, "not-included"],
      [9, "not-included"],
      [10, "not-included"],
      [11, "not-included"],
    ]);
  });

  it("reports bullets or another header in place of the table, and a header without its delimiter row", () => {
    const files = repository({
      "docs/a/one.md": "## Not included\n\n- **Kernel equating.** No fit: it is rare.\n",
      "docs/a/two.md": "## Not included\n\n| What | Reason |\n| --- | --- |\n",
      "docs/a/three.md": "## Not included\n\n| Not included | Why |\n| Kernel equating | No fit: it is rare. |\n",
      "docs/a/four.md": "## Not included\n",
    });
    expect(checkNotIncluded(files).map(({ file, line }) => [file, line])).toStrictEqual([
      ["docs/a/one.md", 3],
      ["docs/a/two.md", 3],
      ["docs/a/three.md", 4],
      ["docs/a/four.md", 1],
    ]);
  });

  it("reads only the Not included sections of the pages of docs", () => {
    const files = repository({
      "docs/a/one.md": "## Differences from the references\n\n- Anything goes here.\n",
      "README.md": "## Not included\n\n- Anything.\n",
    });
    expect(checkNotIncluded(files)).toStrictEqual([]);
  });
});

describe("README sections", () => {
  const root = "# A\n## Which package\n## Documentation\n## Development\n## Layout\n## License\n";
  const pkg = "# A\n## Quick start\n## Architecture\n## API\n## Supported\n## Documentation\n## License\n";

  it("accepts the sections of the root and of a package in their order, with others between them", () => {
    const files = repository({
      "README.md": root,
      "packages/a/README.md": pkg.replace("## API", "## Install\n## API"),
      "examples/a/README.md": "# Example\n",
    });
    expect(checkReadmeSections(files)).toStrictEqual([]);
  });

  it("names the section that is missing or out of order", () => {
    const files = repository({
      "README.md": root.replace("## Layout\n", ""),
      "packages/a/README.md":
        "# A\n## Quick start\n## API\n## Architecture\n## Supported\n## Documentation\n## License\n",
    });
    expect(
      checkReadmeSections(files).map(({ file, message }) => [file, message.match(/; "([^"]+)" is/)?.[1]]),
    ).toStrictEqual([
      ["README.md", "Layout"],
      ["packages/a/README.md", "API"],
    ]);
  });
});

describe("all checks", () => {
  it("reports nothing for a repository that holds the standards", () => {
    const files = repository({
      ...familyFiles(),
      "packages/a/src/index.ts": 'export { one } from "./one.js";\n',
      "packages/a/src/one.ts":
        "/**\n * One.\n *\n * @example\n * ```ts\n * one(); // 1\n * ```\n */\nexport function one(): number { return 1; }\n",
      "packages/a/test/a.test.ts": 'it("returns one", () => {});\n',
    });
    expect(checkStandards(files)).toStrictEqual([]);
  });

  it("names the rule and the line of each violation", () => {
    const files = repository({ ...familyFiles(), "packages/a/src/x.ts": "const a = 1;\n// §1\n" });
    expect(checkStandards(files)).toStrictEqual([
      {
        file: "packages/a/src/x.ts",
        line: 2,
        rule: "section-sign",
        message: "Name the rule in words, or link the specification; no section signs.",
      },
    ]);
  });
});

describe("security limits", () => {
  const limits = `const MAX_CELLS = 2 ** 28;
const MAX_ITEMS = 1_000;
export const MAX_WIDE = 24_000_000;
const DEFAULT_MAX_STATES = 10_000;
const DEFAULT_MAX_ITERATIONS = 500;
export function f(cells: number, items: number, maxStates = DEFAULT_MAX_STATES): void {
  if (cells > MAX_CELLS) throw new DataError(\`At most \${MAX_CELLS} cells.\`);
  if (items > MAX_ITEMS) throw new DataError(\`At most \${MAX_ITEMS} items.\`);
  if (items > maxStates) throw new DataError(\`More than \${maxStates} states: raise maxStates.\`);
}
`;
  const importer = `import { MAX_WIDE } from "./a.js";
export const g = (n: number) => { if (n > MAX_WIDE * 2) throw new DataError(\`\${MAX_WIDE}\`); };
`;
  const security = (table: string): string =>
    `# Security\n\n## Default limits\n\n${table}\n\n## Runtime dependencies\n\n2^28, 1,000\n`;

  it("reports each limit whose value the section Default limits does not state: a constant MAX_* that a DataError names, or the default of an option max* that one names", () => {
    const files = repository({
      "packages/a/src/a.ts": limits,
      "packages/a/src/b.ts": importer,
      "SECURITY.md": security("| x | 7 |"),
    });
    expect(
      checkSecurityLimits(files).map(({ file, line, message }) => [file, line, message.split(",")[0]]),
    ).toStrictEqual([
      ["packages/a/src/a.ts", 1, "The limit MAX_CELLS"],
      ["packages/a/src/a.ts", 2, "The limit MAX_ITEMS"],
      ["packages/a/src/a.ts", 4, "The limit DEFAULT_MAX_STATES"],
      ["packages/a/src/b.ts", 2, "The limit MAX_WIDE"],
    ]);
  });

  it("finds a value in each way the table writes a number, and not inside another number", () => {
    const files = (table: string) =>
      repository({ "packages/a/src/a.ts": limits, "packages/a/src/b.ts": importer, "SECURITY.md": security(table) });
    expect(checkSecurityLimits(files("| a | 2^28; 1,000; 10,000; 24 million |"))).toStrictEqual([]);
    expect(checkSecurityLimits(files("| a | 2²⁸, 1,000, as R's; `maxStates`, 10,000 | 24,000,000 |"))).toStrictEqual(
      [],
    );
    expect(checkSecurityLimits(files("| a | 2^280; 21,000; 110,000; 124 million |")).length).toBe(4);
  });

  it("reports every limit when SECURITY.md is lost", () => {
    const files = repository({ "packages/a/src/a.ts": limits, "packages/a/src/b.ts": importer });
    expect(checkSecurityLimits(files)).toHaveLength(4);
  });
});
