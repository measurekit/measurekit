import { describe, expect, it } from "vitest";
import {
  checkConsole,
  checkDiagrams,
  checkEntries,
  checkErrorNames,
  checkFamilies,
  checkFileNames,
  checkHistory,
  checkModuleExamples,
  checkModuleGuides,
  checkNotIncluded,
  checkReadonlyResults,
  checkSectionSigns,
  checkStandards,
  checkTestTitles,
  type Files,
} from "../check-standards.js";

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

describe("not included", () => {
  const bullet = "- **Kernel equating.** Outside the sources: Kolen and Brennan give it no chapter.\n";

  it("accepts a bullet of a method, a reason of the three and why, wrapped or not, at any heading level", () => {
    const files = repository({
      "docs/a/one.md": `# One\n\n## Not included\n\n${bullet}- **Deep learning.** No fit: it needs a tensor\n  runtime.\n- **Spline.** No\n  reference: nothing runs it.\n\n## Next\n`,
      "docs/a/two.md": `# Two\n\n### Not included\n\n${bullet}`,
    });
    expect(checkNotIncluded(files)).toStrictEqual([]);
  });

  it("reports a bullet without a method in bold, with another reason, without why, and prose or a table", () => {
    const files = repository({
      "docs/a/one.md": [
        "## Not included",
        "",
        "- Kernel equating.",
        "- **Kernel equating.** Little use: it is rare.",
        "- **Kernel equating** No fit: no period.",
        "- **Kernel equating.** No fit:",
        "Some prose.",
        "| Method | Reason |",
        bullet.trim(),
      ].join("\n"),
    });
    expect(checkNotIncluded(files).map(({ line }) => line)).toStrictEqual([3, 4, 5, 6, 7, 8]);
  });

  it("reads only the Not included sections of the guides", () => {
    const files = repository({
      "docs/a/one.md": "## Differences\n\n- Anything goes here.\n",
      "README.md": "## Not included\n\n- Anything.\n",
    });
    expect(checkNotIncluded(files)).toStrictEqual([]);
  });
});

describe("module guides", () => {
  const source = {
    "packages/a/src/index.ts": 'export * from "./one/index.js";\n',
    "packages/a/src/one/index.ts": "export const one = 1;\n",
    "packages/a/src/two/index.ts": "export const two = 2;\n",
    "packages/a/src/math/internal.ts": "export const three = 3;\n",
  };
  const section = "## Not included\n\n- **Kernel equating.** No fit: it is rare.\n";

  it("reports the guide of a module without a Not included section, with the guide and the module in the message", () => {
    const files = repository({
      ...source,
      "docs/a/one.md": `# One\n\n${DIAGRAM}`,
      "docs/a/two.md": `# Two\n\n${section}`,
    });
    expect(checkModuleGuides(files)).toStrictEqual([
      {
        file: "docs/a/one.md",
        line: 0,
        rule: "module-guide",
        message:
          'The guide of the one module has no "## Not included" section; list each method the module leaves out, with its reason.',
      },
    ]);
  });

  it("accepts the section at the end of a guide, whatever the sections around it", () => {
    const files = repository({
      ...source,
      "docs/a/one.md": `# One\n\n## Options\n\n${section}\n## References\n`,
      "docs/a/two.md": `# Two\n\n## Options\n\n${section.trimEnd()}`,
    });
    expect(checkModuleGuides(files)).toStrictEqual([]);
  });

  it("does not take a heading of another level, a fenced heading or a similar title for the section", () => {
    const files = repository({
      ...source,
      "docs/a/one.md": "# One\n\n### Not included\n\n- **A.** No fit: b.\n",
      "docs/a/two.md": "# Two\n\n```md\n## Not included\n```\n\n## Not included here\n",
    });
    expect(checkModuleGuides(files).map(({ file }) => file)).toStrictEqual(["docs/a/one.md", "docs/a/two.md"]);
  });

  it("asks nothing of the other documents of a package, nor of a guide of another package", () => {
    const files = repository({
      ...source,
      "docs/a/architecture.md": "# Architecture\n",
      "docs/a/validation.md": "# Validation\n",
      "docs/a/math.md": "# Math\n",
      "docs/b/one.md": "# One\n",
      "docs/a/one/extra.md": "# Extra\n",
      "README.md": "# A\n",
    });
    expect(checkModuleGuides(files)).toStrictEqual([]);
  });

  it("is one of the checks of the repository", () => {
    const files = repository({ ...source, "docs/a/one.md": `# One\n\n${DIAGRAM}` });
    expect(checkStandards(files).filter(({ rule }) => rule === "module-guide")).toHaveLength(1);
  });
});

describe("all checks", () => {
  it("reports nothing for a repository that holds the standards", () => {
    const files = repository({
      "README.md": `# A\n\n${DIAGRAM}`,
      "packages/a/src/index.ts": 'export { one } from "./one.js";\n',
      "packages/a/src/one.ts":
        "/**\n * One.\n *\n * @example\n * ```ts\n * one(); // 1\n * ```\n */\nexport function one(): number { return 1; }\n",
      "packages/a/test/a.test.ts": 'it("returns one", () => {});\n',
    });
    expect(checkStandards(files)).toStrictEqual([]);
  });

  it("names the rule and the line of each violation", () => {
    const files = repository({ "packages/a/src/x.ts": "const a = 1;\n// §1\n" });
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
