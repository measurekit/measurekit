import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { basename, dirname, extname, join, posix, resolve } from "node:path";
import ts from "typescript";
import {
  checkClaimWords,
  checkEvidencePages,
  checkIdentity,
  checkReadmeLinks,
  checkTrademarks,
  manifestOf,
} from "./check-identity.ts";

/** One breach of the standards: the file, the 1-based line (0 for the whole file), the rule and what to do. */
export interface Violation {
  readonly file: string;
  readonly line: number;
  readonly rule: string;
  readonly message: string;
}

/** The text of the files to check, by path relative to the repository root, with `/` as separator. */
export type Files = ReadonlyMap<string, string>;

/** The file types that carry prose or code of the repository. */
const TEXT_EXTENSIONS = new Set([".ts", ".md", ".css", ".yml", ".grit"]);
/** Paths that hold generated or third-party text. */
const EXCLUDED = /(^|\/)(node_modules|dist|docs\/api|\.cache|\.changeset|fixtures)\/|(^|\/)CHANGELOG\.md$/;
/** The script and its test, which hold violations as their input. */
const EXEMPT = /^scripts\/(check-standards\.ts|test\/check-standards\.test\.ts)$/;
/** Scopes of other families of packages, which no package of this repository may import. */
const FOREIGN_SCOPES = ["@assesskit/", "@interopkit/"];
/** Where the foreign scopes are allowed: an example adapts one family to the other. */
const FOREIGN_ALLOWED = /^examples\//;
/** The one package of this family that the others may import: the libraries share core and nothing else. */
const SHARED_PACKAGE = "@measurekit/core";
/** The reasons that a method stays out of a module: the three tests of what is complete. */
const REASONS = ["No fit", "No reference", "Outside the sources"];
/** File names, without extension, that say nothing about the concept they hold. */
const VAGUE_FILE_NAMES = new Set(["utils", "util", "helpers", "helper", "misc"]);

/** Splits the text of a file into its lines. */
function linesOf(text: string): string[] {
  return text.split("\n");
}

/** Returns the violations of a rule for each line of a file that matches a pattern. */
function matchLines(
  file: string,
  text: string,
  rule: string,
  pattern: RegExp,
  message: (match: string) => string,
): Violation[] {
  const found: Violation[] = [];
  linesOf(text).forEach((line, index) => {
    const match = pattern.exec(line);
    if (match !== null) found.push({ file, line: index + 1, rule, message: message(match[0]) });
  });
  return found;
}

const isTypeScript = (file: string): boolean => extname(file) === ".ts";
const isMarkdown = (file: string): boolean => extname(file) === ".md";
const isTest = (file: string): boolean => /(^|\/)test\/.*\.ts$/.test(file) || /\.(test|bench)\.ts$/.test(file);
const isLibrary = (file: string): boolean => /^packages\/[^/]+\/src\/.*\.ts$/.test(file);

/** The lines of a Markdown file outside fenced code, with inline code and quoted strings removed, as `[line number, text]`. */
function proseLines(text: string): [number, string][] {
  const prose: [number, string][] = [];
  let fenced = false;
  linesOf(text).forEach((line, index) => {
    if (/^\s*```/.test(line)) fenced = !fenced;
    else if (!fenced) prose.push([index + 1, line.replace(/`[^`]*`|"[^"]*"/g, "")]);
  });
  return prose;
}

/** The comment lines of a TypeScript file, as `[line number, text]`. */
function commentLines(text: string): [number, string][] {
  const comments: [number, string][] = [];
  linesOf(text).forEach((line, index) => {
    if (/^\s*(\/\/|\/\*|\*)/.test(line)) comments.push([index + 1, line]);
  });
  return comments;
}

/** Section signs name a numbered section of a document, which a reader cannot follow: name the rule or link it. */
export function checkSectionSigns(files: Files): Violation[] {
  return [...files].flatMap(([file, text]) =>
    matchLines(
      file,
      text,
      "section-sign",
      /§/,
      () => "Name the rule in words, or link the specification; no section signs.",
    ),
  );
}

/** The titles of the blocks of a test file, which are sentences about behavior. */
const TITLE = /\b(?:describe|it|test|bench)(?:\.\w+)*\(\s*(["'`])((?:\\.|(?!\1).)*)\1/;
/** A code that names a row, a unit or a numbered clause: `[T48]`, `E19:`, `3.4.3.1`. */
const TITLE_CODE =
  /\[[A-Z]{1,4}-?\d+[a-z]?\]|^[A-Z]{1,4}-?\d+[a-z]?[:.\s]|\(\s*[A-Z]{1,4}-?\d+\s*\)|\b(?!\d+\.0\.\d)\d+\.\d+\.\d+(?:\.\d+)?\b/;

/** A test title is a plain sentence about behavior, with no id, row code or clause number. */
export function checkTestTitles(files: Files): Violation[] {
  const found: Violation[] = [];
  for (const [file, text] of files) {
    if (!isTest(file) || !isTypeScript(file)) continue;
    linesOf(text).forEach((line, index) => {
      const title = TITLE.exec(line)?.[2];
      const code = title === undefined ? null : TITLE_CODE.exec(title);
      if (code !== null)
        found.push({
          file,
          line: index + 1,
          rule: "test-title",
          message: `The title "${title}" holds the code ${code[0]}; say it in words.`,
        });
    });
  }
  return found;
}

/** Narrative of how the code came to be, which belongs in the commit and never in the text. */
const HISTORY =
  /\b(?:previously|formerly|used to be|renamed|now (?:accepts|returns|reads|throws|uses|holds|runs|has|is|are|checks|rejects|refuses|writes|sends|supports|takes|gives|lists|stores)|we (?:found|decided|added|removed|changed|moved|fixed|chose|used)|(?:was|were) (?:removed|renamed|replaced|moved|added)|earlier versions?|in the past)\b/i;

/** Docs, and the comments of the code, state the current behavior only: no history. */
export function checkHistory(files: Files): Violation[] {
  const found: Violation[] = [];
  const report = (file: string, line: number, text: string): void => {
    const match = HISTORY.exec(text);
    if (match !== null)
      found.push({
        file,
        line,
        rule: "history",
        message: `"${match[0]}" tells how the text came to be; state the current behavior.`,
      });
  };
  for (const [file, text] of files) {
    if (isMarkdown(file)) for (const [line, prose] of proseLines(text)) report(file, line, prose);
    else if (isTypeScript(file) && (isLibrary(file) || isTest(file)))
      for (const [line, comment] of commentLines(text)) report(file, line, comment);
  }
  return found;
}

/** A file name says which concept the file holds. */
export function checkFileNames(files: Files): Violation[] {
  return [...files.keys()]
    .filter((file) => VAGUE_FILE_NAMES.has(basename(file, extname(file))))
    .map((file) => ({
      file,
      line: 0,
      rule: "file-name",
      message: "Name the file after the one concept it holds, never utils, helpers or misc.",
    }));
}

/** Whether a Markdown file is a README or a guide, which opens with a picture. */
function needsDiagram(file: string): boolean {
  return (
    isMarkdown(file) &&
    (basename(file) === "README.md" || basename(file) === "CONTRIBUTING.md" || /^docs\/[^/]+\/[^/]+\.md$/.test(file))
  );
}

/** Every README and guide has a Mermaid diagram: visualization first, then tables, then prose. */
export function checkDiagrams(files: Files): Violation[] {
  return [...files]
    .filter(([file, text]) => needsDiagram(file) && !/^```mermaid$/m.test(text))
    .map(([file]) => ({
      file,
      line: 0,
      rule: "diagram",
      message: "Add a Mermaid diagram of the architecture, flow or states that the document describes.",
    }));
}

/** The brand's error is the base class; a package that has an error class of its own breaks the rule of one family of errors. */
export function checkErrorNames(files: Files): Violation[] {
  const packages = new Set(
    [...files.keys()]
      .flatMap((file) => /^packages\/([^/]+)\//.exec(file)?.[1] ?? [])
      .map((name) => name.replaceAll("-", "")),
  );
  const found: Violation[] = [];
  for (const [file, text] of files) {
    if (!isTypeScript(file) || !isLibrary(file)) continue;
    linesOf(text).forEach((line, index) => {
      const name = /\bclass\s+([A-Za-z]+)Error\b/.exec(line)?.[1];
      if (name !== undefined && packages.has(name.toLowerCase()))
        found.push({
          file,
          line: index + 1,
          rule: "package-error",
          message: `${name}Error is named after a package; extend the brand's error and name the failure instead.`,
        });
    });
  }
  return found;
}

/** The module specifiers that a TypeScript file imports or re-exports, with their lines. */
function specifiersOf(text: string): [number, string][] {
  const found: [number, string][] = [];
  linesOf(text).forEach((line, index) => {
    const specifier = /(?:from|import)\s*\(?\s*["']([^"']+)["']/.exec(line)?.[1];
    if (specifier !== undefined) found.push([index + 1, specifier]);
  });
  return found;
}

/** The package that holds a file, as the name its scope gives it, or `undefined` for a file outside every package. */
function packageOf(file: string): string | undefined {
  const folder = /^packages\/([^/]+)\//.exec(file)?.[1];
  return folder === undefined ? undefined : `@measurekit/${folder}`;
}

/**
 * Packages of another family are never imported, apart from the examples that adapt them, and a library imports no other
 * library of this family: they meet in core.
 */
export function checkFamilies(files: Files): Violation[] {
  const found: Violation[] = [];
  for (const [file, text] of files) {
    if (!isTypeScript(file)) continue;
    const own = packageOf(file);
    for (const [line, specifier] of specifiersOf(text)) {
      const foreign = !FOREIGN_ALLOWED.test(file) && FOREIGN_SCOPES.some((prefix) => specifier.startsWith(prefix));
      const sibling =
        own !== undefined &&
        specifier.startsWith("@measurekit/") &&
        specifier !== SHARED_PACKAGE &&
        !specifier.startsWith(`${SHARED_PACKAGE}/`) &&
        specifier !== own &&
        !specifier.startsWith(`${own}/`);
      if (foreign)
        found.push({
          file,
          line,
          rule: "family",
          message: `${specifier} belongs to another family; define a port or speak an open standard, and put the adapter in examples.`,
        });
      else if (sibling)
        found.push({
          file,
          line,
          rule: "family",
          message: `${specifier} is another library of this family; the libraries share @measurekit/core and do not import each other.`,
        });
    }
  }
  return found;
}

/**
 * The module that holds a file: the nearest directory with an `index.ts` or, for a module that only the package uses, an
 * `internal.ts`; `undefined` for a file outside every module.
 */
function moduleOf(file: string, files: Files): string | undefined {
  for (
    let directory = dirname(file);
    basename(directory) !== "src" && directory !== ".";
    directory = dirname(directory)
  ) {
    if (files.has(`${directory}/index.ts`) || files.has(`${directory}/internal.ts`)) return directory;
  }
  return undefined;
}

/** The module that a relative import of a file reaches when it goes past the entries of that module, if any. */
function deepTarget(file: string, specifier: string, files: Files): string | undefined {
  const target = posix.normalize(posix.join(dirname(file), specifier.replace(/\.js$/, ".ts")));
  const targetModule = moduleOf(target, files);
  if (targetModule === undefined || targetModule === moduleOf(file, files)) return undefined;
  const isEntry =
    ["index.ts", "internal.ts"].includes(posix.basename(target)) && posix.dirname(target) === targetModule;
  return isEntry ? undefined : targetModule;
}

/** A module reaches another only through its `index.ts` or `internal.ts`, never by a deep path. */
export function checkEntries(files: Files): Violation[] {
  const found: Violation[] = [];
  for (const [file, text] of files) {
    if (!isTypeScript(file) || !/^packages\/[^/]+\/(src|test)\//.test(file)) continue;
    for (const [line, specifier] of specifiersOf(text)) {
      const reached = specifier.startsWith(".") ? deepTarget(file, specifier, files) : undefined;
      if (reached !== undefined)
        found.push({
          file,
          line,
          rule: "entry",
          message: `${specifier} reaches into ${reached}; import its index.js or internal.js.`,
        });
    }
  }
  return found;
}

/** A library never writes to the console: the caller passes a logger, or reads what the library returns. */
export function checkConsole(files: Files): Violation[] {
  return [...files].flatMap(([file, text]) =>
    isLibrary(file)
      ? matchLines(
          file,
          text,
          "console",
          /\bconsole\.\w+/,
          (call) => `${call} writes to the console; library code takes a logger or returns the result.`,
        )
      : [],
  );
}

/** The pages of `docs/<package>/` that are not the guide of a module. */
const OTHER_PAGES = new Set([
  "architecture",
  "choosing",
  "coming-from-pybkt",
  "coming-from-r",
  "evidence",
  "performance",
  "storage",
  "supported",
  "validation",
]);
/** The sections that the guide of every module has, by the text of their `##` heading. */
const GUIDE_SECTIONS = ["Quick start", "Errors", "Not included", "Differences from the references"];
/** The first row of the table of what a page leaves out, as every page writes it. */
const NOT_INCLUDED_HEADER = "| Not included | Why |";

/** The `##` headings of a Markdown file outside fenced code, with the 1-based line of each. */
function sectionsOf(text: string): { readonly title: string; readonly line: number }[] {
  const found: { title: string; line: number }[] = [];
  let fenced = false;
  linesOf(text).forEach((line, index) => {
    if (/^\s*```/.test(line)) fenced = !fenced;
    const title = fenced ? undefined : /^## (.+?)\s*$/.exec(line)?.[1];
    if (title !== undefined) found.push({ title, line: index + 1 });
  });
  return found;
}

/** The sections of the guide of a module that its page lacks. */
function missingGuideSections(file: string, sections: readonly { readonly title: string }[]): Violation[] {
  return GUIDE_SECTIONS.filter((title) => !sections.some((section) => section.title === title)).map((title) => ({
    file,
    line: 0,
    rule: "guide-sections",
    message: `Every guide of a module has a "## ${title}" section; add it, with "None" and the reason where there is nothing to say.`,
  }));
}

/** The cells of a row of a Markdown table, split at the pipes that are not escaped. */
function cellsOf(row: string): string[] {
  return row
    .trim()
    .replace(/^\||\|$/g, "")
    .split(/(?<!\\)\|/)
    .map((cell) => cell.trim());
}

/** The lines of a section that are not blank, as `[1-based line, text]`, from its heading to the next heading of its level. */
function sectionBody(text: string, heading: number): [number, string][] {
  const body: [number, string][] = [];
  for (const [index, line] of linesOf(text).slice(heading).entries()) {
    if (/^#{1,2} /.test(line)) break;
    if (line.trim() !== "") body.push([heading + index + 1, line]);
  }
  return body;
}

/** A row of the table of what a page leaves out whose Why does not open with a reason of the three and a colon. */
const WITHOUT_REASON = new RegExp(`^(?!(?:${REASONS.join("|")}): \\S)`);

/** The breaches of one `## Not included` section: a first row other than the header, a line out of the table, a Why without its reason. */
function oddNotIncluded(file: string, text: string, heading: number): Violation[] {
  const violation = (line: number, message: string): Violation => ({ file, line, rule: "not-included", message });
  const [header, delimiter, ...rows] = sectionBody(text, heading);
  if (header === undefined || header[1].replace(/\s+/g, " ").trim() !== NOT_INCLUDED_HEADER)
    return [
      violation(
        header?.[0] ?? heading,
        `Write what the page leaves out as a table whose first row reads ${NOT_INCLUDED_HEADER}.`,
      ),
    ];
  if (delimiter === undefined || !/^\|(\s*:?-+:?\s*\|){2}$/.test(delimiter[1].trim()))
    return [violation(delimiter?.[0] ?? header[0], "Follow the first row of the table with its delimiter row.")];
  return rows.flatMap(([line, row]) => {
    const cells = row.startsWith("|") ? cellsOf(row) : [];
    return cells.length === 2 && cells[0] !== "" && !WITHOUT_REASON.test(cells[1] ?? "")
      ? []
      : [
          violation(
            line,
            `A row names one method and why: | Method | Reason: why |, with Reason one of ${REASONS.join(", ")}.`,
          ),
        ];
  });
}

/**
 * The guide of every module has a quick start, the errors, what is not included and the differences from the references, so
 * that the guides read alike.
 */
export function checkGuideSections(files: Files): Violation[] {
  return [...files].flatMap(([file, text]) => {
    const page = /^docs\/[^/]+\/([^/]+)\.md$/.exec(file)?.[1];
    return page === undefined || OTHER_PAGES.has(page) ? [] : missingGuideSections(file, sectionsOf(text));
  });
}

/**
 * What a page of `docs/<package>/` leaves out is one table that opens with `| Not included | Why |`, a row for each method,
 * whose Why opens with one of the three reasons that a method stays out of a module: `| Method | Reason: why |`.
 */
export function checkNotIncluded(files: Files): Violation[] {
  return [...files].flatMap(([file, text]) =>
    /^docs\/[^/]+\/[^/]+\.md$/.test(file)
      ? sectionsOf(text)
          .filter(({ title }) => title === "Not included")
          .flatMap(({ line }) => oddNotIncluded(file, text, line))
      : [],
  );
}

/**
 * The sections of a README, in order: the root's guide the reader to a package, and each package's is its npm page, which gives
 * the install line and a minimal example before the diagram, since npm shows a Mermaid diagram as its source.
 */
const README_SECTIONS: readonly { readonly pattern: RegExp; readonly sections: readonly string[] }[] = [
  { pattern: /^README\.md$/, sections: ["Which package", "Documentation", "Development", "Layout", "License"] },
  {
    pattern: /^packages\/[^/]+\/README\.md$/,
    sections: ["Quick start", "Architecture", "API", "Supported", "Documentation", "License"],
  },
];

/** Every README has the sections of its kind, in their order, with others allowed between them. */
export function checkReadmeSections(files: Files): Violation[] {
  const found: Violation[] = [];
  for (const [file, text] of files) {
    const required = README_SECTIONS.find(({ pattern }) => pattern.test(file))?.sections;
    if (required === undefined) continue;
    const titles = sectionsOf(text).map(({ title }) => title);
    let from = 0;
    for (const title of required) {
      const at = titles.indexOf(title, from);
      if (at < 0)
        found.push({
          file,
          line: 0,
          rule: "readme-sections",
          message: `A README has the sections ${required.map((name) => `"${name}"`).join(", ")} in this order; "${title}" is missing or comes too early.`,
        });
      else from = at + 1;
    }
  }
  return found;
}

/** The types of the standard library that are mutable lists or records, and so are not a result. */
const MUTABLE_CONTAINERS = new Set(["Array", "Record", "Map", "Set"]);

/** The part of a return type that a caller could write to, or `undefined` when every list and record in it is readonly. */
function mutablePart(node: ts.TypeNode): string | undefined {
  if (ts.isParenthesizedTypeNode(node)) return mutablePart(node.type);
  if (ts.isUnionTypeNode(node)) return node.types.map(mutablePart).find((part) => part !== undefined);
  if (ts.isArrayTypeNode(node)) return node.getText();
  if (ts.isTypeOperatorNode(node))
    return ts.isArrayTypeNode(node.type) ? mutablePart(node.type.elementType) : undefined;
  if (!ts.isTypeReferenceNode(node)) return undefined;
  const name = node.typeName.getText();
  if (MUTABLE_CONTAINERS.has(name)) return node.getText();
  // A readonly record holds readonly values; a promise or a readonly array holds what it wraps.
  const wrapped = ["Readonly", "ReadonlyArray", "Promise"].includes(name) ? node.typeArguments?.at(-1) : undefined;
  if (wrapped === undefined) return undefined;
  return mutablePart(
    ts.isTypeReferenceNode(wrapped) && wrapped.typeName.getText() === "Record"
      ? (wrapped.typeArguments?.at(-1) ?? wrapped)
      : wrapped,
  );
}

/** The value names that a re-export of an `index.ts` takes from the file at its relative specifier, if it is one. */
function reexportedValues(file: string, statement: ts.Statement): { target: string; names: string[] } | undefined {
  if (!ts.isExportDeclaration(statement) || statement.isTypeOnly) return undefined;
  const { moduleSpecifier, exportClause } = statement;
  if (moduleSpecifier === undefined || exportClause === undefined || !ts.isNamedExports(exportClause)) return undefined;
  const specifier = (moduleSpecifier as ts.StringLiteral).text;
  if (!specifier.startsWith(".")) return undefined;
  return {
    target: posix.normalize(posix.join(dirname(file), specifier.replace(/\.js$/, ".ts"))),
    names: exportClause.elements
      .filter((element) => !element.isTypeOnly)
      .map((element) => (element.propertyName ?? element.name).text),
  };
}

/** The names that the `index.ts` of each module exports as values, by the path of the file that declares them. */
function publicValues(files: Files): Map<string, Set<string>> {
  const exported = new Map<string, Set<string>>();
  for (const [file, text] of files) {
    if (!/^packages\/[^/]+\/src\/(?:[^/]+\/)?index\.ts$/.test(file)) continue;
    for (const statement of ts.createSourceFile(file, text, ts.ScriptTarget.ES2022).statements) {
      const reexport = reexportedValues(file, statement);
      if (reexport === undefined) continue;
      const names = exported.get(reexport.target) ?? new Set<string>();
      for (const name of reexport.names) names.add(name);
      exported.set(reexport.target, names);
    }
  }
  return exported;
}

/**
 * A public function returns plain data whose lists and records are `readonly`, at the top level as inside objects: a
 * caller who sorts in place copies first, and no function hands out an array it also keeps.
 */
export function checkReadonlyResults(files: Files): Violation[] {
  const found: Violation[] = [];
  for (const [file, names] of publicValues(files)) {
    const text = files.get(file);
    if (text === undefined) continue;
    const source = ts.createSourceFile(file, text, ts.ScriptTarget.ES2022, true);
    for (const statement of source.statements) {
      if (!ts.isFunctionDeclaration(statement) || statement.name === undefined || statement.type === undefined)
        continue;
      const mutable = names.has(statement.name.text) ? mutablePart(statement.type) : undefined;
      if (mutable !== undefined)
        found.push({
          file,
          line: source.getLineAndCharacterOfPosition(statement.type.getStart(source)).line + 1,
          rule: "readonly-result",
          message: `${statement.name.text} returns ${mutable}; write it readonly, as a caller who sorts in place copies first.`,
        });
    }
  }
  return found;
}

/** The entries that a manifest exports, as import specifiers and source files: `./dist/<path>.js` is built from `src/<path>.ts`. */
function entriesOf(folder: string, text: string): [string, string][] {
  const manifest = manifestOf(text);
  const exported = manifest?.exports;
  if (typeof manifest?.name !== "string" || typeof exported !== "object" || exported === null) return [];
  return Object.entries(exported).flatMap(([subpath, target]): [string, string][] => {
    const built = typeof target === "object" && target !== null ? (target as { default?: unknown }).default : undefined;
    return typeof built === "string" && /^\.\/dist\/.+\.js$/.test(built)
      ? [[`${manifest.name}${subpath.slice(1)}`, `packages/${folder}/src/${built.slice(7, -3)}.ts`]]
      : [];
  });
}

/** The source file of each entry of the packages, by the import specifier of its subpath, from the `exports` of each manifest. */
function entrySources(files: Files): ReadonlyMap<string, string> {
  return new Map(
    [...files].flatMap(([file, text]) => {
      const folder = /^packages\/([^/]+)\/package\.json$/.exec(file)?.[1];
      return folder === undefined ? [] : entriesOf(folder, text);
    }),
  );
}

/** A re-export of a source file: the path of the file it takes from, and the names it takes. */
interface Reexport {
  readonly target: string;
  readonly names: ReadonlySet<string> | "all";
}

/** The source file of a specifier: a relative path from the file, or an entry of a package of the family; undefined for any other package. */
function sourceOf(file: string, specifier: string, entries: ReadonlyMap<string, string>): string | undefined {
  if (specifier.startsWith(".")) return posix.normalize(posix.join(dirname(file), specifier.replace(/\.js$/, ".ts")));
  return entries.get(specifier);
}

/** The re-exports of values in a source file, relative or from a package of the family. */
function reexportsOf(file: string, source: ts.SourceFile, entries: ReadonlyMap<string, string>): Reexport[] {
  return source.statements.flatMap((statement): Reexport[] => {
    if (!ts.isExportDeclaration(statement) || statement.isTypeOnly) return [];
    const specifier = (statement.moduleSpecifier as ts.StringLiteral | undefined)?.text;
    const target = specifier === undefined ? undefined : sourceOf(file, specifier, entries);
    if (target === undefined) return [];
    const { exportClause } = statement;
    if (exportClause === undefined) return [{ target, names: "all" }];
    if (!ts.isNamedExports(exportClause)) return [];
    const names = exportClause.elements
      .filter((element) => !element.isTypeOnly)
      .map((element) => (element.propertyName ?? element.name).text);
    return [{ target, names: new Set(names) }];
  });
}

/** The tag `@example` at the start of a line of a comment, and not an address such as `learner@example.org` in its text. */
const EXAMPLE_TAG = /^[ \t]*\*[ \t]*@example\b/m;

/** Whether a source file declares an exported function, among the names given, whose TSDoc has an `@example`. */
function declaresExample(source: ts.SourceFile, names: ReadonlySet<string> | "all"): boolean {
  return source.statements.some(
    (statement) =>
      ts.isFunctionDeclaration(statement) &&
      statement.name !== undefined &&
      isExported(statement) &&
      (names === "all" || names.has(statement.name.text)) &&
      EXAMPLE_TAG.test(source.text.slice(statement.getFullStart(), statement.getStart(source))),
  );
}

/** Whether what a file exports includes a function with an `@example`, through the modules it re-exports. */
function exportsExample(
  entry: string,
  files: Files,
  entries: ReadonlyMap<string, string>,
  names: ReadonlySet<string> | "all" = "all",
  reading: ReadonlySet<string> = new Set(),
): boolean {
  const text = files.get(entry);
  if (text === undefined || reading.has(entry)) return false;
  const source = ts.createSourceFile(entry, text, ts.ScriptTarget.ES2022, true);
  if (declaresExample(source, names)) return true;
  return reexportsOf(entry, source, entries).some(({ target, names: taken }) => {
    const wanted =
      names === "all" ? taken : taken === "all" ? names : new Set([...taken].filter((name) => names.has(name)));
    return exportsExample(target, files, entries, wanted, new Set([...reading, entry]));
  });
}

/**
 * The entry of every subpath of a package, as its manifest exports it, exports a function whose TSDoc has an `@example`: the
 * main function of the module. The samples test of each package runs every example and compares the results that it shows.
 */
export function checkModuleExamples(files: Files): Violation[] {
  const entries = entrySources(files);
  return [...entries]
    .filter(([, entry]) => files.has(entry) && !exportsExample(entry, files, entries))
    .map(([specifier, entry]) => ({
      file: entry,
      line: 0,
      rule: "module-example",
      message: `The subpath ${specifier} exports no function whose TSDoc has an @example; put one on the main function of the module.`,
    }));
}

/** Tells whether a node carries the `export` keyword. */
function isExported(node: ts.Node): boolean {
  return (
    ts.canHaveModifiers(node) && (ts.getModifiers(node) ?? []).some(({ kind }) => kind === ts.SyntaxKind.ExportKeyword)
  );
}

/** The superscript of each digit, as SECURITY.md writes an exponent. */
const SUPERSCRIPTS = "⁰¹²³⁴⁵⁶⁷⁸⁹";

/** The constants of a file of a package, by name, with the line of each. */
function constantsOf(
  source: ts.SourceFile,
): Map<string, { readonly initializer: ts.Expression; readonly line: number }> {
  const constants = new Map<string, { readonly initializer: ts.Expression; readonly line: number }>();
  for (const statement of source.statements) {
    if (!ts.isVariableStatement(statement)) continue;
    for (const { name, initializer } of statement.declarationList.declarations) {
      if (!ts.isIdentifier(name) || initializer === undefined) continue;
      constants.set(name.text, {
        initializer,
        line: source.getLineAndCharacterOfPosition(name.getStart(source)).line + 1,
      });
    }
  }
  return constants;
}

/**
 * The value of a numeric constant: a literal, or the arithmetic of literals and of other constants, of its file or, for one
 * it imports, exported by a file of the packages.
 */
function numberOf(node: ts.Expression, constants: ReadonlyMap<string, ts.Expression>): number | undefined {
  if (ts.isNumericLiteral(node)) return Number(node.text.replaceAll("_", ""));
  if (ts.isParenthesizedExpression(node)) return numberOf(node.expression, constants);
  if (ts.isIdentifier(node)) {
    const initializer = constants.get(node.text);
    return initializer === undefined ? undefined : numberOf(initializer, constants);
  }
  if (!ts.isBinaryExpression(node)) return undefined;
  const left = numberOf(node.left, constants);
  const right = numberOf(node.right, constants);
  if (left === undefined || right === undefined) return undefined;
  const operations: Partial<Record<ts.SyntaxKind, number>> = {
    [ts.SyntaxKind.AsteriskAsteriskToken]: left ** right,
    [ts.SyntaxKind.AsteriskToken]: left * right,
    [ts.SyntaxKind.SlashToken]: left / right,
    [ts.SyntaxKind.PlusToken]: left + right,
    [ts.SyntaxKind.MinusToken]: left - right,
  };
  return operations[node.operatorToken.kind];
}

/** The ways SECURITY.md writes a number: 1,000; 2^28 and 2²⁸; 2³¹ - 1; 10⁶ and 1e7; 24 million. */
function writtenForms(value: number): string[] {
  const superscript = (exponent: number): string =>
    [...String(exponent)].map((digit) => SUPERSCRIPTS[Number(digit)]).join("");
  const forms = [value.toLocaleString("en-US")];
  if (value >= 1e6 && value % 1e6 === 0) forms.push(`${value / 1e6} million`);
  const two = Math.log2(value);
  if (Number.isInteger(two) && two >= 10) forms.push(`2^${two}`, `2${superscript(two)}`);
  const below = Math.log2(value + 1);
  if (Number.isInteger(below) && below >= 10) forms.push(`2^${below} - 1`, `2${superscript(below)} - 1`);
  const ten = Math.log10(value);
  if (Number.isInteger(ten) && ten >= 6) forms.push(`10${superscript(ten)}`, `1e${ten}`);
  return forms;
}

/** The name of the option whose default a constant `DEFAULT_MAX_*` is: `maxStates` for `DEFAULT_MAX_STATES`. */
function optionOf(name: string): string {
  return name
    .slice("DEFAULT_".length)
    .toLowerCase()
    .replace(/_(\w)/g, (_, letter: string) => letter.toUpperCase());
}

/**
 * The limits of a file of a package, each with its line and value: each constant `MAX_*`, of the file or exported by
 * another, that a `DataError` names, as its message states the limit, and the default `DEFAULT_MAX_*` of each option `max*` that a `DataError` names,
 * as passing it is refused; an option such as `maxIterations` ends a loop and refuses nothing.
 */
function limitsOf(
  file: string,
  text: string,
  shared: ReadonlyMap<string, ts.Expression>,
): { line: number; name: string; value: number | undefined }[] {
  const source = ts.createSourceFile(file, text, ts.ScriptTarget.Latest, true);
  const declared = constantsOf(source);
  const constants = new Map([
    ...shared,
    ...[...declared].map(([name, { initializer }]) => [name, initializer] as const),
  ]);
  const named = new Set<string>();
  const lines = new Map([...declared].map(([name, { line }]) => [name, line] as const));
  const errors: string[] = [];
  const visit = (node: ts.Node, inError: boolean): void => {
    const error = inError || (ts.isNewExpression(node) && node.expression.getText(source) === "DataError");
    if (error && !inError) errors.push(node.getText(source));
    if (error && ts.isIdentifier(node) && node.text.startsWith("MAX_") && constants.has(node.text)) {
      named.add(node.text);
      lines.set(
        node.text,
        lines.get(node.text) ?? source.getLineAndCharacterOfPosition(node.getStart(source)).line + 1,
      );
    }
    ts.forEachChild(node, (child) => visit(child, error));
  };
  visit(source, false);
  for (const name of declared.keys()) {
    if (name.startsWith("DEFAULT_MAX_") && errors.some((error) => error.includes(optionOf(name)))) named.add(name);
  }
  return [...named].map((name) => {
    const initializer = constants.get(name);
    return { line: lines.get(name) ?? 0, name, value: initializer && numberOf(initializer, constants) };
  });
}

/** The constants `MAX_*` that the files of the packages export, which another file may import and name in a `DataError`. */
function exportedLimits(files: Files): Map<string, ts.Expression> {
  const exported = new Map<string, ts.Expression>();
  for (const [file, text] of files) {
    if (!isLibrary(file) || !/export const MAX_/.test(text)) continue;
    const source = ts.createSourceFile(file, text, ts.ScriptTarget.Latest, true);
    for (const [name, { initializer }] of constantsOf(source))
      if (name.startsWith("MAX_")) exported.set(name, initializer);
  }
  return exported;
}

/** Returns whether the section Default limits of SECURITY.md writes a number in one of its forms, not inside another number. */
function statesNumber(security: string): (value: number) => boolean {
  const heading = security.indexOf("\n## Default limits");
  const table = heading < 0 ? "" : security.slice(heading, security.indexOf("\n## ", heading + 1) >>> 0);
  const literal = (form: string): string => form.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  return (value) =>
    writtenForms(value).some((form) => new RegExp(`(?<![\\d.^]|\\d,)${literal(form)}(?!\\d|,\\d)`).test(table));
}

/**
 * SECURITY.md states every limit of the packages in its section Default limits, so that a limit added without its row, or
 * the file emptied by a merge, fails: the value of each constant `MAX_*` that a `DataError` names, and of the default
 * `DEFAULT_MAX_*` of each option that one names, in one of the ways the table writes a number.
 */
export function checkSecurityLimits(files: Files): Violation[] {
  const states = statesNumber(files.get("SECURITY.md") ?? "");
  const exported = exportedLimits(files);
  return [...files].flatMap(([file, text]) =>
    isLibrary(file) && /\bMAX_/.test(text)
      ? limitsOf(file, text, exported)
          .filter(({ value }) => value === undefined || !states(value))
          .map(({ line, name, value }) => ({
            file,
            line,
            rule: "security-limit",
            message: `The limit ${name}${value === undefined ? "" : `, ${value},`} is not stated in the Default limits of SECURITY.md; give it a row with its value.`,
          }))
      : [],
  );
}

/** Every check, in the order of the standards. */
export const CHECKS: readonly ((files: Files) => Violation[])[] = [
  checkSectionSigns,
  checkTestTitles,
  checkHistory,
  checkFileNames,
  checkDiagrams,
  checkErrorNames,
  checkFamilies,
  checkEntries,
  checkConsole,
  checkGuideSections,
  checkNotIncluded,
  checkReadmeSections,
  checkReadonlyResults,
  checkModuleExamples,
  checkIdentity,
  checkTrademarks,
  checkClaimWords,
  checkReadmeLinks,
  checkEvidencePages,
  checkSecurityLimits,
];

/** Runs every check over the files. */
export function checkStandards(files: Files): Violation[] {
  return CHECKS.flatMap((check) => check(files));
}

/** Reads the text files of the repository that git tracks or would track, with the manifests, NOTICE files and the files of the identity, without the generated and third-party ones. */
export function readRepository(root: string): Files {
  const listing = execFileSync("git", ["ls-files", "-z", "--cached", "--others", "--exclude-standard"], {
    cwd: root,
    encoding: "utf8",
    maxBuffer: 64 * 1024 * 1024,
  });
  const files = new Map<string, string>();
  for (const file of listing.split("\0")) {
    if (
      file === "" ||
      EXCLUDED.test(file) ||
      EXEMPT.test(file) ||
      !(
        TEXT_EXTENSIONS.has(extname(file)) ||
        ["package.json", "NOTICE", "context7.json", "CITATION.cff"].includes(basename(file))
      )
    )
      continue;
    try {
      files.set(file, readFileSync(join(root, file), "utf8"));
    } catch (error) {
      // A file that git lists and the working tree lacks is a deletion that is not staged yet.
      if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
    }
  }
  return files;
}

/** Prints the violations of the files below the given paths, or of every file, and returns the exit code. */
function main(paths: readonly string[]): number {
  const violations = checkStandards(readRepository(resolve(import.meta.dirname, ".."))).filter(
    ({ file }) => paths.length === 0 || paths.some((path) => file.startsWith(path)),
  );
  for (const { file, line, rule, message } of violations)
    console.error(`${file}${line > 0 ? `:${line}` : ""} [${rule}] ${message}`);
  console.log(violations.length === 0 ? "The standards hold." : `${violations.length} violations of the standards.`);
  return violations.length === 0 ? 0 : 1;
}

if (import.meta.main) process.exitCode = main(process.argv.slice(2));
