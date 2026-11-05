import { readFileSync } from "node:fs";
import { resolve, sep } from "node:path";
import ts from "typescript";
import { describe, expect, it } from "vitest";
import { entrySources } from "../entry-sizes.ts";

const PACKAGES = resolve(import.meta.dirname, "../../packages");
const CORE = resolve(PACKAGES, "core") + sep;
const LIBRARIES = ["psychometrics", "knowledge-tracing"] as const;
/** What each root holds besides the types of core that its API names: the generator and the errors of core. */
const ROOT_NAMES = ["createRandom", "DataError", "MeasurekitError", "Random"];
/** Type-checking both libraries twice takes about 20 seconds, longer on a loaded machine. */
const TIMEOUT = 180_000;

const OPTIONS: ts.CompilerOptions = {
  ...ts.convertCompilerOptionsFromJson(
    JSON.parse(readFileSync(resolve(PACKAGES, "../tsconfig.base.json"), "utf8")).compilerOptions,
    PACKAGES,
  ).options,
  declarationMap: false,
  sourceMap: false,
  emitDeclarationOnly: true,
  paths: { "@measurekit/core": [resolve(CORE, "src/index.ts")] },
};

/**
 * Emits the declarations of every entry point of the libraries and of core, as a build writes them, into a map by the
 * path of each `.d.ts`, and returns a checker over them alone: a host that serves those files and hides the sources.
 */
function declarations(): { readonly checker: ts.TypeChecker; readonly program: ts.Program } {
  const sources = [...entrySources()].filter(([specifier]) => !specifier.startsWith("@measurekit/core"));
  const emitted = new Map<string, string>();
  const program = ts.createProgram(
    sources.map(([, source]) => source),
    OPTIONS,
  );
  const { diagnostics } = program.emit(undefined, (file, text) => emitted.set(resolve(file), text), undefined, true);
  if (diagnostics.length > 0) throw new Error(ts.formatDiagnostics(diagnostics, ts.createCompilerHost(OPTIONS)));
  const host = ts.createCompilerHost(OPTIONS);
  const isSource = (file: string) => resolve(file).startsWith(PACKAGES) && !file.endsWith(".d.ts");
  const readFile = host.readFile.bind(host);
  const fileExists = host.fileExists.bind(host);
  host.fileExists = (file) => emitted.has(resolve(file)) || (!isSource(file) && fileExists(file));
  host.readFile = (file) => emitted.get(resolve(file)) ?? (isSource(file) ? undefined : readFile(file));
  host.getSourceFile = (file, language) => {
    const text = host.readFile(file);
    return text === undefined ? undefined : ts.createSourceFile(file, text, language);
  };
  const declared = ts.createProgram(
    sources.map(([, source]) => source.replace(/\.ts$/, ".d.ts")),
    { ...OPTIONS, noEmit: true, paths: { "@measurekit/core": [resolve(CORE, "src/index.d.ts")] } },
    host,
  );
  return { checker: declared.getTypeChecker(), program: declared };
}

/**
 * The names of core that the declarations of a library's API name, following every type of the library that a public
 * declaration names, and the names of core that its root exports.
 */
function coreNames(library: string): { readonly named: ReadonlySet<string>; readonly root: ReadonlySet<string> } {
  const { checker, program } = DECLARATIONS;
  const folder = resolve(PACKAGES, library) + sep;
  const rootFile = resolve(folder, "src/index.d.ts");
  const named = new Set<string>();
  const root = new Set<string>();
  const seen = new Set<ts.Symbol>();
  const queue: ts.Symbol[] = [];
  const resolved = (symbol: ts.Symbol) =>
    symbol.flags & ts.SymbolFlags.Alias ? checker.getAliasedSymbol(symbol) : symbol;
  const isIn = (symbol: ts.Symbol, path: string) =>
    symbol.declarations?.some((declaration) => resolve(declaration.getSourceFile().fileName).startsWith(path)) ?? false;
  const follow = (symbol: ts.Symbol | undefined) => {
    if (symbol === undefined) return;
    const target = resolved(symbol);
    if (isIn(target, CORE)) named.add(target.name);
    else if (isIn(target, folder) && !seen.has(target)) {
      seen.add(target);
      queue.push(target);
    }
  };
  const rightmost = (name: ts.EntityName | ts.Expression): ts.Node =>
    ts.isQualifiedName(name) ? name.right : ts.isPropertyAccessExpression(name) ? name.name : name;
  const visit = (node: ts.Node): void => {
    if (ts.isTypeReferenceNode(node)) follow(checker.getSymbolAtLocation(rightmost(node.typeName)));
    else if (ts.isExpressionWithTypeArguments(node)) follow(checker.getSymbolAtLocation(rightmost(node.expression)));
    else if (ts.isTypeQueryNode(node)) follow(checker.getSymbolAtLocation(rightmost(node.exprName)));
    else if (ts.isImportTypeNode(node) && node.qualifier !== undefined) {
      follow(checker.getSymbolAtLocation(rightmost(node.qualifier)));
    }
    ts.forEachChild(node, visit);
  };
  for (const file of program.getRootFileNames()) {
    if (!resolve(file).startsWith(folder)) continue;
    const source = program.getSourceFile(file);
    const module = source === undefined ? undefined : checker.getSymbolAtLocation(source);
    for (const exported of module === undefined ? [] : checker.getExportsOfModule(module)) {
      const target = resolved(exported);
      if (resolve(file) === rootFile && isIn(target, CORE)) root.add(target.name);
      else follow(exported);
    }
  }
  for (let symbol = queue.pop(); symbol !== undefined; symbol = queue.pop()) {
    for (const declaration of symbol.declarations ?? []) {
      if (resolve(declaration.getSourceFile().fileName).startsWith(folder)) visit(declaration);
    }
  }
  return { named, root };
}

const DECLARATIONS = declarations();

describe("the root of a library", () => {
  // A user installs one library, and pnpm does not let a program import a dependency of a dependency: every type of core
  // that the library's API names is exported by its root, beside the generator and the errors.
  it.each(LIBRARIES)(
    "of %s exports the generator and the errors of core, and every type of core its declarations name, and nothing else of core",
    (library) => {
      const { named, root } = coreNames(library);
      expect([...root].sort()).toEqual([...new Set([...ROOT_NAMES, ...named])].sort());
    },
    TIMEOUT,
  );
});
