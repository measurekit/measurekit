// The size of every entry point of the packages bundled alone, as a host's bundler makes it, with code splitting on, and
// the modules it holds.
// `test/entry-sizes.test.ts` checks them against `size-budgets.ts`; `node scripts/entry-sizes.ts` prints them as the table
// of docs/performance.md.

import { readFileSync } from "node:fs";
import { relative, resolve } from "node:path";
import { gzipSync } from "node:zlib";
import { build } from "esbuild";

const PACKAGES = resolve(import.meta.dirname, "../packages");
const PACKAGE_FOLDERS = ["core", "psychometrics", "knowledge-tracing"] as const;
/** The language level of the sources, `target` of tsconfig.base.json. */
const TARGET = "es2022";
/** The best level of gzip, which is what a server sends. */
const GZIP_LEVEL = 9;

/** The measures of an entry point bundled alone. */
export interface EntrySize {
  /** The bytes of the minified bundle. */
  readonly raw: number;
  /** The bytes of the minified bundle after gzip. */
  readonly gzip: number;
  /** The modules other than its own that have code in the bundle, sorted. */
  readonly holds: readonly string[];
}

/**
 * Returns the source file of every entry point of the packages, by its specifier, from the `exports` of each package's
 * manifest: `./dist/<path>.js` is built from `src/<path>.ts`.
 */
export function entrySources(): ReadonlyMap<string, string> {
  return new Map(
    PACKAGE_FOLDERS.flatMap((folder) => {
      const manifest: { name: string; exports: Record<string, { default: string }> } = JSON.parse(
        readFileSync(resolve(PACKAGES, folder, "package.json"), "utf8"),
      );
      return Object.entries(manifest.exports).map(
        ([subpath, { default: target }]) =>
          [
            `${manifest.name}${subpath.slice(1)}`,
            resolve(PACKAGES, folder, target.replace(/^\.\/dist\//, "src/").replace(/\.js$/, ".ts")),
          ] as const,
      );
    }),
  );
}

/**
 * Returns the module a source file belongs to: `core` for any file of core, which has one entry point, and
 * `<package>/<folder>` or `<package>/<file>` for a file of a library, such as `psychometrics/irt` or
 * `knowledge-tracing/model`.
 */
export function moduleOf(path: string): string {
  const [folder = "", , first = ""] = relative(PACKAGES, path).split("/");
  return folder === "core" ? "core" : `${folder}/${first.replace(/\.ts$/, "")}`;
}

/** What a bundle of a program holds: its bytes before and after gzip, and the modules with code in it, sorted. */
export interface Bundle {
  readonly raw: number;
  readonly gzip: number;
  readonly modules: readonly string[];
}

/**
 * Bundles a program that imports the sources of the packages, minified, with core from its sources, and measures it.
 * The bundler splits the code, as a host's does: a module that a dynamic `import()` loads is a chunk of its own. The
 * bytes and the modules are those of the entry chunk and the chunks that it imports statically, which load with it.
 *
 * @param program - The module to bundle; it names sources by absolute path.
 */
export async function bundle(program: string): Promise<Bundle> {
  const result = await build({
    stdin: { contents: program, resolveDir: PACKAGES, loader: "ts" },
    bundle: true,
    splitting: true,
    write: false,
    metafile: true,
    format: "esm",
    platform: "neutral",
    target: TARGET,
    minify: true,
    legalComments: "none",
    logLevel: "silent",
    outdir: "out",
    absWorkingDir: PACKAGES,
    alias: { "@measurekit/core": resolve(PACKAGES, "core/src/index.ts") },
  });
  const outputs = result.metafile.outputs;
  const entry = Object.keys(outputs).find((path) => outputs[path]?.entryPoint !== undefined);
  if (entry === undefined) throw new Error("The bundle has no output.");
  // The chunks that load with the entry: those that it, or a chunk that loads with it, imports statically.
  const eager = new Set([entry]);
  for (const path of eager) {
    for (const { path: imported, kind } of outputs[path]?.imports ?? []) {
      if (kind === "import-statement" && imported in outputs) eager.add(imported);
    }
  }
  const files = result.outputFiles.filter(({ path }) => eager.has(relative(PACKAGES, path)));
  // `inputs` lists every file the bundler read; the outputs say which of them still have code after tree shaking. The
  // program itself is not a module of the packages.
  const sources = [...eager].flatMap((path) =>
    Object.entries(outputs[path]?.inputs ?? {}).flatMap(([file, { bytesInOutput }]) =>
      bytesInOutput > 0 && file !== "<stdin>" ? [resolve(PACKAGES, file)] : [],
    ),
  );
  return {
    raw: files.reduce((sum, { contents }) => sum + contents.byteLength, 0),
    gzip: files.reduce((sum, { contents }) => sum + gzipSync(contents, { level: GZIP_LEVEL }).byteLength, 0),
    modules: [...new Set(sources.map(moduleOf))].sort(),
  };
}

/** Bundles the entry point of `source` alone, with all that it exports, and measures it. */
export async function entrySize(source: string): Promise<EntrySize> {
  const { raw, gzip, modules } = await bundle(`export * from ${JSON.stringify(source)};`);
  const own = moduleOf(source);
  return { raw, gzip, holds: modules.filter((name) => name !== own) };
}

if (import.meta.main) {
  console.log("| Entry point | Minified (bytes) | Gzip (bytes) | Holds besides its own module |");
  console.log("| ----------- | ---------------: | -----------: | ---------------------------- |");
  for (const [specifier, source] of entrySources()) {
    const { raw, gzip, holds } = await entrySize(source);
    const format = (bytes: number) => bytes.toLocaleString("en-US");
    console.log(`| \`${specifier}\` | ${format(raw)} | ${format(gzip)} | ${holds.join(", ") || "nothing"} |`);
  }
}
