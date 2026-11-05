import { readdirSync } from "node:fs";
import { defineConfig } from "vitest/config";

/** Resolves core to its source, so that the tests of the packages that depend on it run the code and not a build. */
const CORE_SOURCE = { "@measurekit/core": new URL("./packages/core/src/index.ts", import.meta.url).pathname };

/** Resolves the packages and their subpaths to their sources, for the examples that import them by name. */
const PACKAGE_SOURCES = [
  { find: /^@measurekit\/core$/, replacement: new URL("./packages/core/src/index.ts", import.meta.url).pathname },
  ...["psychometrics", "knowledge-tracing"].flatMap((name) => [
    {
      find: new RegExp(`^@measurekit/${name}$`),
      replacement: new URL(`./packages/${name}/src/index.ts`, import.meta.url).pathname,
    },
    {
      find: new RegExp(`^@measurekit/${name}/(.+)$`),
      replacement: new URL(`./packages/${name}/src/$1/index.ts`, import.meta.url).pathname,
    },
  ]),
];

/**
 * Returns the project of the benchmarks of a package. They time its build, as Node.js loads it: Vitest's module runner
 * turns each import of a source file into a getter, which stops the compiler from inlining a call across modules and
 * makes a call of a small function several times as slow as it is for a user. A benchmark imports an entry,
 * `../src/index.js` or `../src/<module>/index.js`; the project points it at the build, which `pnpm bench` makes first,
 * and keeps Vitest from transforming the build. The files run one after another, so that no benchmark competes with
 * another for the processor.
 */
function benchmarkProject(name: string, test: { readonly timeout?: number }) {
  const build = new URL(`./packages/${name}/dist/`, import.meta.url).pathname;
  return {
    resolve: { alias: [{ find: /^\.\.\/src\/((?:[^/]+\/)?index\.js)$/, replacement: `${build}$1` }] },
    test: {
      name: `${name}-bench`,
      include: [],
      testTimeout: test.timeout,
      hookTimeout: test.timeout,
      fileParallelism: false,
      server: { deps: { external: [/\/packages\/[^/]+\/dist\//] } },
      benchmark: { include: [`packages/${name}/test/**/*.bench.ts`], suppressExportGetterWarnings: true },
    },
  };
}

/** A floor of coverage, in percent of the statements, branches, functions and lines. */
interface Floor {
  readonly statements: number;
  readonly branches: number;
  readonly functions: number;
  readonly lines: number;
}

/**
 * Returns the thresholds of a package, as STANDARDS section 7 sets them: its branches over the package, as V8 counts the
 * fallback of `a[i] ?? 0`, which no test can take, as a branch, and its statements, functions and lines over each module
 * on its own, so that a module whose tests fall short fails, whatever the others reach. A module is a folder of `src/`,
 * or a file beside them: the modules of core's flat `src/`, a root's entry and the files the modules share.
 */
function packageFloors(name: string, { branches, ...perModule }: Floor): Record<string, Partial<Floor>> {
  const source = new URL(`./packages/${name}/src/`, import.meta.url);
  return {
    [`packages/${name}/src/**`]: { branches },
    ...Object.fromEntries(
      readdirSync(source, { withFileTypes: true }).map((entry) => [
        `packages/${name}/src/${entry.name}${entry.isDirectory() ? "/**" : ""}`,
        perModule,
      ]),
    ),
  };
}

/** The timeout of a benchmark that takes minutes: 30 minutes. */
const SLOW_BENCHMARK_TIMEOUT = 1_800_000;

// biome-ignore lint/style/noDefaultExport: Vitest reads its configuration from the default export.
export default defineConfig({
  test: {
    coverage: {
      provider: "v8",
      include: ["packages/*/src/**/*.ts"],
      reporter: ["text-summary"],
      // One floor for each package: its branches over the package, and its statements, functions and lines over each
      // module on its own, so that a gap in the tests of a module fails its own floor and no other. CI checks core and knowledge tracing in the job of their project on Node.js 22, with the
      // files of that package alone, and rules.yml checks psychometrics each week and before a release. Psychometrics'
      // floor is its measure on Node.js 22 rounded down; under coverage its tests take about 36 minutes, 4.6 times as long.
      thresholds: {
        ...packageFloors("core", { statements: 94, branches: 74, functions: 96, lines: 95 }),
        ...packageFloors("knowledge-tracing", { statements: 98, branches: 82, functions: 99, lines: 99 }),
        ...packageFloors("psychometrics", { statements: 99, branches: 72, functions: 99, lines: 99 }),
      },
    },
    projects: [
      {
        test: {
          name: "core",
          include: ["packages/core/test/**/*.test.ts"],
          setupFiles: ["packages/core/test/setup.ts"],
        },
      },
      {
        resolve: { alias: CORE_SOURCE },
        // A test ends within half its timeout on a runner under coverage, which runs two to three times as long as a
        // developer's machine: 60 s holds the tests of about 10 s there that STANDARDS section 7 allows, and a hook the
        // same. A test that needs more is cut, never given a longer timeout of its own.
        test: {
          name: "psychometrics",
          include: ["packages/psychometrics/test/**/*.test.ts"],
          setupFiles: ["packages/psychometrics/test/setup.ts"],
          testTimeout: 60_000,
          hookTimeout: 60_000,
        },
      },
      {
        resolve: { alias: CORE_SOURCE },
        test: {
          name: "knowledge-tracing",
          include: ["packages/knowledge-tracing/test/**/*.test.ts"],
          setupFiles: ["packages/knowledge-tracing/test/setup.ts"],
          // Fits of BLIM and the drift of IRT take a second or two, several times as long under coverage and load.
          testTimeout: 60_000,
        },
      },
      { test: { name: "scripts", include: ["scripts/test/**/*.test.ts"] } },
      {
        resolve: { alias: PACKAGE_SOURCES },
        // The examples run on the sources, as the libraries' own tests do.
        test: { name: "examples", include: ["examples/test/**/*.test.ts"], testTimeout: 120_000 },
      },
      // `pnpm bench` runs the benchmarks of each package with `vitest bench`, which gives a project the suffix "(bench)".
      benchmarkProject("core", {}),
      benchmarkProject("psychometrics", { timeout: SLOW_BENCHMARK_TIMEOUT }),
      benchmarkProject("knowledge-tracing", { timeout: SLOW_BENCHMARK_TIMEOUT }),
    ],
  },
});
