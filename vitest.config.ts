import { defineConfig } from "vitest/config";

/** Resolves core to its source, so that the tests of the packages that depend on it run the code and not a build. */
const CORE_SOURCE = { "@measurekit/core": new URL("./packages/core/src/index.ts", import.meta.url).pathname };

/**
 * Returns the project of the benchmarks of a package. They time its build, as Node.js loads it: Vitest's module runner
 * turns each import of a source file into a getter, which stops the compiler from inlining a call across modules and
 * makes a call of a small function several times as slow as it is for a user. A benchmark imports `../src/index.js`; the
 * project points it at the build, which `pnpm bench` makes first, and keeps Vitest from transforming the build. The
 * files run one after another, so that no benchmark competes with another for the processor.
 */
function benchmarkProject(name: string, test: { readonly timeout?: number }) {
  const build = new URL(`./packages/${name}/dist/index.js`, import.meta.url).pathname;
  return {
    resolve: { alias: [{ find: /^\.\.\/src\/index\.js$/, replacement: build }] },
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

/** The timeout of a benchmark that takes minutes: 30 minutes. */
const SLOW_BENCHMARK_TIMEOUT = 1_800_000;

// biome-ignore lint/style/noDefaultExport: Vitest reads its configuration from the default export.
export default defineConfig({
  test: {
    projects: [
      {
        test: { name: "core", include: ["packages/core/test/**/*.test.ts"] },
      },
      {
        resolve: { alias: CORE_SOURCE },
        // Calibrations to a tolerance of 1e-8 take seconds, the testlet, diagnosis and multidimensional models many
        // times more, on a machine under load longer still.
        test: { name: "psychometrics", include: ["packages/psychometrics/test/**/*.test.ts"], testTimeout: 120_000 },
      },
      {
        resolve: { alias: CORE_SOURCE },
        test: { name: "knowledge-tracing", include: ["packages/knowledge-tracing/test/**/*.test.ts"] },
      },
      // `pnpm bench` runs the benchmarks of each package with `vitest bench`, which gives a project the suffix "(bench)".
      benchmarkProject("core", {}),
      benchmarkProject("psychometrics", { timeout: SLOW_BENCHMARK_TIMEOUT }),
      benchmarkProject("knowledge-tracing", { timeout: SLOW_BENCHMARK_TIMEOUT }),
    ],
  },
});
