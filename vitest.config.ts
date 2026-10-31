import { defineConfig } from "vitest/config";

/** Resolves core to its source, so that the tests of the packages that depend on it run the code and not a build. */
const CORE_SOURCE = { "@measurekit/core": new URL("./packages/core/src/index.ts", import.meta.url).pathname };

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
      {
        // The benchmarks of each package: `pnpm bench` runs them with `vitest bench`, which gives a project the suffix
        // "(bench)". The files run one after another, so that no benchmark competes with another for the processor, and a
        // test may take minutes.
        test: {
          name: "core-bench",
          include: [],
          fileParallelism: false,
          benchmark: { include: ["packages/core/test/**/*.bench.ts"], suppressExportGetterWarnings: true },
        },
      },
      {
        resolve: { alias: CORE_SOURCE },
        test: {
          name: "psychometrics-bench",
          include: [],
          testTimeout: 1_800_000,
          hookTimeout: 1_800_000,
          fileParallelism: false,
          benchmark: { include: ["packages/psychometrics/test/**/*.bench.ts"], suppressExportGetterWarnings: true },
        },
      },
      {
        resolve: { alias: CORE_SOURCE },
        test: {
          name: "knowledge-tracing-bench",
          include: [],
          testTimeout: 1_800_000,
          hookTimeout: 1_800_000,
          fileParallelism: false,
          benchmark: { include: ["packages/knowledge-tracing/test/**/*.bench.ts"], suppressExportGetterWarnings: true },
        },
      },
    ],
  },
});
