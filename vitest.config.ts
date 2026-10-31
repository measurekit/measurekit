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
    ],
  },
});
