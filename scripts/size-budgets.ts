/** The budget of an entry point bundled alone: the most it may take after gzip, and the modules other than its own it holds. */
export interface EntryBudget {
  /** The most that the entry point may take after gzip, in bytes. */
  readonly gzip: number;
  /**
   * The modules other than its own that have code in the bundle, sorted: `core`, and `<package>/<folder>` or
   * `<package>/<file>` of a library.
   */
  readonly holds: readonly string[];
}

/**
 * The budget of every entry point of the packages, bundled alone as a host's bundler makes it: minified, with `export *` of
 * the entry. The size is 1.25 times the size measured after gzip, rounded up to the next 500 bytes; the modules are those it
 * holds, which the import rules of AGENTS.md allow, so that an import that pulls another module into an entry point fails
 * until the list says so. docs/<package>/performance.md lists them. Every budget is set again by this rule from main before
 * a release, and one budget, in a commit that says why, when its entry passes it; gzip is measured on Node.js 22.
 */
export const ENTRY_BUDGETS: Readonly<Record<string, EntryBudget>> = {
  "@measurekit/core": { gzip: 9000, holds: [] },
  "@measurekit/psychometrics": { gzip: 2000, holds: ["core"] },
  "@measurekit/psychometrics/cat": {
    gzip: 31_500,
    holds: [
      "core",
      "psychometrics/cdm",
      "psychometrics/checks",
      "psychometrics/data",
      "psychometrics/irt",
      "psychometrics/math",
      "psychometrics/multidimensional",
      "psychometrics/sim",
    ],
  },
  "@measurekit/psychometrics/cdm": {
    gzip: 21_500,
    holds: ["core", "psychometrics/checks", "psychometrics/data", "psychometrics/irt", "psychometrics/math"],
  },
  "@measurekit/psychometrics/change": { gzip: 3500, holds: ["core", "psychometrics/checks"] },
  "@measurekit/psychometrics/ctt": { gzip: 12_000, holds: ["core", "psychometrics/data", "psychometrics/math"] },
  "@measurekit/psychometrics/data": { gzip: 3500, holds: ["core"] },
  "@measurekit/psychometrics/dif": {
    gzip: 43_500,
    holds: ["core", "psychometrics/data", "psychometrics/equating", "psychometrics/irt", "psychometrics/math"],
  },
  "@measurekit/psychometrics/dimensionality": {
    gzip: 19_500,
    holds: ["core", "psychometrics/data", "psychometrics/math"],
  },
  "@measurekit/psychometrics/equating": {
    gzip: 27_500,
    holds: ["core", "psychometrics/checks", "psychometrics/irt", "psychometrics/math"],
  },
  "@measurekit/psychometrics/explanatory": {
    gzip: 23_000,
    holds: ["core", "psychometrics/data", "psychometrics/irt", "psychometrics/math"],
  },
  "@measurekit/psychometrics/fit": {
    gzip: 52_000,
    holds: [
      "core",
      "psychometrics/checks",
      "psychometrics/data",
      "psychometrics/irt",
      "psychometrics/math",
      "psychometrics/sim",
    ],
  },
  "@measurekit/psychometrics/generalizability": {
    gzip: 7500,
    holds: ["core", "psychometrics/data", "psychometrics/math"],
  },
  "@measurekit/psychometrics/irt": {
    gzip: 41_000,
    holds: ["core", "psychometrics/checks", "psychometrics/data", "psychometrics/math"],
  },
  "@measurekit/psychometrics/mokken": {
    gzip: 16_000,
    holds: ["core", "psychometrics/checks", "psychometrics/data", "psychometrics/math"],
  },
  "@measurekit/psychometrics/multidimensional": {
    gzip: 58_500,
    holds: ["core", "psychometrics/checks", "psychometrics/data", "psychometrics/irt", "psychometrics/math"],
  },
  "@measurekit/psychometrics/parameterization": {
    gzip: 8000,
    holds: ["core", "psychometrics/irt", "psychometrics/multidimensional"],
  },
  "@measurekit/psychometrics/rater": {
    gzip: 29_500,
    holds: [
      "core",
      "psychometrics/checks",
      "psychometrics/data",
      "psychometrics/fit",
      "psychometrics/irt",
      "psychometrics/math",
    ],
  },
  "@measurekit/psychometrics/response-time": {
    gzip: 26_000,
    holds: ["core", "psychometrics/checks", "psychometrics/data", "psychometrics/irt", "psychometrics/math"],
  },
  "@measurekit/psychometrics/scaling": {
    gzip: 17_500,
    holds: ["core", "psychometrics/checks", "psychometrics/equating", "psychometrics/irt", "psychometrics/math"],
  },
  "@measurekit/psychometrics/sim": {
    gzip: 9500,
    holds: [
      "core",
      "psychometrics/cdm",
      "psychometrics/checks",
      "psychometrics/data",
      "psychometrics/irt",
      "psychometrics/multidimensional",
    ],
  },
  "@measurekit/psychometrics/similarity": {
    gzip: 11_500,
    holds: ["core", "psychometrics/checks", "psychometrics/data", "psychometrics/irt", "psychometrics/math"],
  },
  "@measurekit/psychometrics/standard-setting": {
    gzip: 16_500,
    holds: ["core", "psychometrics/checks", "psychometrics/equating", "psychometrics/irt", "psychometrics/math"],
  },
  "@measurekit/knowledge-tracing": { gzip: 2000, holds: ["core"] },
  "@measurekit/knowledge-tracing/bkt": {
    gzip: 12_000,
    holds: ["core", "knowledge-tracing/data", "knowledge-tracing/math"],
  },
  "@measurekit/knowledge-tracing/data": { gzip: 3500, holds: ["core"] },
  "@measurekit/knowledge-tracing/elo": {
    gzip: 3500,
    holds: ["core", "knowledge-tracing/data", "knowledge-tracing/math"],
  },
  "@measurekit/knowledge-tracing/evaluate": {
    gzip: 8500,
    holds: ["core", "knowledge-tracing/data", "knowledge-tracing/math", "knowledge-tracing/model"],
  },
  "@measurekit/knowledge-tracing/irt": {
    gzip: 7500,
    holds: ["core", "knowledge-tracing/data", "knowledge-tracing/math"],
  },
  "@measurekit/knowledge-tracing/kst": {
    gzip: 22_500,
    holds: ["core", "knowledge-tracing/data", "knowledge-tracing/math"],
  },
  "@measurekit/knowledge-tracing/logistic": {
    gzip: 21_000,
    holds: ["core", "knowledge-tracing/data", "knowledge-tracing/math", "knowledge-tracing/model"],
  },
  "@measurekit/knowledge-tracing/practice": {
    gzip: 3500,
    holds: ["core", "knowledge-tracing/data", "knowledge-tracing/model"],
  },
  "@measurekit/knowledge-tracing/sim": {
    gzip: 2500,
    holds: ["core", "knowledge-tracing/data", "knowledge-tracing/model"],
  },
};
