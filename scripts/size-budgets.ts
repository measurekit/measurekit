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
 * The budget of every entry point of the packages, bundled alone as a host's bundler makes it: minified, with
 * `export *` of the entry. The size is 1.25 times the size measured after gzip, rounded up to the next 500 bytes; the
 * modules are those it holds, which the import rules of AGENTS.md allow, so that an import that pulls another module
 * into an entry point fails until the list says so. docs/performance.md lists them. A budget changes only with the
 * reason in the commit that changes it.
 */
export const ENTRY_BUDGETS: Readonly<Record<string, EntryBudget>> = {
  "@measurekit/core": { gzip: 7000, holds: [] },
  "@measurekit/psychometrics": {
    gzip: 208_000,
    holds: [
      "core",
      "psychometrics/cat",
      "psychometrics/cdm",
      "psychometrics/change",
      "psychometrics/checks",
      "psychometrics/ctt",
      "psychometrics/data",
      "psychometrics/dif",
      "psychometrics/dimensionality",
      "psychometrics/equating",
      "psychometrics/explanatory",
      "psychometrics/fit",
      "psychometrics/generalizability",
      "psychometrics/irt",
      "psychometrics/math",
      "psychometrics/mokken",
      "psychometrics/multidimensional",
      "psychometrics/parameterization",
      "psychometrics/rater",
      "psychometrics/response-time",
      "psychometrics/scaling",
      "psychometrics/sim",
      "psychometrics/standard-setting",
    ],
  },
  "@measurekit/psychometrics/cat": {
    gzip: 28_500,
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
    gzip: 19_500,
    holds: ["core", "psychometrics/checks", "psychometrics/data", "psychometrics/irt", "psychometrics/math"],
  },
  "@measurekit/psychometrics/change": { gzip: 3000, holds: ["core", "psychometrics/checks"] },
  "@measurekit/psychometrics/ctt": { gzip: 10_500, holds: ["core", "psychometrics/data", "psychometrics/math"] },
  "@measurekit/psychometrics/data": { gzip: 3000, holds: ["core"] },
  "@measurekit/psychometrics/dif": {
    gzip: 36_500,
    holds: ["core", "psychometrics/data", "psychometrics/equating", "psychometrics/irt", "psychometrics/math"],
  },
  "@measurekit/psychometrics/dimensionality": {
    gzip: 18_000,
    holds: ["core", "psychometrics/data", "psychometrics/math"],
  },
  "@measurekit/psychometrics/equating": {
    gzip: 25_000,
    holds: ["core", "psychometrics/checks", "psychometrics/irt", "psychometrics/math"],
  },
  "@measurekit/psychometrics/explanatory": {
    gzip: 21_500,
    holds: ["core", "psychometrics/data", "psychometrics/irt", "psychometrics/math"],
  },
  "@measurekit/psychometrics/fit": {
    gzip: 24_500,
    holds: ["core", "psychometrics/checks", "psychometrics/data", "psychometrics/irt", "psychometrics/math"],
  },
  "@measurekit/psychometrics/generalizability": {
    gzip: 7000,
    holds: ["core", "psychometrics/data", "psychometrics/math"],
  },
  "@measurekit/psychometrics/irt": {
    gzip: 34_000,
    holds: ["core", "psychometrics/checks", "psychometrics/data", "psychometrics/math"],
  },
  "@measurekit/psychometrics/mokken": { gzip: 14_000, holds: ["core", "psychometrics/data", "psychometrics/math"] },
  "@measurekit/psychometrics/multidimensional": {
    gzip: 51_000,
    holds: ["core", "psychometrics/checks", "psychometrics/data", "psychometrics/irt", "psychometrics/math"],
  },
  "@measurekit/psychometrics/parameterization": {
    gzip: 7500,
    holds: ["core", "psychometrics/irt", "psychometrics/multidimensional"],
  },
  "@measurekit/psychometrics/rater": {
    gzip: 22_500,
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
    gzip: 23_000,
    holds: ["core", "psychometrics/checks", "psychometrics/data", "psychometrics/irt", "psychometrics/math"],
  },
  "@measurekit/psychometrics/scaling": {
    gzip: 14_500,
    holds: ["core", "psychometrics/checks", "psychometrics/equating", "psychometrics/irt", "psychometrics/math"],
  },
  "@measurekit/psychometrics/sim": {
    gzip: 9000,
    holds: [
      "core",
      "psychometrics/cdm",
      "psychometrics/checks",
      "psychometrics/data",
      "psychometrics/irt",
      "psychometrics/multidimensional",
    ],
  },
  "@measurekit/psychometrics/standard-setting": {
    gzip: 13_500,
    holds: ["core", "psychometrics/checks", "psychometrics/equating", "psychometrics/irt", "psychometrics/math"],
  },
  "@measurekit/knowledge-tracing": {
    gzip: 58_000,
    holds: [
      "core",
      "knowledge-tracing/bkt",
      "knowledge-tracing/data",
      "knowledge-tracing/elo",
      "knowledge-tracing/evaluate",
      "knowledge-tracing/irt",
      "knowledge-tracing/kst",
      "knowledge-tracing/logistic",
      "knowledge-tracing/math",
      "knowledge-tracing/model",
      "knowledge-tracing/practice",
      "knowledge-tracing/sim",
    ],
  },
  "@measurekit/knowledge-tracing/bkt": {
    gzip: 10_500,
    holds: ["core", "knowledge-tracing/data", "knowledge-tracing/math"],
  },
  "@measurekit/knowledge-tracing/data": { gzip: 3000, holds: ["core", "knowledge-tracing/math"] },
  "@measurekit/knowledge-tracing/elo": {
    gzip: 3000,
    holds: ["core", "knowledge-tracing/data", "knowledge-tracing/math"],
  },
  "@measurekit/knowledge-tracing/evaluate": {
    gzip: 8000,
    holds: ["core", "knowledge-tracing/data", "knowledge-tracing/math", "knowledge-tracing/model"],
  },
  "@measurekit/knowledge-tracing/irt": {
    gzip: 7000,
    holds: ["core", "knowledge-tracing/data", "knowledge-tracing/math"],
  },
  "@measurekit/knowledge-tracing/kst": {
    gzip: 19_000,
    holds: ["core", "knowledge-tracing/data", "knowledge-tracing/math"],
  },
  "@measurekit/knowledge-tracing/logistic": {
    gzip: 20_500,
    holds: ["core", "knowledge-tracing/data", "knowledge-tracing/math", "knowledge-tracing/model"],
  },
  "@measurekit/knowledge-tracing/practice": {
    gzip: 3000,
    holds: ["core", "knowledge-tracing/data", "knowledge-tracing/model"],
  },
  "@measurekit/knowledge-tracing/sim": {
    gzip: 2500,
    holds: ["core", "knowledge-tracing/data", "knowledge-tracing/model"],
  },
};
