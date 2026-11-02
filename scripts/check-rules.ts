import { spawnSync } from "node:child_process";
import {
  cpSync,
  existsSync,
  mkdtempSync,
  readdirSync,
  readFileSync,
  rmSync,
  symlinkSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { basename, dirname, join, resolve } from "node:path";

/**
 * A rule of the code that a test must guard. Replacing `find` with `replace` in `file` removes the rule and leaves valid
 * code, and the tests that `titles` selects in the file `test` must then fail: a test that still passes does not guard
 * the rule.
 */
export interface Rule {
  /** What the rule says, as a sentence. */
  readonly rule: string;
  /** The source file that holds the rule, relative to the repository root. */
  readonly file: string;
  /** The text of the file that implements the rule. It occurs once. */
  readonly find: string;
  /** The text that stands in its place without the rule, and does the rest. */
  readonly replace: string;
  /** The test file that guards the rule, relative to the repository root. */
  readonly test: string;
  /** A pattern of the titles of the tests that guard the rule. */
  readonly titles: string;
}

/** The rules that keep untrusted input from being misread or from costing what its size does not, each with its test. */
export const RULES: readonly Rule[] = [
  {
    rule: "A response above 2³¹ - 1 is refused, as the 32-bit arrays of the analyses would wrap it into another category.",
    file: "packages/psychometrics/src/data/response.ts",
    find: "value <= MAX_RESPONSE",
    replace: "true",
    test: "packages/psychometrics/test/data.test.ts",
    titles: "2³¹|invalid responses",
  },
  {
    rule: "A person with a missing response to a studied or matching item leaves a SIBTEST analysis, and -1 never counts as a score.",
    file: "packages/psychometrics/src/dif/sibtest.ts",
    find: "if (response < 0) return null;",
    replace: "",
    test: "packages/psychometrics/test/dif.test.ts",
    titles: "missing response to a matching|on verbal, sparse",
  },
  {
    rule: "A bundle tests the items of the matrix alone, and a bundle's item is not read as another.",
    file: "packages/psychometrics/src/dif/sibtest.ts",
    find: "if (!positions.has(item))",
    replace: "if (false)",
    test: "packages/psychometrics/test/dif.test.ts",
    titles: "rejects bundles",
  },
  {
    rule: "A bundle counts an item once, as a repeat would weigh it twice in the studied score.",
    file: "packages/psychometrics/src/dif/sibtest.ts",
    find: "if (seen.has(item))",
    replace: "if (false)",
    test: "packages/psychometrics/test/dif.test.ts",
    titles: "rejects bundles",
  },
  {
    rule: "SIBTEST tables the matching scores that occur, so one far response costs no table up to it.",
    file: "packages/psychometrics/src/dif/sibtest.ts",
    find: "const ranks = ranksOf(matchingScores);",
    replace:
      "const ranks = new Map(Array.from({ length: Math.max(...matchingScores) + 1 }, (_, score) => [score, score]));",
    test: "packages/psychometrics/test/dif.test.ts",
    titles: "tables only the matching scores",
  },
  {
    rule: "Every category below an item's highest response has a response.",
    file: "packages/psychometrics/src/irt/responses.ts",
    find: "if (counts.size <= highest) {",
    replace: "if (false) {",
    test: "packages/psychometrics/test/irt.test.ts",
    titles: "rejects an item without responses",
  },
  {
    rule: "An item is fixed by an entry of its own, and not by a property of every object.",
    file: "packages/psychometrics/src/irt/calibration-setup.ts",
    find: "Object.hasOwn(fixed, id) ? fixed[id] : undefined",
    replace: "fixed[id]",
    test: "packages/psychometrics/test/irt.test.ts",
    titles: "names of the properties",
  },
  {
    rule: "An item has a threshold by an entry of its own, and not by a property of every object.",
    file: "packages/psychometrics/src/response-time/rapid-guessing.ts",
    find: "(Object.hasOwn(thresholds, id) ? (thresholds[id] ?? Number.NaN) : Number.NaN)",
    replace: "(thresholds[id] ?? Number.NaN)",
    test: "packages/psychometrics/test/response-time.test.ts",
    titles: "counts a response at its threshold",
  },
  {
    rule: "A feature of an item is an entry of its own, and not a property of every object.",
    file: "packages/psychometrics/src/explanatory/predict-item-parameter.ts",
    find: "Object.hasOwn(features, effect.name) ? features[effect.name] : undefined",
    replace: "features[effect.name]",
    test: "packages/psychometrics/test/explanatory.test.ts",
    titles: "rejects a regression, features or values",
  },
  {
    rule: "The incomplete standard errors stop at their limit of success probabilities, before the calibration.",
    file: "packages/psychometrics/src/cdm/standard-errors.ts",
    find: "if (probabilities > MAX_INCOMPLETE_PROBABILITIES) {",
    replace: "if (false) {",
    test: "packages/psychometrics/test/cdm.test.ts",
    titles: "refuses incomplete standard errors",
  },
  {
    rule: "A component that leaves an item no variance ends the test, and gives no NaN.",
    file: "packages/psychometrics/src/dimensionality/minimum-average-partial.ts",
    find: "for (let i = 0; i < size; i++) if (!((covariance[i * size + i] ?? 0) > 0)) return null;",
    replace: "",
    test: "packages/psychometrics/test/dimensionality.test.ts",
    titles: "refuses items whose correlations are all 0",
  },
  {
    rule: "Kappa takes a table of the range only where the range is no wider than the pairs.",
    file: "packages/core/src/kappa.ts",
    find: "highest - lowest < n",
    replace: "true",
    test: "packages/core/test/kappa.test.ts",
    titles: "however far apart",
  },
  {
    rule: "A minimization never reports convergence at a point that is not finite.",
    file: "packages/core/src/bfgs.ts",
    find: "converged: isFinitePoint(point)",
    replace: "converged: true",
    test: "packages/core/test/optimize.test.ts",
    titles: "not finite",
  },
  {
    rule: "An id sets an entry of a record, and never its prototype.",
    file: "packages/knowledge-tracing/src/data/record.ts",
    find: 'if (key === "__proto__")',
    replace: "if (false)",
    test: "packages/knowledge-tracing/test/logistic.test.ts",
    titles: "logistic fits replayed",
  },
  {
    rule: "A state serves the learner it belongs to.",
    file: "packages/knowledge-tracing/src/data/learner.ts",
    find: "if (attempt.learner !== learner) {",
    replace: "if (false) {",
    test: "packages/knowledge-tracing/test/elo.test.ts",
    titles: "the state of another learner",
  },
];

/** The folders that a scratch copy leaves out: generated output and installed packages, which it links instead. */
const SKIPPED = new Set(["node_modules", ".git", "dist", "coverage", "api"]);
/** The longest a test of a copy may take, in milliseconds. */
const TEST_TIMEOUT_MS = 120_000;
/** The longest a run of a copy may take, in milliseconds: a rule that bounds work leaves a run that does not end. */
const RUN_TIMEOUT_MS = 150_000;

/** What checking one rule found. */
export interface Verdict {
  readonly guarded: boolean;
  /** A sentence that says what happened and, when the rule is not guarded, what to do. */
  readonly message: string;
}

/**
 * Removes a rule from the text of its file.
 *
 * @throws `Error` when the text to replace does not occur exactly once.
 */
export function removeRule(source: string, rule: Rule): string {
  const count = source.split(rule.find).length - 1;
  if (count !== 1) {
    throw new Error(`${rule.file} holds ${JSON.stringify(rule.find)} ${count} times; the rule needs it exactly once.`);
  }
  return source.replace(rule.find, () => rule.replace);
}

/** Returns the number of failed tests in the report of a Vitest run, or `undefined` when the run did not run tests. */
export function failedTests(report: string): number | undefined {
  const summary = /^\s*Tests\s+(.*)$/m.exec(report)?.[1];
  if (summary === undefined || !/\d+ (?:failed|passed)/.test(summary)) return undefined;
  return Number(/(\d+) failed/.exec(summary)?.[1] ?? 0);
}

/** The folders directly below a folder of the repository, if it exists. */
function foldersIn(root: string, folder: string): string[] {
  const parent = join(root, folder);
  if (!existsSync(parent)) return [];
  return readdirSync(parent, { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .map((entry) => join(parent, entry.name));
}

/** Copies the repository into a scratch directory without generated output, and links the installed packages. */
function copyRepository(root: string, scratch: string): void {
  cpSync(root, scratch, { recursive: true, filter: (path) => !SKIPPED.has(basename(path)) });
  for (const directory of [root, ...foldersIn(root, "packages"), join(root, "examples")]) {
    const modules = join(directory, "node_modules");
    if (existsSync(modules)) symlinkSync(modules, join(scratch, directory.slice(root.length), "node_modules"));
  }
}

/** Runs the tests of a rule in a copy and returns what they reported; a run that does not end counts as a failure. */
function runTests(root: string, scratch: string, rule: Rule): { failed: number | undefined; report: string } {
  const vitest = join(root, "node_modules/vitest/vitest.mjs");
  // Threads, so that a run that is stopped leaves no worker behind.
  const options = ["--pool=threads", "--maxWorkers=2", `--testTimeout=${TEST_TIMEOUT_MS}`, "-t", rule.titles];
  const run = spawnSync(process.execPath, [vitest, "run", ...options, rule.test], {
    cwd: scratch,
    encoding: "utf8",
    // biome-ignore lint/style/useNamingConvention: these are the names of the variables.
    env: { ...process.env, NO_COLOR: "1", FORCE_COLOR: "0" },
    killSignal: "SIGKILL",
    timeout: RUN_TIMEOUT_MS,
  });
  const report = `${run.stdout}${run.stderr}`;
  return { failed: run.error === undefined ? failedTests(report) : 1, report };
}

/**
 * Checks that the tests of a rule pass on the code, and fail on a copy of the code without the rule.
 *
 * @param root - The repository.
 * @param rule - The rule.
 */
export function checkRule(root: string, rule: Rule): Verdict {
  const scratch = mkdtempSync(join(tmpdir(), "measurekit-rule-"));
  try {
    copyRepository(root, scratch);
    const baseline = runTests(root, scratch, rule);
    if (baseline.failed !== 0) {
      return {
        guarded: false,
        message: `${rule.test} does not pass on the unchanged code; fix it first.\n${baseline.report}`,
      };
    }
    const path = join(scratch, rule.file);
    writeFileSync(path, removeRule(readFileSync(path, "utf8"), rule));
    const mutated = runTests(root, scratch, rule);
    if (mutated.failed === undefined) {
      return {
        guarded: false,
        message: `${rule.test} did not run on the copy without the rule; the replacement may be invalid code.\n${mutated.report}`,
      };
    }
    if (mutated.failed === 0) {
      return {
        guarded: false,
        message: `${rule.test} passes without the rule: add a test that fails when "${rule.rule}" is broken.`,
      };
    }
    return {
      guarded: true,
      message: `${mutated.failed} ${mutated.failed === 1 ? "test fails" : "tests fail"} without the rule.`,
    };
  } finally {
    rmSync(scratch, { recursive: true, force: true });
  }
}

/** Checks every rule, or the rules of the given files, and returns the exit code. */
function main(paths: readonly string[]): number {
  const root = resolve(dirname(import.meta.dirname));
  const rules = RULES.filter(({ file }) => paths.length === 0 || paths.some((path) => file.startsWith(path)));
  let unguarded = 0;
  for (const rule of rules) {
    const { guarded, message } = checkRule(root, rule);
    if (!guarded) unguarded += 1;
    console.log(`${guarded ? "guarded  " : "UNGUARDED"} ${rule.rule} ${message}`);
  }
  console.log(
    unguarded === 0 ? `Every one of ${rules.length} rules is guarded.` : `${unguarded} rules are not guarded.`,
  );
  return unguarded === 0 ? 0 : 1;
}

if (import.meta.main) process.exitCode = main(process.argv.slice(2));
