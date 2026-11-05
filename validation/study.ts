// What the studies of validation/ share: each writes its results to results/<study>.json beside it, fit by fit, and
// records the seconds of each fit. A study runs on the build: `pnpm build`, then `node validation/<package>/<study>.ts`.

import { mkdirSync, writeFileSync } from "node:fs";
import { basename, dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

/**
 * Writes the results of the study at `studyUrl`, its `import.meta.url`, to `results/<study>.json` beside it, so that a run
 * cut short keeps the fits it finished.
 */
export function writeResults(studyUrl: string, results: object): void {
  const study = fileURLToPath(studyUrl);
  const folder = join(dirname(study), "results");
  mkdirSync(folder, { recursive: true });
  writeFileSync(join(folder, `${basename(study, ".ts")}.json`), `${JSON.stringify(results, null, 2)}\n`);
}

/** Runs `compute` and returns its value with the seconds it took, to a tenth. */
export function timed<T>(compute: () => T): { readonly value: T; readonly seconds: number } {
  const start = performance.now();
  const value = compute();
  return { value, seconds: Math.round((performance.now() - start) / 100) / 10 };
}
