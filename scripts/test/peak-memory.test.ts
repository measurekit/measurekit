import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { bundleLibraries, peakRise, READERS } from "../peak-memory.js";

/** A case runs two child processes, each of which builds and parses its input: a few seconds, longer on a loaded machine. */
const CASE_TIMEOUT = 120_000;

describe("the peak memory of every reader of untrusted input", () => {
  let directory = "";

  beforeAll(async () => {
    directory = mkdtempSync(join(tmpdir(), "measurekit-peak-"));
    await bundleLibraries(directory);
  }, CASE_TIMEOUT);

  afterAll(() => {
    rmSync(directory, { recursive: true, force: true });
  });

  it.each(READERS.map(({ name, budget }) => [name, budget] as const))(
    "keeps %s within its budget of %i MB",
    (name, budget) => {
      const { rise } = peakRise(directory, name);
      expect(rise, `The megabytes that ${name} adds to the peak`).toBeLessThan(budget);
    },
    CASE_TIMEOUT,
  );
});
