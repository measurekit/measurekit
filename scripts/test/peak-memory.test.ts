import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { bundleLibraries, peakRise, READERS } from "../peak-memory.ts";

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

  it(
    "fails a reading that keeps more alive at once than its heap holds",
    () => {
      const [first] = READERS;
      expect(() => peakRise(directory, first?.name ?? "", 16)).toThrow("did not end within a heap of 16 MiB");
    },
    CASE_TIMEOUT,
  );

  // Each process has the case's heap, so that a reading that keeps more alive at once fails, whenever the collector runs and
  // however busy the machine is; the budget bounds the rise that the heap does not hold, the buffers of typed arrays.
  it.each(READERS.map(({ name, budget, heap }) => [name, budget, heap] as const))(
    "keeps %s within its budget of %i MB, in a heap of %i MiB",
    (name, budget, heap) => {
      const { rise } = peakRise(directory, name, heap);
      expect(rise, `The megabytes that ${name} adds to the peak`).toBeLessThan(budget);
    },
    CASE_TIMEOUT,
  );
});
