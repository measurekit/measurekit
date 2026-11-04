import { readdirSync, readFileSync, rmSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { afterAll, afterEach, describe, expect, it, vi } from "vitest";
import { chainsOf, resultsShown, runSample } from "../../scripts/guide-samples.js";

/** The root of the repository. */
const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..", "..");

/** The folder that holds the samples while they run, which git ignores. */
const SAMPLES = join(ROOT, "examples", ".samples");

/** The documents with samples: the guides of each library and the READMEs. */
const DOCUMENTS = [
  ...["psychometrics", "knowledge-tracing"].flatMap((name) =>
    readdirSync(join(ROOT, "docs", name))
      .filter((file) => file.endsWith(".md"))
      .map((file) => `docs/${name}/${file}`),
  ),
  "README.md",
  "packages/core/README.md",
  "packages/psychometrics/README.md",
  "packages/knowledge-tracing/README.md",
];

const chains = DOCUMENTS.flatMap((document) => chainsOf(ROOT, document));

describe("the samples of the guides and READMEs", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  afterAll(() => {
    rmSync(SAMPLES, { recursive: true, force: true });
  });

  it("shows each result after a statement that gives it, with its mark", () => {
    expect(chains.flatMap(({ problems }) => problems)).toEqual([]);
  });

  // A result that no statement picked up, such as one in a sample that does not run, would pass unseen.
  it.each(DOCUMENTS)("checks every result that %s shows", (document) => {
    const checked = chains
      .filter((chain) => chain.document === document)
      .reduce((sum, { results }) => sum + results, 0);
    expect(checked).toBe(resultsShown(readFileSync(join(ROOT, document), "utf8")));
  });

  it.each(chains)(
    "runs the sample of $document from line $line and gets the $results results it shows",
    async (chain) => {
      vi.spyOn(console, "log").mockImplementation(() => {});
      await runSample(SAMPLES, `${chain.document.replaceAll(/[/.]/g, "-")}-${chain.line}`, chain.source);
    },
  );
});
