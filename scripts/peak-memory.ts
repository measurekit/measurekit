// The peak memory that each reader of untrusted input adds to a Node.js process, on its worst inputs within the limits of
// SECURITY.md: the peak resident memory of a process that parses the input from JSON and reads it, less that of the
// same process that parses it and does not. `test/peak-memory.test.ts` checks each against its budget; `node
// scripts/peak-memory.ts` prints them, the median of three pairs, as the table of docs/performance.md.

import { spawnSync } from "node:child_process";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { build } from "esbuild";

const PACKAGES = resolve(import.meta.dirname, "../packages");
const LIBRARIES = ["psychometrics", "knowledge-tracing"] as const;
const MEGABYTE = 2 ** 20;
/** The heap a reader may take, above V8's default, so that a large input fails on its budget, not on the heap. */
const HEAP_MEGABYTES = 8192;
/** The first argument of the script in a child process. */
const CHILD = "--child";

/** The libraries as the child process loads them, bundled from their sources. */
// biome-ignore lint/suspicious/noExplicitAny: the bundles are loaded at run time, and each case checks what it calls.
type Libraries = Readonly<Record<(typeof LIBRARIES)[number], any>>;

/**
 * A reader of untrusted input on one input: the input, built in the child process, what both processes make of it
 * before the reading, such as a response matrix, and the reading.
 */
interface ReaderCase {
  /** The reader and its input, as the table of docs/performance.md names them. */
  readonly name: string;
  /**
   * The most that the reading may add to the peak, in megabytes: twice the largest rise measured on Node.js 22 and 24, on
   * macOS and on Linux, rounded up to 10 MB.
   */
  readonly budget: number;
  readonly input: () => unknown;
  readonly prepare?: (input: never, libraries: Libraries) => unknown;
  readonly read: (prepared: never, libraries: Libraries) => unknown;
}

const ids = (prefix: string, count: number): string[] => Array.from({ length: count }, (_, i) => `${prefix}${i + 1}`);

/** The cases, the worst input of each reader within its limits. */
export const READERS: readonly ReaderCase[] = [
  {
    name: "responseMatrix: 1 person, 1,000,000 items, with the ids it gives them",
    budget: 150,
    input: () => [Array.from({ length: 1_000_000 }, (_, i) => i % 2)],
    read: (rows: number[][], { psychometrics }) => psychometrics.responseMatrix(rows),
  },
  {
    name: "responseRecords: 1,000,000 records, each of its own person and item",
    budget: 1060,
    input: () => Array.from({ length: 1_000_000 }, (_, k) => ({ person: `p${k}`, item: `i${k}`, response: k % 2 })),
    read: (records: unknown[], { psychometrics }) => psychometrics.responseRecords(records),
  },
  {
    name: "eventLog: 1,000,000 events, each of its own learner, item and skill",
    budget: 230,
    input: () =>
      Array.from({ length: 1_000_000 }, (_, k) => ({
        learner: `l${k}`,
        item: `i${k}`,
        skills: [`s${k}`],
        time: k,
        outcome: k % 2,
      })),
    read: (events: unknown[], { "knowledge-tracing": tracing }) => tracing.eventLog(events),
  },
  {
    name: "iita: 1,000,000 events, each of its own learner and item, which it refuses, from the events",
    budget: 740,
    input: () =>
      Array.from({ length: 1_000_000 }, (_, k) => ({
        learner: `l${k}`,
        item: `i${k}`,
        skills: [],
        time: k,
        outcome: k % 2,
      })),
    read: (events: unknown[], { "knowledge-tracing": tracing }) => {
      try {
        return tracing.iita(tracing.eventLog(events));
      } catch (error) {
        return error;
      }
    },
  },
  {
    name: "qMatrixIdentifiability: 100,000 items of 20 attributes",
    budget: 320,
    input: () => Object.fromEntries(ids("item", 100_000).map((id) => [id, ids("a", 20)])),
    read: (qMatrix: unknown, { psychometrics }) => psychometrics.qMatrixIdentifiability(qMatrix),
  },
  {
    name: "calibrateCdm: 64 items of one of 20 attributes, the most items × 2^K, two persons",
    budget: 4120,
    input: () => ({
      itemIds: ids("item", 64),
      rows: [Array(64).fill(0), Array(64).fill(1)],
      qMatrix: Object.fromEntries(ids("item", 64).map((id, i) => [id, [`a${(i % 20) + 1}`]])),
    }),
    prepare: (
      { itemIds, rows, qMatrix }: { itemIds: string[]; rows: number[][]; qMatrix: unknown },
      { psychometrics },
    ) => ({
      matrix: psychometrics.responseMatrix(rows, { itemIds }),
      qMatrix,
    }),
    read: ({ matrix, qMatrix }: { matrix: unknown; qMatrix: unknown }, { psychometrics }) =>
      psychometrics.calibrateCdm(matrix, qMatrix, { maxIterations: 2 }),
  },
  {
    name: "bkt: the stored parameters of 100,000 skills",
    budget: 250,
    input: () => ({
      skills: Object.fromEntries(
        ids("s", 100_000).map((skill) => [skill, { prior: 0.3, learn: 0.1, forget: 0, guess: 0.2, slip: 0.1 }]),
      ),
    }),
    read: (parameters: unknown, { "knowledge-tracing": tracing }) => tracing.bkt({ parameters }),
  },
  {
    name: "personScalability: 2 persons, 1,000,000 items",
    budget: 400,
    input: () => {
      const row = Array.from({ length: 1_000_000 }, (_, i) => i % 2);
      return [row, row.map((response) => 1 - response)];
    },
    prepare: (rows: number[][], { psychometrics }) => psychometrics.responseMatrix(rows),
    read: (matrix: unknown, { psychometrics }) => psychometrics.personScalability(matrix),
  },
  {
    name: "localDependenceChiSquare: 2 persons, 2,000 2PL items, 1,999,000 pairs",
    budget: 1110,
    input: () => {
      const row = Array.from({ length: 2000 }, (_, i) => i % 2);
      return {
        rows: [row, row.map((response) => 1 - response)],
        items: Array.from({ length: 2000 }, (_, i) => ({ a: 1, b: (i % 7) / 3 - 1 })),
      };
    },
    prepare: ({ rows, items }: { rows: number[][]; items: { a: number; b: number }[] }, { psychometrics }) => ({
      matrix: psychometrics.responseMatrix(rows),
      items: items.map((parameters) => ({
        model: psychometrics.twoParameterLogistic,
        parameters: { ...parameters, c: 0, d: 1 },
      })),
    }),
    read: ({ matrix, items }: { matrix: unknown; items: unknown }, { psychometrics }) =>
      psychometrics.localDependenceChiSquare(matrix, items),
  },
  {
    name: "diagnosticAdaptiveTest: 64 items of one of 20 attributes, the most items × 2^K",
    budget: 1940,
    input: () => ({
      attributes: ids("a", 20),
      pool: Array.from({ length: 64 }, (_, j) => ({
        id: `q${j + 1}`,
        attributes: [`a${(j % 20) + 1}`],
        probabilities: [0.2, 0.8],
      })),
    }),
    read: (options: { attributes: string[]; pool: unknown[] }, { psychometrics }) =>
      psychometrics.diagnosticAdaptiveTest({ ...options, stopping: psychometrics.testLength(10) }),
  },
];

/** Bundles each library from its sources, with core from its sources, into `directory`. */
export async function bundleLibraries(directory: string): Promise<void> {
  await build({
    entryPoints: Object.fromEntries(LIBRARIES.map((name) => [name, resolve(PACKAGES, name, "src/index.ts")])),
    bundle: true,
    format: "esm",
    platform: "node",
    outdir: directory,
    logLevel: "silent",
    alias: { "@measurekit/core": resolve(PACKAGES, "core/src/index.ts") },
  });
}

/** The peak of one child process: the bytes of the input's JSON and the peak resident memory, both in bytes. */
interface ChildPeak {
  readonly bytes: number;
  readonly peak: number;
}

/** Runs a case in a child process, reading the input or not, and returns its peak. */
function childPeak(directory: string, name: string, read: boolean): ChildPeak {
  const run = spawnSync(
    process.execPath,
    [
      `--max-old-space-size=${HEAP_MEGABYTES}`,
      "--expose-gc",
      import.meta.filename,
      CHILD,
      directory,
      name,
      String(read),
    ],
    { encoding: "utf8" },
  );
  if (run.status !== 0) throw new Error(`The case "${name}" failed: ${run.stderr}`);
  return JSON.parse(run.stdout);
}

/** The rise of the peak that reading an input adds, in megabytes, and the megabytes of its JSON. */
export interface PeakRise {
  readonly json: number;
  readonly rise: number;
}

/**
 * Returns the median rise of the peak that a case's reading adds over `pairs` pairs of child processes, one that does
 * not read and one that does, run one after the other.
 *
 * @param directory - The directory of the bundles of {@link bundleLibraries}.
 */
export function peakRise(directory: string, name: string, pairs = 1): PeakRise {
  const rises: number[] = [];
  let bytes = 0;
  for (let pair = 0; pair < pairs; pair++) {
    const without = childPeak(directory, name, false);
    const reading = childPeak(directory, name, true);
    bytes = reading.bytes;
    rises.push(reading.peak - without.peak);
  }
  rises.sort((a, b) => a - b);
  return { json: bytes / MEGABYTE, rise: (rises[Math.floor(rises.length / 2)] ?? 0) / MEGABYTE };
}

/** Runs one case in this process, as the child: builds its input, parses it from JSON and reads it or not. */
async function runChild([directory = "", name = "", read = ""]: readonly string[]): Promise<void> {
  const reader = READERS.find((candidate) => candidate.name === name);
  if (reader === undefined) throw new Error(`No case is named "${name}".`);
  const libraries = Object.fromEntries(
    await Promise.all(
      LIBRARIES.map(async (library) => [library, await import(pathToFileURL(join(directory, `${library}.js`)).href)]),
    ),
  );
  let text: string | null = JSON.stringify(reader.input());
  const bytes = text.length;
  const input = JSON.parse(text);
  text = null;
  const gc = globalThis.gc ?? (() => undefined);
  const prepared = reader.prepare === undefined ? input : reader.prepare(input as never, libraries);
  gc();
  gc();
  // The result stays referenced until the peak is read, as a caller holds it.
  const result = read === "true" ? reader.read(prepared as never, libraries) : null;
  console.log(JSON.stringify({ bytes, peak: process.resourceUsage().maxRSS * 1024, held: result !== null }));
}

if (import.meta.main) {
  const [first, ...rest] = process.argv.slice(2);
  if (first === CHILD) {
    await runChild(rest);
  } else {
    // `node scripts/peak-memory.ts calibrateCdm bkt` measures the cases whose names begin with the words given.
    const chosen = READERS.filter(
      ({ name }) => first === undefined || [first, ...rest].some((word) => name.startsWith(word)),
    );
    const directory = mkdtempSync(join(tmpdir(), "measurekit-peak-"));
    try {
      await bundleLibraries(directory);
      console.log("| Reader | JSON (MB) | Peak added (MB) | Budget (MB) |");
      console.log("| ------ | --------: | --------------: | ----------: |");
      for (const { name, budget } of chosen) {
        const { json, rise } = peakRise(directory, name, 3);
        console.log(`| ${name} | ${json.toFixed(1)} | ${Math.round(rise)} | ${budget} |`);
      }
    } finally {
      rmSync(directory, { recursive: true, force: true });
    }
  }
}
