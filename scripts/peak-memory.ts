// The peak memory that each reader of untrusted input adds to a Node.js process, on its worst inputs within the limits of
// SECURITY.md: the peak resident memory of a process that parses the input from JSON and reads it, less that of the
// same process that parses it and does not. Each process has a heap bounded to the case's `heap`, so that a reading that
// keeps more alive at once fails, whenever the collector runs and however busy the machine is; the budget of the rise bounds
// what the heap does not hold, the buffers of typed arrays. `test/peak-memory.test.ts` checks each case; `node
// scripts/peak-memory.ts` prints the rises, the median of three pairs, as the tables of docs/<package>/performance.md, and `node
// scripts/peak-memory.ts --heap` the least heap in which each reading ends, from which `heap` is chosen.

import { spawnSync } from "node:child_process";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { build } from "esbuild";

const PACKAGES = resolve(import.meta.dirname, "../packages");
const LIBRARIES = ["psychometrics", "knowledge-tracing"] as const;
const MEGABYTE = 2 ** 20;
/**
 * The heap of the processes whose rise the tables of docs/<package>/performance.md give, in MiB: above V8's default, as the rise that a
 * host sees depends on when the collector runs, which a bounded heap brings forward.
 */
const TABLE_HEAP = 8192;
/** The first argument of the script in a child process. */
const CHILD = "--child";
/** The argument that prints the least heap of each case. */
const LEAST_HEAP = "--heap";
/** The step of the search for the least heap, in MiB. */
const HEAP_STEP = 8;
/** The most heap that the search tries, in MiB; a reading that needs more is reported as above it. */
const HEAP_SEARCH_LIMIT = 4096;

/** The libraries as the child process loads them, bundled from their sources. */
// biome-ignore lint/suspicious/noExplicitAny: the bundles are loaded at run time, and each case checks what it calls.
type Libraries = Readonly<Record<(typeof LIBRARIES)[number], any>>;

/**
 * A reader of untrusted input on one input: the input, built in the child process, what both processes make of it
 * before the reading, such as a response matrix, and the reading.
 */
interface ReaderCase {
  /** The reader and its input, as the tables of docs/<package>/performance.md name them. */
  readonly name: string;
  /**
   * The most that the reading may add to the peak, in megabytes: twice the largest rise measured on Node.js 22 and 24, on
   * macOS and on Linux, rounded up to 10 MB.
   */
  readonly budget: number;
  /**
   * The heap of both processes, in MiB, with the input and the code: twice the least heap in which the reading ended on
   * Node.js 22, 24 and 26, rounded up to 16 MiB.
   */
  readonly heap: number;
  readonly input: () => unknown;
  readonly prepare?: (input: never, libraries: Libraries) => unknown;
  readonly read: (prepared: never, libraries: Libraries) => unknown;
}

const ids = (prefix: string, count: number): string[] => Array.from({ length: count }, (_, i) => `${prefix}${i + 1}`);

/** Two persons who answer the items alike or opposite, and the parameters of 2PL items. */
interface PairItems {
  readonly rows: number[][];
  readonly items: { readonly a: number; readonly b: number }[];
}

/** The input of a statistic of every pair of items: two persons and `count` 2PL items. */
const pairItems = (count: number): PairItems => {
  const row = Array.from({ length: count }, (_, i) => i % 2);
  return {
    rows: [row, row.map((response) => 1 - response)],
    items: Array.from({ length: count }, (_, i) => ({ a: 1, b: (i % 7) / 3 - 1 })),
  };
};

/** Reads the input of {@link pairItems} into a response matrix and 2PL items. */
const pairInput = ({ rows, items }: PairItems, psychometrics: Libraries["psychometrics"]) => ({
  matrix: psychometrics.buildResponseMatrix(rows),
  items: items.map((parameters) => ({
    model: psychometrics.twoParameterLogistic,
    parameters: { ...parameters, c: 0, d: 1 },
  })),
});

/** The cases, the worst input of each reader within its limits. */
export const READERS: readonly ReaderCase[] = [
  {
    name: "buildResponseMatrix: 1 person, 1,000,000 items, with the ids it gives them",
    budget: 150,
    heap: 144,
    input: () => [Array.from({ length: 1_000_000 }, (_, i) => i % 2)],
    read: (rows: number[][], { psychometrics }) => psychometrics.buildResponseMatrix(rows),
  },
  {
    name: "buildResponseRecords: 1,000,000 records, each of its own person and item",
    budget: 1070,
    heap: 1056,
    input: () => Array.from({ length: 1_000_000 }, (_, k) => ({ person: `p${k}`, item: `i${k}`, response: k % 2 })),
    read: (records: unknown[], { psychometrics }) => psychometrics.buildResponseRecords(records),
  },
  {
    name: "buildEventLog: 1,000,000 events, each of its own learner, item and skill",
    budget: 230,
    heap: 1136,
    input: () =>
      Array.from({ length: 1_000_000 }, (_, k) => ({
        learner: `l${k}`,
        item: `i${k}`,
        skills: [`s${k}`],
        time: k,
        outcome: k % 2,
      })),
    read: (events: unknown[], { "knowledge-tracing": tracing }) => tracing.buildEventLog(events),
  },
  {
    name: "iita: 1,000,000 events, each of its own learner and item, which it refuses, from the events",
    budget: 740,
    heap: 1344,
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
        return tracing.iita(tracing.buildEventLog(events));
      } catch (error) {
        return error;
      }
    },
  },
  {
    name: "iita: 1,000 items, 276 learners, 24 candidates of 23.9 million cells, from the events",
    budget: 620,
    heap: 608,
    // Learners who fail one item each, 1 to 23 of them per item: 24 candidates of a million prerequisites, the most
    // that the limit of the candidates' cells lets through.
    input: () =>
      Array.from({ length: 23 }, (_, k) => Array.from({ length: k + 1 }, () => k))
        .flat()
        .flatMap((failed, learner) =>
          Array.from({ length: 1000 }, (_, k) => ({
            learner: `l${learner}`,
            item: `i${k}`,
            skills: [],
            time: learner * 1000 + k,
            outcome: k === failed ? 0 : 1,
          })),
        ),
    read: (events: unknown[], { "knowledge-tracing": tracing }) => tracing.iita(tracing.buildEventLog(events)),
  },
  {
    name: "surmiseStructure: 1,000 elements, 12 free and a class of 988, 8,192 states",
    budget: 270,
    heap: 160,
    input: () => ({
      ...Object.fromEntries(ids("e", 12).map((element) => [element, []])),
      ...Object.fromEntries(ids("c", 988).map((element, k) => [element, [`c${((k + 1) % 988) + 1}`]])),
    }),
    read: (prerequisites: unknown, { "knowledge-tracing": tracing }) => tracing.surmiseStructure(prerequisites),
  },
  {
    name: "kst: the masks of 4,096 sets of 16 skills over 65,536 states, the most cells",
    budget: 520,
    heap: 64,
    input: () => {
      const skills = ids("s", 16);
      return {
        skills,
        attempts: Array.from({ length: 4096 }, (_, set) => ({
          learner: "ana",
          item: "q",
          skills: skills.filter((_, k) => (((set + 1) >>> k) & 1) === 1),
          time: 0,
        })),
      };
    },
    // Both processes build the model of 16 skills without prerequisites; the reading fills its masks.
    prepare: ({ skills, attempts }: { skills: string[]; attempts: unknown[] }, { "knowledge-tracing": tracing }) => {
      const structure = tracing.surmiseStructure(Object.fromEntries(skills.map((skill) => [skill, []])), {
        maxStates: 2 ** 16,
      });
      const probabilities = structure.states.map(() => 1 / structure.states.length);
      const model = tracing.kst({ structure, parameters: { probabilities, items: { q: { guess: 0.1, slip: 0.1 } } } });
      return { model, state: model.initialState("ana"), attempts };
    },
    read: ({
      model,
      state,
      attempts,
    }: {
      model: { nextQuestion: (s: unknown, a: unknown[]) => unknown };
      state: unknown;
      attempts: unknown[];
    }) => model.nextQuestion(state, attempts),
  },
  {
    name: "qMatrixIdentifiability: 100,000 items of 20 attributes",
    budget: 320,
    heap: 352,
    input: () => Object.fromEntries(ids("item", 100_000).map((id) => [id, ids("a", 20)])),
    read: (qMatrix: unknown, { psychometrics }) => psychometrics.qMatrixIdentifiability(qMatrix),
  },
  {
    name: "calibrateCdm: 128 items of 11 of 15 attributes, the most items × 2^K and latent groups, two persons, LLM",
    budget: 460,
    heap: 48,
    // Item i measures the 11 attributes from the i-th, round the 15: 2^22 cells and 2^18 latent groups, both limits.
    input: () => ({
      itemIds: ids("item", 128),
      rows: [Array(128).fill(0), Array(128).fill(1)],
      qMatrix: Object.fromEntries(
        ids("item", 128).map((id, i) => [id, Array.from({ length: 11 }, (_, k) => `a${((i + k) % 15) + 1}`)]),
      ),
    }),
    prepare: (
      { itemIds, rows, qMatrix }: { itemIds: string[]; rows: number[][]; qMatrix: unknown },
      { psychometrics },
    ) => ({
      matrix: psychometrics.buildResponseMatrix(rows, { itemIds }),
      qMatrix,
    }),
    read: ({ matrix, qMatrix }: { matrix: unknown; qMatrix: unknown }, { psychometrics }) =>
      psychometrics.calibrateCdm(matrix, qMatrix, { maxIterations: 2, model: "llm" }),
  },
  {
    name: "bkt: the stored parameters of 100,000 skills",
    budget: 250,
    heap: 112,
    input: () => ({
      skills: Object.fromEntries(
        ids("s", 100_000).map((skill) => [skill, { prior: 0.3, learn: 0.1, forget: 0, guess: 0.2, slip: 0.1 }]),
      ),
    }),
    read: (parameters: unknown, { "knowledge-tracing": tracing }) => tracing.bkt({ parameters }),
  },
  {
    name: "manyFacetRasch: 1,994 persons, 1 item, 997 raters, each person by two: 1,000 parameters, one EM iteration",
    budget: 160,
    heap: 32,
    // Person j is scored by raters j and j + 1, round the raters, so that the design connects 997 rater measures: with
    // the threshold, the mean and the SD, the most parameters. One EM iteration without standard errors holds the
    // matrices of every pair of them, and keeps the case within the time of a test; a fit to convergence takes 99 s.
    input: () =>
      Array.from({ length: 1994 }, (_, person) =>
        [0, 1].map((step) => ({
          levels: { person: `p${person + 1}`, item: "i1", rater: `r${((person + step) % 997) + 1}` },
          score: (person + step) % 3,
        })),
      ).flat(),
    read: (scores: unknown[], { psychometrics }) =>
      psychometrics.manyFacetRasch(scores, ["item", "rater"], { standardErrors: "none", maxIterations: 1 }),
  },
  {
    name: "personScalability: 2 persons, 1,000,000 items",
    budget: 400,
    heap: 224,
    input: () => {
      const row = Array.from({ length: 1_000_000 }, (_, i) => i % 2);
      return [row, row.map((response) => 1 - response)];
    },
    prepare: (rows: number[][], { psychometrics }) => psychometrics.buildResponseMatrix(rows),
    read: (matrix: unknown, { psychometrics }) => psychometrics.personScalability(matrix),
  },
  {
    name: "localDependenceChiSquare: 2 persons, 1,000 2PL items, the most, 499,500 pairs",
    budget: 460,
    heap: 224,
    input: () => pairItems(1000),
    prepare: (input: PairItems, { psychometrics }) => pairInput(input, psychometrics),
    read: ({ matrix, items }: { matrix: unknown; items: unknown }, { psychometrics }) =>
      psychometrics.localDependenceChiSquare(matrix, items),
  },
  {
    name: "localDependenceChiSquare: 2 graded items of 4,096 categories, the most cells of its tables",
    budget: 530,
    heap: 32,
    input: () => Array.from({ length: 4095 }, (_, k) => -3 + (6 * k) / 4094),
    prepare: (thresholds: number[], { psychometrics }) => ({
      matrix: psychometrics.buildResponseMatrix([
        [0, 1],
        [1, 0],
      ]),
      items: [0, 1].map(() => ({ model: psychometrics.gradedResponse, parameters: { a: 1, b: thresholds } })),
    }),
    read: ({ matrix, items }: { matrix: unknown; items: unknown }, { psychometrics }) =>
      psychometrics.localDependenceChiSquare(matrix, items),
  },
  {
    name: "summedScoreFit: 2 persons, 300 2PL items, the highest total score",
    budget: 30,
    heap: 32,
    input: () => pairItems(300),
    prepare: (input: PairItems, { psychometrics }) => pairInput(input, psychometrics),
    read: ({ matrix, items }: { matrix: unknown; items: unknown }, { psychometrics }) =>
      psychometrics.summedScoreFit(matrix, items),
  },
  {
    name: "yenQ3: 2 persons, 1,000 2PL items, the most",
    budget: 50,
    heap: 48,
    input: () => pairItems(1000),
    prepare: (input: PairItems, { psychometrics }) => pairInput(input, psychometrics),
    read: ({ matrix, items }: { matrix: unknown; items: unknown }, { psychometrics }) =>
      psychometrics.yenQ3(matrix, items, [0, 0]),
  },
  {
    name: "selectPairs: 10,000 persons in 50 groups of 200, the most, 995,000 pairs, formed by two contexts alike",
    budget: 320,
    heap: 176,
    input: () => {
      const groups = Array.from({ length: 10_000 }, (_, person) => `g${Math.floor(person / 200)}`);
      return { personIds: ids("p", 10_000), contexts: { room: groups, network: groups } };
    },
    read: ({ personIds, contexts }: { personIds: string[]; contexts: unknown }, { psychometrics }) =>
      psychometrics.selectPairs(personIds, contexts),
  },
  {
    name: "m4Index: 2 persons, 1,000 2PL items, the most",
    budget: 50,
    heap: 32,
    input: () => pairItems(1000),
    prepare: (input: PairItems, { psychometrics }) => pairInput(input, psychometrics),
    read: ({ matrix, items }: { matrix: unknown; items: unknown }, { psychometrics }) =>
      psychometrics.m4Index(matrix, items, [0, 0], [["person1", "person2"]]),
  },
  {
    name: "kIndex: 2,896 persons with 0 to 2,895 incorrect responses to 2,895 items, the most cells of its counts",
    budget: 330,
    heap: 448,
    input: () =>
      Array.from({ length: 2896 }, (_, person) => Array.from({ length: 2895 }, (_, i) => (i < person ? 0 : 1))),
    prepare: (rows: number[][], { psychometrics }) => psychometrics.buildResponseMatrix(rows),
    read: (matrix: unknown, { psychometrics }) => psychometrics.kIndex(matrix, [["person2", "person1"]]),
  },
  {
    name: "scalability: 200 persons of distinct patterns, 1,000 items, the most",
    budget: 240,
    heap: 160,
    input: () =>
      Array.from({ length: 200 }, (_, person) => Array.from({ length: 1000 }, (_, i) => ((person + 1) * (i + 3)) % 5)),
    prepare: (rows: number[][], { psychometrics }) => psychometrics.buildResponseMatrix(rows),
    read: (matrix: unknown, { psychometrics }) => psychometrics.scalability(matrix),
  },
  {
    name: "assembleForms: 8,191 forms of the one item of a pool, the most counts",
    budget: 580,
    heap: 48,
    input: () => ({ id: "q1", a: 1, b: 0 }),
    prepare: ({ id, a, b }: { id: string; a: number; b: number }, { psychometrics }) => ({
      pool: [{ id, model: psychometrics.twoParameterLogistic, parameters: { a, b, c: 0, d: 1 } }],
    }),
    read: ({ pool }: { pool: unknown[] }, { psychometrics }) =>
      psychometrics.assembleForms({ pool, length: 1, forms: 8191, overlap: 1 }),
  },
  {
    name: "createDiagnosticAdaptiveTest: 128 items of 11 of 15 attributes, the most items × 2^K and latent groups",
    budget: 180,
    heap: 48,
    input: () => ({
      attributes: ids("a", 15),
      pool: Array.from({ length: 128 }, (_, j) => ({
        id: `q${j + 1}`,
        attributes: Array.from({ length: 11 }, (_, k) => `a${((j + k) % 15) + 1}`),
        probabilities: Array.from({ length: 2 ** 11 }, (_, g) => (g === 2 ** 11 - 1 ? 0.8 : 0.2)),
      })),
    }),
    read: (options: { attributes: string[]; pool: unknown[] }, { psychometrics }) =>
      psychometrics.createDiagnosticAdaptiveTest({ ...options, stopping: psychometrics.testLength(10) }),
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

/** Runs a case in a child process whose heap holds `heap` MiB, reading the input or not; the end of its error when it fails. */
function attempt(directory: string, name: string, read: boolean, heap: number): ChildPeak | string {
  const run = spawnSync(
    process.execPath,
    [`--max-old-space-size=${heap}`, "--expose-gc", import.meta.filename, CHILD, directory, name, String(read)],
    { encoding: "utf8" },
  );
  return run.status === 0 ? (JSON.parse(run.stdout) as ChildPeak) : run.stderr.slice(-1000);
}

/** Runs a case in a child process whose heap holds `heap` MiB, reading the input or not, and returns its peak. */
function childPeak(directory: string, name: string, read: boolean, heap: number): ChildPeak {
  const peak = attempt(directory, name, read, heap);
  if (typeof peak === "string") throw new Error(`The case "${name}" did not end within a heap of ${heap} MiB: ${peak}`);
  return peak;
}

/** Finds the least heap, in steps of 8 MiB, in which a case's reading ends; `undefined` when it needs more than the search tries. */
function leastHeap(directory: string, name: string): number | undefined {
  let low = 1;
  let high = HEAP_SEARCH_LIMIT / HEAP_STEP;
  let found: number | undefined;
  while (low <= high) {
    const middle = Math.floor((low + high) / 2);
    if (typeof attempt(directory, name, true, middle * HEAP_STEP) === "string") {
      low = middle + 1;
    } else {
      found = middle * HEAP_STEP;
      high = middle - 1;
    }
  }
  return found;
}

/** The rise of the peak that reading an input adds, in megabytes, and the megabytes of its JSON. */
export interface PeakRise {
  readonly json: number;
  readonly rise: number;
}

/**
 * Returns the median rise of the peak that a case's reading adds over `pairs` pairs of child processes, one that does
 * not read and one that does, run one after the other, each with a heap of `heap` MiB. Throws when a process does not end.
 *
 * @param directory - The directory of the bundles of {@link bundleLibraries}.
 */
export function peakRise(directory: string, name: string, heap: number, pairs = 1): PeakRise {
  const rises: number[] = [];
  let bytes = 0;
  for (let pair = 0; pair < pairs; pair++) {
    const without = childPeak(directory, name, false, heap);
    const reading = childPeak(directory, name, true, heap);
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
    const heaps = first === LEAST_HEAP;
    const words = heaps ? rest : process.argv.slice(2);
    const chosen = READERS.filter(({ name }) => words.length === 0 || words.some((word) => name.startsWith(word)));
    const directory = mkdtempSync(join(tmpdir(), "measurekit-peak-"));
    try {
      await bundleLibraries(directory);
      if (heaps) {
        console.log("| Reader | Least heap (MiB) | Heap (MiB) |");
        console.log("| ------ | ---------------: | ---------: |");
        for (const { name, heap } of chosen) {
          console.log(`| ${name} | ${leastHeap(directory, name) ?? `above ${HEAP_SEARCH_LIMIT}`} | ${heap} |`);
        }
      } else {
        console.log("| Reader | JSON (MB) | Peak added (MB) | Budget (MB) |");
        console.log("| ------ | --------: | --------------: | ----------: |");
        for (const { name, budget } of chosen) {
          const { json, rise } = peakRise(directory, name, TABLE_HEAP, 3);
          console.log(`| ${name} | ${json.toFixed(1)} | ${Math.round(rise)} | ${budget} |`);
        }
      }
    } finally {
      rmSync(directory, { recursive: true, force: true });
    }
  }
}
