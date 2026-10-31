import { symmetrized } from "./matrix.js";
import { type PersonResponses, responsesByPerson } from "./mhrm-chain.js";
import { latentObservedInformation, personLatentScores, secondMoments } from "./mhrm-latent.js";
import { observedItemInformation, priorMoments, responseScores } from "./mhrm-moments.js";
import { drawCycle, type Run } from "./mhrm-run.js";
import { covarianceOfInformation } from "./standard-errors.js";

/** The cycles of draws at the estimates that the standard errors use: mirt's `SE.type = "MHRM"` runs at least 400. */
const STANDARD_ERROR_CYCLES = 400;
/** The persons' scores are sampled every this many cycles, so that the draws they average over are nearly independent. */
const SCORE_INTERVAL = 4;
/** The observed complete-data information is computed every this many cycles, as it varies little between draws. */
const STANDARD_ERROR_INFORMATION_INTERVAL = 10;

/** The persons' complete-data scores at the draws of the sampled cycles. */
interface Scores {
  readonly people: PersonResponses;
  /** The start of each response's entries in `current` and `sums`, the responses in item-major order. */
  readonly offsets: Int32Array;
  readonly current: Float64Array;
  readonly sums: Float64Array;
  /** The correlations' scores by person, `pairs` entries each: this cycle's, and their sums over the sampled cycles. */
  readonly latentCurrent: Float64Array;
  readonly latentSums: Float64Array;
}

/**
 * Returns the covariance of the estimated parameters on their own scale, by rows, in the order of the items' estimated
 * entries and then the correlations; `null` when the information cannot be inverted or gives a variance that is not
 * positive.
 *
 * @remarks
 * Louis' formula from draws at the estimates: the observed information of the log-posterior is the mean of the observed
 * complete-data information, plus the priors' information, less the variance of the complete-data score under the
 * posterior. Persons are independent given the parameters, so that variance is the sum over persons of each person's
 * variance of the score across the person's draws, estimated from every fourth cycle of 400. Estimating it person by
 * person averages the noise of the draws over the persons, which a variance of the total score across cycles does not.
 * `variance` is the proposal's of the calibration.
 *
 * @see {@link https://doi.org/10.1111/j.2517-6161.1982.tb01203.x | Louis (1982)}
 */
export function observedInformation<TParameters>(run: Run<TParameters>, variance: number): Float64Array | null {
  const { layout, items } = run;
  const size = layout.size;
  const scores = createScores(run);
  const second = new Float64Array(size * size);
  const complete = new Float64Array(size * size);
  let sampled = 0;
  let observed = 0;
  for (let cycle = 0; cycle < STANDARD_ERROR_CYCLES; cycle++) {
    drawCycle(run, variance, false);
    if (cycle % SCORE_INTERVAL === 0) {
      sampleScores(run, scores);
      addProducts(run, scores, scores.current, scores.latentCurrent, 1, second);
      sampled++;
    }
    if (cycle % STANDARD_ERROR_INFORMATION_INTERVAL === 0) {
      addCompleteInformation(run, complete);
      observed++;
    }
  }
  const sumProducts = new Float64Array(size * size);
  addProducts(run, scores, scores.sums, scores.latentSums, 1 / sampled, sumProducts);
  const information = new Float64Array(size * size);
  for (let k = 0; k < information.length; k++) {
    const covariance = ((second[k] ?? 0) - (sumProducts[k] ?? 0)) / (sampled - 1);
    information[k] = (complete[k] ?? 0) / observed - covariance;
  }
  mirror(information, size);
  addPriorInformation(run, information);
  return covarianceOfInformation(symmetrized(information, size), items, size);
}

function createScores<TParameters>({ items, layout, draws }: Run<TParameters>): Scores {
  const people = responsesByPerson(draws.responses);
  const offsets = new Int32Array(people.itemOf.length);
  let total = 0;
  people.itemOf.forEach((item, r) => {
    offsets[r] = total;
    total += items[item]?.free.length ?? 0;
  });
  return {
    people,
    offsets,
    current: new Float64Array(total),
    sums: new Float64Array(total),
    latentCurrent: new Float64Array(draws.responses.count * layout.pairs),
    latentSums: new Float64Array(draws.responses.count * layout.pairs),
  };
}

/** Computes every person's complete-data score at the current draws and adds it to the sums. */
function sampleScores<TParameters>(run: Run<TParameters>, scores: Scores): void {
  const { items, draws, scratch, correlations, layout } = run;
  items.forEach((item, i) => {
    responseScores(item, i, draws, scratch, scores.current, scores.offsets);
  });
  for (let k = 0; k < scores.current.length; k++) scores.sums[k] = (scores.sums[k] ?? 0) + (scores.current[k] ?? 0);
  if (layout.pairs === 0) return;
  personLatentScores(correlations, draws.theta, draws.responses.count, scores.latentCurrent);
  for (let k = 0; k < scores.latentCurrent.length; k++) {
    scores.latentSums[k] = (scores.latentSums[k] ?? 0) + (scores.latentCurrent[k] ?? 0);
  }
}

/**
 * Adds `weight` times the sum over persons of the outer product of a person's score vector with itself to `target`,
 * where the vector holds the entries of `items`, by response, and `latent`, by person. The items' entries of two
 * different items go to the block of the earlier item's slots.
 */
function addProducts<TParameters>(
  run: Run<TParameters>,
  { people, offsets }: Scores,
  items: Float64Array,
  latent: Float64Array,
  weight: number,
  target: Float64Array,
): void {
  const { layout } = run;
  const size = layout.size;
  const slots = (response: number): { item: number; slot: number; length: number; base: number } => {
    const item = people.itemOf[response] ?? 0;
    return {
      item,
      slot: layout.offsets[item] ?? 0,
      length: run.items[item]?.free.length ?? 0,
      base: offsets[response] ?? 0,
    };
  };
  for (let person = 0; person < run.draws.responses.count; person++) {
    const start = people.starts[person] ?? 0;
    const end = people.starts[person + 1] ?? 0;
    for (let a = start; a < end; a++) {
      const first = slots(people.responses[a] ?? 0);
      for (let b = a; b < end; b++) addPair(target, size, items, first, slots(people.responses[b] ?? 0), weight);
      addLatentCross(
        target,
        size,
        items,
        first,
        { latent, person, offset: layout.latent, pairs: layout.pairs },
        weight,
      );
    }
    addLatentBlock(target, size, latent, person, layout, weight);
  }
}

interface Slots {
  readonly slot: number;
  readonly length: number;
  readonly base: number;
}

/** Adds the outer product of two responses' score entries to the block of the first's slots. */
function addPair(
  target: Float64Array,
  size: number,
  items: Float64Array,
  first: Slots,
  second: Slots,
  weight: number,
): void {
  for (let l = 0; l < first.length; l++) {
    const left = weight * (items[first.base + l] ?? 0);
    const row = (first.slot + l) * size + second.slot;
    for (let m = 0; m < second.length; m++)
      target[row + m] = (target[row + m] ?? 0) + left * (items[second.base + m] ?? 0);
  }
}

/** A person's correlation scores in `latent`, and where their slots start. */
interface LatentScores {
  readonly latent: Float64Array;
  readonly person: number;
  readonly offset: number;
  readonly pairs: number;
}

/** Adds the outer product of a response's score entries with the person's correlation scores. */
function addLatentCross(
  target: Float64Array,
  size: number,
  items: Float64Array,
  first: Slots,
  { latent, person, offset, pairs }: LatentScores,
  weight: number,
): void {
  for (let l = 0; l < first.length; l++) {
    const left = weight * (items[first.base + l] ?? 0);
    for (let q = 0; q < pairs; q++) {
      const index = (first.slot + l) * size + offset + q;
      target[index] = (target[index] ?? 0) + left * (latent[person * pairs + q] ?? 0);
    }
  }
}

/** Adds the outer product of a person's correlation scores with themselves. */
function addLatentBlock(
  target: Float64Array,
  size: number,
  latent: Float64Array,
  person: number,
  { pairs, latent: offset }: { pairs: number; latent: number },
  weight: number,
): void {
  for (let q = 0; q < pairs; q++) {
    for (let r = 0; r < pairs; r++) {
      const index = (offset + q) * size + offset + r;
      target[index] =
        (target[index] ?? 0) + weight * (latent[person * pairs + q] ?? 0) * (latent[person * pairs + r] ?? 0);
    }
  }
}

/** Copies the upper triangle of a symmetric matrix onto its lower triangle where the lower entry was not written. */
function mirror(matrix: Float64Array, size: number): void {
  for (let i = 0; i < size; i++) {
    for (let j = 0; j < i; j++) if ((matrix[i * size + j] ?? 0) === 0) matrix[i * size + j] = matrix[j * size + i] ?? 0;
  }
}

/** Adds the observed complete-data information at the current draws, block by block, to `out`. */
function addCompleteInformation<TParameters>(run: Run<TParameters>, out: Float64Array): void {
  const { items, layout, correlations, draws, scratch, respondents } = run;
  items.forEach((item, i) => {
    const length = item.free.length;
    const block = new Float64Array(length * length);
    observedItemInformation(item, i, draws, scratch, block, true);
    addBlock(out, layout.size, layout.offsets[i] ?? 0, block, length);
  });
  if (layout.pairs === 0) return;
  const second = new Float64Array(correlations.dimensions ** 2);
  secondMoments(draws.theta, respondents, correlations.dimensions, second);
  const block = new Float64Array(layout.pairs * layout.pairs);
  latentObservedInformation(correlations, second, respondents, block);
  addBlock(out, layout.size, layout.latent, block, layout.pairs);
}

/** Adds a square block to `matrix`, of `size` columns, with its first entry at `offset` on the diagonal. */
function addBlock(matrix: Float64Array, size: number, offset: number, block: Float64Array, length: number): void {
  for (let l = 0; l < length; l++) {
    for (let m = 0; m < length; m++) {
      const index = (offset + l) * size + offset + m;
      matrix[index] = (matrix[index] ?? 0) + (block[l * length + m] ?? 0);
    }
  }
}

/** Adds the information of the log priors of the items' entries, on the unbounded scale, to the diagonal blocks. */
function addPriorInformation<TParameters>({ items, layout }: Run<TParameters>, information: Float64Array): void {
  items.forEach((item, i) => {
    const length = item.free.length;
    const block = new Float64Array(length * length);
    priorMoments(item, new Float64Array(length), block);
    addBlock(information, layout.size, layout.offsets[i] ?? 0, block, length);
  });
}
