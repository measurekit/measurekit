import { checkOptions, checkUnitInterval, DataError, type FieldNames } from "@measurekit/core";
import { type Attempt, checkAttempts } from "../data/internal.js";
import { checkModel, type LearnerModel } from "../model.js";

const DEFAULT_BAND = [0.65, 0.85] as const;

/** A candidate attempt with the model's prediction of its outcome. */
export interface Recommendation {
  /** The candidate attempt. */
  readonly attempt: Attempt;
  /** The model's predicted outcome: P(correct) for a dichotomous item. */
  readonly prediction: number;
}

/** Options of {@link recommend}. */
export interface RecommendOptions {
  /**
   * The band of predicted success to aim for, `[lower, upper]`: its middle is the success rate that adaptive practice
   * of facts targets, its upper end the 85% accuracy at which learning is fastest.
   *
   * @defaultValue `[0.65, 0.85]`
   * @see {@link https://educationaldatamining.org/EDM2014/uploads/procs2014/long%20papers/6_EDM-2014-Full.pdf |
   * Papoušek, Pelánek and Stanislav (2014)}
   * @see {@link https://doi.org/10.1038/s41467-019-12552-4 | Wilson et al. (2019)}
   */
  readonly band?: readonly [lower: number, upper: number];
}

const RECOMMEND_OPTIONS: FieldNames<RecommendOptions> = { band: true };

/**
 * Ranks candidate attempts by the learner's predicted success: the attempts predicted inside the band first,
 * nearest its middle first, then the others, nearest the band first. Ties keep the candidates' order. The first is
 * the next item to practice; filter the candidates first, for instance to the skills not yet mastered.
 *
 * @param model - The learner model.
 * @param state - The learner's state.
 * @param candidates - The attempts to choose from.
 * @param options - The band of predicted success to aim at.
 * @returns The candidates with their predictions, ranked: those predicted inside the band first, nearest its middle,
 *   and the others by distance from the band.
 * @throws `DataError` when the model is not a learner model, the band is not within 0 to 1 with lower ≤ upper, or the
 * model rejects the state or an attempt.
 *
 * @example
 * ```ts
 * import { elo } from "@measurekit/knowledge-tracing/elo";
 * import { recommend } from "@measurekit/knowledge-tracing/practice";
 *
 * const model = elo({
 *   parameters: {
 *     items: {
 *       q1: { difficulty: -2.5, count: 40 },
 *       q2: { difficulty: -1.1, count: 40 },
 *       q3: { difficulty: 0.3, count: 40 },
 *     },
 *   },
 * });
 * const ana = model.initialState("ana");
 * const candidates = ["q1", "q2", "q3"].map((item) => ({ learner: "ana", item, skills: [], time: 1 }));
 * const ranked = recommend(model, ana, candidates).map(({ attempt, prediction }) => [attempt.item, prediction]);
 * ranked; // => [["q2", 0.75], ["q1", 0.92], ["q3", 0.43]], the items nearest the band of success first
 * ```
 */
export function recommend<TState>(
  model: LearnerModel<TState>,
  state: TState,
  candidates: readonly Attempt[],
  options: RecommendOptions = {},
): readonly Recommendation[] {
  checkModel(model, "A recommendation needs a learner model, such as bkt()");
  checkOptions(options, RECOMMEND_OPTIONS);
  const { band = DEFAULT_BAND } = options;
  const [lower, upper] = checkBand(band);
  checkAttempts(candidates, "Candidate");
  const middle = (lower + upper) / 2;
  const ranked = candidates.map((attempt, index) => {
    const prediction = model.predict(state, attempt);
    checkUnitInterval(prediction, `The model's prediction of candidate ${index + 1}`);
    return { attempt, prediction, outside: Math.max(lower - prediction, prediction - upper, 0) };
  });
  ranked.sort((x, y) => x.outside - y.outside || Math.abs(x.prediction - middle) - Math.abs(y.prediction - middle));
  return ranked.map(({ attempt, prediction }) => ({ attempt, prediction }));
}

/** Checks a band and returns its ends. */
function checkBand(band: unknown): readonly [number, number] {
  if (Array.isArray(band) && band.length === 2) {
    const [lower, upper] = band;
    if (typeof lower === "number" && typeof upper === "number" && lower >= 0 && lower <= upper && upper <= 1) {
      return [lower, upper];
    }
  }
  throw new DataError(`The band must be [lower, upper] with 0 ≤ lower ≤ upper ≤ 1, found ${JSON.stringify(band)}.`);
}
