import {
  checkOptions,
  checkRandom,
  checkStopping,
  DataError,
  describe,
  type FieldNames,
  type Random,
} from "@measurekit/core";
import { type AdaptiveTestOptions, createAdaptiveTest } from "./adaptive-test.js";
import type { PoolItem } from "./items.js";
import { type ExposureRate, simulateAdaptiveTest } from "./simulation.js";

/** Options of {@link calibrateExposure}. */
export interface ExposureCalibrationOptions {
  /** The largest exposure rate allowed, r: above 0 and at most 1. */
  readonly maxRate: number;
  /**
   * The procedure has converged when no exposure rate exceeds `maxRate` by more than this. Take it above the sampling
   * error of a rate, √(r (1 - r) / n) for n thetas, several times over, as the largest of many rates exceeds r by about
   * 2.5 of them.
   *
   * @defaultValue `0.02`
   */
  readonly tolerance?: number;
  /**
   * The most simulations to run.
   *
   * @defaultValue `50`
   */
  readonly maxIterations?: number;
}

const EXPOSURE_CALIBRATION_OPTIONS: FieldNames<ExposureCalibrationOptions> = {
  maxRate: true,
  tolerance: true,
  maxIterations: true,
};

/** Sympson-Hetter exposure parameters calibrated by simulation. */
export interface ExposureCalibration {
  /** The pool, each item with its exposure parameter. */
  readonly pool: readonly PoolItem[];
  /** The exposure rates of the last simulation, with these parameters. */
  readonly exposureRates: readonly ExposureRate[];
  /** Whether no exposure rate of the last simulation exceeds `maxRate` by more than `tolerance`. */
  readonly converged: boolean;
  /** The simulations run. */
  readonly iterations: number;
}

const DEFAULT_TOLERANCE = 0.02;
const DEFAULT_MAX_ITERATIONS = 50;

/**
 * Calibrates Sympson-Hetter exposure parameters by repeated simulation.
 *
 * @remarks
 * Every item starts at k = 1; each round simulates the adaptive test of `test` for persons with the given thetas, and,
 * with P(S) the share of persons who had an item selected, administered or withheld, sets k = r / P(S) where P(S) > r
 * and k = 1 elsewhere, until no exposure rate exceeds r by more than `tolerance`. Exposure parameters in `test.pool`
 * are ignored. The same `random` draws every simulation, so a seed reproduces the calibration. When the test length
 * divided by the pool size exceeds r, no parameters can meet r and the procedure does not converge.
 *
 * @param test - The options of the test, with the pool whose exposure parameters are calibrated.
 * @param thetas - The abilities of the persons simulated.
 * @param random - The generator of the simulations.
 * @param options - The largest exposure rate, the tolerance and the most iterations.
 * @returns The pool with the calibrated exposure parameters, the exposure rate of each item, whether the calibration
 *   converged and the iterations run.
 * @throws `DataError` when the options of the test or of the procedure are invalid, or without thetas.
 * @see {@link https://doi.org/10.1037/10244-014 | Hetter and Sympson (1997)}
 */
export function calibrateExposure(
  test: AdaptiveTestOptions,
  thetas: readonly number[],
  random: Random,
  options: ExposureCalibrationOptions,
): ExposureCalibration {
  checkOptions(options, EXPOSURE_CALIBRATION_OPTIONS);
  const { maxRate, tolerance = DEFAULT_TOLERANCE, maxIterations = DEFAULT_MAX_ITERATIONS } = options;
  if (!(typeof maxRate === "number" && maxRate > 0 && maxRate <= 1)) {
    throw new DataError(`The option maxRate must be a number above 0 and at most 1, found ${describe(maxRate)}.`);
  }
  checkStopping(tolerance, maxIterations);
  checkRandom(random);
  const items = createAdaptiveTest(test).pool;
  let parameters = items.map(() => 1);
  for (let iteration = 1; ; iteration++) {
    const pool = items.map((item, i) => ({ ...item, exposure: parameters[i] ?? 1 }));
    const { persons, exposureRates } = simulateAdaptiveTest(createAdaptiveTest({ ...test, pool }), thetas, random);
    const converged = exposureRates.every(({ rate }) => rate <= maxRate + tolerance);
    if (converged || iteration === maxIterations) return { pool, exposureRates, converged, iterations: iteration };
    const selected = new Map(pool.map(({ id }) => [id, 0]));
    for (const { state } of persons) {
      for (const id of [...state.items, ...state.withheld]) selected.set(id, (selected.get(id) ?? 0) + 1);
    }
    parameters = pool.map(({ id }) => {
      const rate = (selected.get(id) ?? 0) / persons.length;
      return rate > maxRate ? maxRate / rate : 1;
    });
  }
}
