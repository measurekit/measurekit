import { checkFinite, DataError, mean } from "@measurekit/core";
import { readStations } from "./stations.js";

/** The borderline regression of one station. */
export interface StationRegression {
  /** The station's cut: the line's mark at the borderline grade; `null` when the ratings do not vary. */
  readonly cutScore: number | null;
  /**
   * The standard error of the cut, s √(1/n + (x₀ - x̄)² / Sxx), as R's `predict(se.fit = TRUE)`; `null` without a line
   * or with fewer than three persons.
   */
  readonly standardError: number | null;
  /** The share of the marks' variance the ratings explain, 1 - RSS / TSS; `null` without a line or when the marks agree. */
  readonly rSquared: number | null;
  /** The line's mark at a rating of 0; `null` without a line. */
  readonly intercept: number | null;
  /** The inter-grade discrimination: the line's rise in marks per grade; `null` without a line. */
  readonly slope: number | null;
}

/** The cut scores of a performance test by borderline regression. */
export interface BorderlineRegressionCutScores {
  /** The regression of each station, in station order. */
  readonly stations: readonly StationRegression[];
  /** The test's cut on the sum of the stations' marks: the sum of the stations' cuts; `null` when one has none. */
  readonly cutScore: number | null;
  /**
   * The standard error of the test's cut, with the covariances of the stations' cuts; `null` when a station has no
   * line or there are fewer than three persons.
   */
  readonly standardError: number | null;
}

/**
 * The most products over the persons of the stations' covariances, persons × stations²: the test's standard error
 * sums one over the persons for each pair of stations. At the limit, 100 stations take 0.3 s and 20 stations 1.4 s.
 */
const MAX_PAIR_PRODUCTS = 2 ** 28;

/**
 * Returns the cut score of each station and of the test by the borderline regression method: the regression of each
 * station's checklist marks on its global ratings, read at the borderline grade.
 *
 * @remarks
 * For each station, the least-squares line of the marks y on the ratings x, as R's `lm(y ~ x)`, gives the cut
 * ȳ + b (x₀ - x̄) at the borderline grade x₀, with b = Sxy / Sxx, the inter-grade discrimination, and R² = 1 - RSS / TSS.
 * The cut's standard error is s √(1/n + (x₀ - x̄)² / Sxx), with s² = RSS / (n - 2). The test's cut is the sum of the
 * stations' cuts, on the scale of the sum of a person's marks; a cut on the mean of the stations' marks is it divided by
 * the stations, and classifies alike. Each station's cut is Σᵢ wᵢ yᵢ with wᵢ = 1/n + (x₀ - x̄)(xᵢ - x̄) / Sxx, so two
 * stations' cuts have the covariance σ_st Σᵢ w_si w_ti, σ_st the covariance of a person's residuals at the two,
 * estimated by their cross products over n - 2; the test's standard error is the root of the sum of these over every
 * pair of stations. It holds the ratings fixed, as each station's does. Ratings are numbers, such as 1 fail, 2
 * borderline, 3 pass and 4 superior, and the borderline grade a point on their scale, such as 2, or 2.5 between a
 * borderline fail and a borderline pass.
 *
 * @param marks - Each person's checklist mark at each station, by person, then station.
 * @param ratings - Each person's global rating at each station, in the order of `marks`.
 * @param borderline - The borderline grade on the scale of the ratings.
 * @returns The regression of each station, and the test's cut and its standard error.
 * @throws `DataError` when the marks or the ratings are not finite numbers, one per person and station, the same
 * for both, there is no person or station, `borderline` is not a finite number, persons × stations² is above 2^28, or
 * a result is not finite because the marks, the ratings or the borderline grade are too large.
 * @see {@link https://doi.org/10.1046/j.1365-2923.2003.01429.x | Kramer et al. (2003)}
 * @see {@link https://doi.org/10.3109/0142159X.2010.507716 | Pell, Fuller, Homer and Roberts (2010)}
 * @see {@link https://doi.org/10.1007/s10459-005-7853-1 | Wood, Humphrey-Murto and Norman (2006)}
 * @see {@link https://doi.org/10.1080/01621459.1962.10480664 | Zellner (1962)}
 */
export function borderlineRegressionCutScores(
  marks: readonly (readonly number[])[],
  ratings: readonly (readonly number[])[],
  borderline: number,
): BorderlineRegressionCutScores {
  checkFinite(borderline, "The borderline grade");
  const stations = readStations(marks, ratings);
  const { persons } = stations;
  const count = stations.marks.length;
  if (persons * count * count > MAX_PAIR_PRODUCTS) {
    throw new DataError(
      `Borderline regression sums over the persons for each pair of stations, so it takes at most ${MAX_PAIR_PRODUCTS} persons × stations², found ${persons} × ${count}²; set the standard of fewer persons or stations at a time.`,
    );
  }
  const fits = stations.marks.map((y, s) => stationFit(y, stations.ratings[s] ?? y, borderline, s));
  const results = fits.map((fit) => fit?.result ?? NO_LINE);
  const lines = fits.filter((fit): fit is StationFit => fit !== null);
  if (lines.length < fits.length) return { stations: results, cutScore: null, standardError: null };
  const cutScore = lines.reduce((sum, fit) => sum + (fit.result.cutScore ?? 0), 0);
  const standardError = persons < 3 ? null : totalStandardError(lines, persons);
  if (!(Number.isFinite(cutScore) && (standardError === null || Number.isFinite(standardError)))) throw tooLarge();
  return { stations: results, cutScore, standardError };
}

const NO_LINE: StationRegression = {
  cutScore: null,
  standardError: null,
  rSquared: null,
  intercept: null,
  slope: null,
};

/** A station's line, with what the test's standard error needs. */
interface StationFit {
  readonly result: StationRegression;
  /** Each person's residual. */
  readonly residuals: Float64Array;
  /** Each person's rating less their mean. */
  readonly deviations: Float64Array;
  /** (x₀ - x̄) / Sxx, the weight of a person's deviation in the cut. */
  readonly leverage: number;
}

/**
 * Returns a station's least-squares line of the marks y on the ratings x, or `null` when the ratings do not vary.
 *
 * @throws `DataError` when a sum or a result is not finite.
 */
function stationFit(y: Float64Array, x: Float64Array, borderline: number, station: number): StationFit | null {
  const n = y.length;
  const meanX = mean(x);
  const meanY = mean(y);
  const deviations = Float64Array.from(x, (value) => value - meanX);
  let sxx = 0;
  let sxy = 0;
  let syy = 0;
  for (let i = 0; i < n; i++) {
    const dx = deviations[i] ?? 0;
    const dy = (y[i] ?? 0) - meanY;
    sxx += dx * dx;
    sxy += dx * dy;
    syy += dy * dy;
  }
  if (!(Number.isFinite(sxx) && Number.isFinite(sxy) && Number.isFinite(syy))) throw tooLarge(station);
  if (sxx === 0) return null;
  const slope = sxy / sxx;
  const residuals = new Float64Array(n);
  let rss = 0;
  for (let i = 0; i < n; i++) {
    const residual = (y[i] ?? 0) - meanY - slope * (deviations[i] ?? 0);
    residuals[i] = residual;
    rss += residual * residual;
  }
  const offset = borderline - meanX;
  const fit = {
    result: {
      cutScore: meanY + slope * offset,
      standardError: n < 3 ? null : Math.sqrt((rss / (n - 2)) * (1 / n + (offset * offset) / sxx)),
      // A line through marks that all agree explains no variance and leaves none: R² is 0 / 0.
      rSquared: syy > 0 ? Math.max(0, 1 - rss / syy) : null,
      intercept: meanY - slope * meanX,
      slope,
    },
    residuals,
    deviations,
    leverage: offset / sxx,
  };
  for (const value of Object.values(fit.result)) {
    if (value !== null && !Number.isFinite(value)) throw tooLarge(station);
  }
  return fit;
}

/**
 * Returns the standard error of the sum of the stations' cuts: the root of Σ_s Σ_t σ_st W_st, with σ_st = Σᵢ e_si e_ti
 * / (n - 2) and W_st = Σᵢ w_si w_ti = 1/n + k_s k_t Σᵢ d_si d_ti, for d the ratings' deviations and k = (x₀ - x̄) / Sxx.
 */
function totalStandardError(fits: readonly StationFit[], persons: number): number {
  let variance = 0;
  fits.forEach((first, s) => {
    for (let t = s; t < fits.length; t++) {
      const term = stationCovariance(first, fits[t] ?? first, persons);
      variance += t === s ? term : 2 * term;
    }
  });
  // The sum is the trace of a product of two positive semidefinite matrices, so only rounding takes it below 0.
  return Math.sqrt(Math.max(0, variance));
}

/** Returns the covariance of two stations' cuts, σ_st W_st. */
function stationCovariance(first: StationFit, second: StationFit, persons: number): number {
  let residuals = 0;
  let deviations = 0;
  for (let i = 0; i < persons; i++) {
    residuals += (first.residuals[i] ?? 0) * (second.residuals[i] ?? 0);
    deviations += (first.deviations[i] ?? 0) * (second.deviations[i] ?? 0);
  }
  return (residuals / (persons - 2)) * (1 / persons + first.leverage * second.leverage * deviations);
}

/** Returns the error of a result that is not finite, at a station or of the test. */
function tooLarge(station?: number): DataError {
  const where = station === undefined ? "the test" : `station ${station + 1}`;
  return new DataError(
    `The borderline regression of ${where} is not finite: the marks, the ratings or the borderline grade are too large; rescale them.`,
  );
}
