import { checkArray, checkChoice, checkFinite, checkOptions, DataError, mean } from "@measurekit/core";
import { sampleQuantile, standardDeviation } from "../math/internal.js";
import { readStations } from "./stations.js";

/** Options of {@link borderlineGroupCutScores}. */
export interface BorderlineGroupOptions {
  /**
   * The statistic of the borderline group's marks: `"mean"`, as OSCE guides take it, or `"median"`, as Livingston and
   * Zieky's method does.
   *
   * @defaultValue `"mean"`
   */
  readonly statistic?: "mean" | "median";
}

/** The borderline group of one station. */
export interface StationGroup {
  /** The station's cut: the statistic of the group's marks; `null` when no person has a borderline grade. */
  readonly cutScore: number | null;
  /** The standard error of a mean, sd / √n (denominator n - 1); `null` for the median and with fewer than two persons. */
  readonly standardError: number | null;
  /** The persons with a borderline grade at the station. */
  readonly persons: number;
}

/** The cut scores of a performance test by the borderline group method. */
export interface BorderlineGroupCutScores {
  /** The group of each station, in station order. */
  readonly stations: readonly StationGroup[];
  /** The test's cut on the sum of the stations' marks: the sum of the stations' cuts; `null` when one has none. */
  readonly cutScore: number | null;
}

const STATISTICS = ["mean", "median"];

/**
 * Returns the cut score of each station and of the test by the borderline group method: the mean or the median of the
 * checklist marks of the persons whose global rating at the station is a borderline grade.
 *
 * @remarks
 * Each station's group holds the persons it rated at a borderline grade, so the groups of two stations differ. With
 * several grades, such as a borderline fail and a borderline pass, the group holds the persons at any of them, as the
 * modified borderline group method does. The test's cut is the sum of the stations' cuts, on the scale of the sum of a
 * person's marks; a cut on the mean of the stations' marks is it divided by the stations, and classifies alike.
 *
 * @param marks - Each person's checklist mark at each station, by person, then station.
 * @param ratings - Each person's global rating at each station, in the order of `marks`.
 * @param borderline - The borderline grade, or the grades, on the scale of the ratings.
 * @param options - The statistic of the group's marks.
 * @returns The group of each station and the test's cut.
 * @throws `DataError` when the marks or the ratings are not finite numbers, one per person and station, the same
 * for both, there is no person or station, `borderline` is not a finite number or a list of at least one, or
 * `statistic` is not `"mean"` or `"median"`.
 * @see {@link https://doi.org/10.4135/9781412985918 | Cizek and Bunch (2007)}
 * @see {@link https://doi.org/10.3109/0142159X.2013.853119 | McKinley and Norcini (2014)}
 * @see {@link https://doi.org/10.1080/0142159X.2017.1296563 | Homer, Pell and Fuller (2017)}
 */
export function borderlineGroupCutScores(
  marks: readonly (readonly number[])[],
  ratings: readonly (readonly number[])[],
  borderline: number | readonly number[],
  options: BorderlineGroupOptions = {},
): BorderlineGroupCutScores {
  checkOptions(options);
  const { statistic = "mean" } = options;
  checkChoice(statistic, "The option statistic", STATISTICS);
  const grades = borderlineGrades(borderline);
  const { marks: byStation, ratings: ratingsByStation } = readStations(marks, ratings);
  const stations = byStation.map((stationMarks, s): StationGroup => {
    const stationRatings = ratingsByStation[s] ?? stationMarks;
    const group = stationMarks.filter((_, person) => grades.has(stationRatings[person] ?? 0));
    const persons = group.length;
    if (persons === 0) return { cutScore: null, standardError: null, persons };
    const station =
      statistic === "median"
        ? { cutScore: sampleQuantile(group.sort(), 0.5), standardError: null, persons }
        : {
            cutScore: mean(group),
            standardError: persons < 2 ? null : standardDeviation(group) / Math.sqrt(persons),
            persons,
          };
    if (
      !(Number.isFinite(station.cutScore) && (station.standardError === null || Number.isFinite(station.standardError)))
    ) {
      throw tooLarge(`station ${s + 1}`);
    }
    return station;
  });
  if (stations.some((station) => station.cutScore === null)) return { stations, cutScore: null };
  const cutScore = stations.reduce((sum, station) => sum + (station.cutScore ?? 0), 0);
  if (!Number.isFinite(cutScore)) throw tooLarge("the test");
  return { stations, cutScore };
}

/** Returns the error of a cut that is not finite, at a station or of the test. */
function tooLarge(where: string): DataError {
  return new DataError(`The borderline group's cut of ${where} is not finite: the marks are too large; rescale them.`);
}

/**
 * Returns the borderline grades as a set.
 *
 * @throws `DataError` when they are not a finite number or a list of at least one.
 */
function borderlineGrades(borderline: number | readonly number[]): ReadonlySet<number> {
  if (typeof borderline === "number") {
    checkFinite(borderline, "The borderline grade");
    return new Set([borderline]);
  }
  checkArray(borderline, "The borderline grades must be a number or an array of numbers");
  if (borderline.length === 0) throw new DataError("The borderline grades need at least one grade.");
  borderline.forEach((grade, index) => {
    checkFinite(grade, `Borderline grade ${index + 1}`);
  });
  return new Set(borderline);
}
