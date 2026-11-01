import fc from "fast-check";
import {
  fourParameterLogistic,
  generalizedPartialCredit,
  gradedResponse,
  type IrtItem,
  nominalResponse,
  partialCredit,
  ratingScale,
} from "../src/index.js";

/** A slope of any sign, including 0. */
export const arbitrarySlope = fc.double({ min: -3, max: 4, noNaN: true });

/** A difficulty, threshold or intercept. */
export const arbitraryLocation = fc.double({ min: -4, max: 4, noNaN: true });

/** One to four thresholds in any order. */
export const arbitraryLocations = fc.array(arbitraryLocation, { minLength: 1, maxLength: 4 });

/** Items of every model, with parameters in range: the 4PL, GRM, GPCM, PCM, RSM and NRM. */
export const arbitraryItem: fc.Arbitrary<IrtItem> = fc.oneof(
  fc
    .record({
      a: arbitrarySlope,
      b: arbitraryLocation,
      c: fc.double({ min: 0, max: 0.4, noNaN: true }),
      d: fc.double({ min: 0.6, max: 1, noNaN: true }),
    })
    .map((parameters) => ({ model: fourParameterLogistic, parameters })),
  fc
    .record({
      a: fc.double({ min: 0.2, max: 3, noNaN: true }),
      positive: fc.boolean(),
      first: arbitraryLocation,
      gaps: fc.array(fc.double({ min: 0.05, max: 2, noNaN: true }), { maxLength: 3 }),
    })
    .map(({ a, positive, first, gaps }) => {
      const b = [first];
      for (const gap of gaps) b.push((b.at(-1) ?? 0) + gap);
      return { model: gradedResponse, parameters: positive ? { a, b } : { a: -a, b: b.reverse() } };
    }),
  fc
    .record({ a: arbitrarySlope, b: arbitraryLocations })
    .map((parameters) => ({ model: generalizedPartialCredit, parameters })),
  fc.record({ a: arbitrarySlope, b: arbitraryLocations }).map((parameters) => ({ model: partialCredit, parameters })),
  fc
    .record({ a: arbitrarySlope, b: arbitraryLocation, steps: arbitraryLocations })
    .map((parameters) => ({ model: ratingScale, parameters })),
  fc
    .integer({ min: 2, max: 5 })
    .chain((count) =>
      fc.record({
        a: fc.array(arbitrarySlope, { minLength: count, maxLength: count }),
        c: fc.array(arbitraryLocation, { minLength: count, maxLength: count }),
      }),
    )
    .map((parameters) => ({ model: nominalResponse, parameters })),
);
