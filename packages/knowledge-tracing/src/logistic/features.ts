import { checkNonNegative, checkPositive, DataError, describe, isRecord } from "@measurekit/core";
import { entry } from "../data/record.js";
import { type ExposureHistory, type HistoryNeeds, leakyAt, type SkillHistory } from "./state.js";

/**
 * A feature of a learner's history with a skill before an attempt, as the logistic knowledge tracing framework (LKT)
 * defines it, the time-window counts of DAS3H, and the exposure features of instructional factors analysis. Counts
 * and log counts take no parameter; the decayed features take a decay d from 0 to 1, by which each earlier attempt
 * weighs d times the next; the others take the parameter that each documents. Time is in the units of the events'
 * `time`. The exposure features read the events without an outcome at the skill, and the others the events with one.
 */
export type LogisticFeature =
  | "opportunities"
  | "successes"
  | "failures"
  | "logOpportunities"
  | "logSuccesses"
  | "logFailures"
  | "exposures"
  | "proportion"
  | "netSuccesses"
  | {
      /** Opportunities decayed by d: Σ dᵃ over the attempts, a the number of later attempts. */
      readonly decayedOpportunities: number;
    }
  | {
      /** Successes decayed by d: Σ dᵃ y. */
      readonly decayedSuccesses: number;
    }
  | {
      /** Failures decayed by d: Σ dᵃ (1 - y). */
      readonly decayedFailures: number;
    }
  | {
      /** The proportion of successes decayed by d, with a prior of one success and one failure. */
      readonly decayedProportion: number;
    }
  | {
      /** The proportion of successes decayed by d, Σ dᵃ y / Σ dᵃ, with three failed attempts before the first. */
      readonly decayedProportionFailed: number;
    }
  | {
      /** The logit of the decayed proportion of successes with decay d. */
      readonly decayedLogit: number;
    }
  | {
      /** The logit ln((g + s) / (g + f)) of the successes s and failures f, with a prior of g of each; g above 0. */
      readonly logit: number;
    }
  | {
      /** nᵉ for the attempts n so far, with an exponent e above 0. */
      readonly powerOpportunities: number;
    }
  | {
      /** Δt⁻ᵉ with exponent e, Δt the time since the last attempt; 0 at the first attempt. */
      readonly recency: number;
    }
  | {
      /** y Δt⁻ᵉ, y the outcome of the last attempt and Δt the time since it; 0 at the first attempt. */
      readonly recencySuccess: number;
    }
  | {
      /** (1 - y) Δt⁻ᵉ, y the outcome of the last attempt and Δt the time since it; 0 at the first attempt. */
      readonly recencyFailure: number;
    }
  | {
      /** ln(1 + n) A⁻ᵈ with decay d, n the attempts and A the time since the first attempt; 0 at an age of 0. */
      readonly base: number;
    }
  | {
      /** As `base` with the successes s in place of n: ln(1 + s) A⁻ᵈ. */
      readonly baseSuccess: number;
    }
  | {
      /** As `base` with the failures f in place of n: ln(1 + f) A⁻ᵈ. */
      readonly baseFailure: number;
    }
  | {
      /**
       * As `base`, with the age A that compresses the time away from practice: the practice time since the first
       * attempt plus `scale` times the rest of the time since it. Every event needs a `duration`.
       */
      readonly basePractice: PracticeAge;
    }
  | {
      /** As `basePractice` with the successes in place of the attempts. */
      readonly basePracticeSuccess: PracticeAge;
    }
  | {
      /** As `basePractice` with the failures in place of the attempts. */
      readonly basePracticeFailure: PracticeAge;
    }
  | {
      /** ln(1 + x), x the attempts of a leaky count: each attempt adds 1, and x decays by exp(-Δt / τ) in time; τ above 0. */
      readonly dash: number;
    }
  | {
      /** As `dash` for the successes: each attempt adds its outcome. */
      readonly dashSuccess: number;
    }
  | {
      /**
       * The Performance Prediction Equation of Walsh et al.: n^power T^-(decay + spacing S), T the mean age of the
       * attempts weighted by their age^-weight and S the mean of 1 / ln(gap + e) over the gaps between attempts. It reads
       * the time of every attempt, so the state grows with them.
       */
      readonly ppe: PerformancePrediction;
    }
  | {
      /** ln(1 + the attempts in the last w units of time, the current moment included); w above 0, or `Infinity`. */
      readonly windowOpportunities: number;
    }
  | {
      /** ln(1 + the successes in the last w units of time): Σ y over the attempts in the window. */
      readonly windowSuccesses: number;
    }
  | {
      /** ln(1 + the failures in the last w units of time): Σ (1 - y) over the attempts in the window. */
      readonly windowFailures: number;
    }
  | {
      /** Exposures decayed by d: Σ dᵃ over the exposures, a the number of later exposures. */
      readonly decayedExposures: number;
    }
  | {
      /** Δt⁻ᵉ with exponent e, Δt the time since the latest exposure; 0 before any exposure. */
      readonly recencyExposure: number;
    };

/** The parameters of the age of the practice features. */
export interface PracticeAge {
  /** The decay d of A⁻ᵈ, at least 0. */
  readonly decay: number;
  /** The weight of the time away from practice, at least 0: 1 counts it as practice, 0 not at all. */
  readonly scale: number;
}

/** The parameters of the Performance Prediction Equation, as LKT's `par1` to `par4`. */
export interface PerformancePrediction {
  /** The exponent of the attempts, at least 0. */
  readonly power: number;
  /** The decay at no spacing, at least 0. */
  readonly decay: number;
  /** The change of the decay with the spacing, at least 0. */
  readonly spacing: number;
  /** The exponent of the weights of the attempts by their age, at least 0. */
  readonly weight: number;
}

const COUNTS = [
  "opportunities",
  "successes",
  "failures",
  "logOpportunities",
  "logSuccesses",
  "logFailures",
  "exposures",
  "proportion",
  "netSuccesses",
] as const;
const DECAYED = [
  "decayedOpportunities",
  "decayedSuccesses",
  "decayedFailures",
  "decayedProportion",
  "decayedProportionFailed",
  "decayedLogit",
  "decayedExposures",
] as const;
const RECENCIES = ["recency", "recencyExposure", "recencySuccess", "recencyFailure"] as const;
const BASES = ["base", "baseSuccess", "baseFailure"] as const;
const POSITIVES = ["logit", "powerOpportunities", "dash", "dashSuccess"] as const;
const WINDOWS = ["windowOpportunities", "windowSuccesses", "windowFailures"] as const;
/** The features with several parameters, and the names of the parameters in order. */
const SHAPES = {
  basePractice: ["decay", "scale"],
  basePracticeSuccess: ["decay", "scale"],
  basePracticeFailure: ["decay", "scale"],
  ppe: ["power", "decay", "spacing", "weight"],
} as const;

type Shaped = keyof typeof SHAPES;
type Kind =
  | (typeof COUNTS)[number]
  | (typeof DECAYED)[number]
  | (typeof RECENCIES)[number]
  | (typeof BASES)[number]
  | (typeof POSITIVES)[number]
  | (typeof WINDOWS)[number]
  | Shaped;

/** A checked feature: its kind, its parameters, and its label, the key of its coefficients. */
export interface FeatureSpec {
  readonly kind: Kind;
  /** The parameters in order: the decay, the exponent or the window; empty for the counts. */
  readonly values: readonly number[];
  /** `opportunities`, or with the parameters, `decayedLogit(0.8)` and `ppe(0.1, 0.5, 0.2, 1)`. */
  readonly label: string;
}

/** The features that read the events without an outcome at a skill, which only skill terms can have. */
export function readsExposures(spec: FeatureSpec): boolean {
  return EXPOSURE_KINDS.includes(spec.kind);
}

/** Returns the checked feature, rejecting an unknown one and parameters out of range. */
export function checkFeature(feature: LogisticFeature): FeatureSpec {
  if (typeof feature === "string") {
    if (!isIn(COUNTS, feature)) throw unknown(feature);
    return { kind: feature, values: [], label: feature };
  }
  const entries = isRecord(feature) ? Object.entries(feature) : [];
  const [kind, parameter] = entries[0] ?? [];
  if (entries.length !== 1 || kind === undefined) throw unknown(feature);
  const values = valuesOf(kind, parameter, feature);
  return { kind: kind as Kind, values, label: `${kind}(${values.join(", ")})` };
}

/** Returns the checked parameters of a feature of `kind`, in order. */
function valuesOf(kind: string, parameter: unknown, feature: LogisticFeature): number[] {
  if (isIn(DECAYED, kind)) {
    if (!(typeof parameter === "number" && parameter > 0 && parameter <= 1)) {
      throw new DataError(`The decay of ${kind} must be a number above 0 and at most 1, found ${describe(parameter)}.`);
    }
    return [parameter];
  }
  if (isIn(RECENCIES, kind)) checkNonNegative(parameter, `The exponent of ${kind}`);
  else if (isIn(BASES, kind)) checkNonNegative(parameter, `The decay of ${kind}`);
  else if (isIn(POSITIVES, kind)) checkPositive(parameter, `The parameter of ${kind}`);
  else if (isIn(WINDOWS, kind)) {
    if (!(typeof parameter === "number" && parameter > 0)) {
      throw new DataError(`The window of ${kind} must be a number above 0, or Infinity, found ${describe(parameter)}.`);
    }
  } else if (kind in SHAPES) return shapeValues(kind as Shaped, parameter);
  else throw unknown(feature);
  return [parameter as number];
}

/** Returns the fields of a feature with several parameters, in order, each at least 0. */
function shapeValues(kind: Shaped, parameter: unknown): number[] {
  const fields: readonly string[] = SHAPES[kind];
  if (!isRecord(parameter)) {
    throw new DataError(`The feature ${kind} needs an object with ${fields.join(", ")}, found ${describe(parameter)}.`);
  }
  for (const key of Object.keys(parameter)) {
    if (!fields.includes(key)) {
      throw new DataError(
        `The feature ${kind} has no field ${JSON.stringify(key)}; its fields are ${fields.join(", ")}.`,
      );
    }
  }
  return fields.map((field) => {
    const value = parameter[field];
    checkNonNegative(value, `The ${field} of ${kind}`);
    return value;
  });
}

function isIn<T extends string>(kinds: readonly T[], kind: string): kind is T {
  return (kinds as readonly string[]).includes(kind);
}

function unknown(feature: unknown): DataError {
  return new DataError(`Unknown feature ${JSON.stringify(feature)}; see LogisticFeature for the features.`);
}

const EXPOSURE_KINDS: readonly Kind[] = ["exposures", "decayedExposures", "recencyExposure"];
const PRACTICE_KINDS: readonly Kind[] = ["basePractice", "basePracticeSuccess", "basePracticeFailure"];

/** Returns what a state must track for the features, which it tracks only then. */
export function needsOf(specs: readonly FeatureSpec[]): HistoryNeeds {
  const of = (kinds: readonly Kind[]): readonly FeatureSpec[] => specs.filter((spec) => kinds.includes(spec.kind));
  const distinct = (chosen: readonly FeatureSpec[]): number[] => [
    ...new Set(chosen.map((spec) => spec.values[0] ?? 0)),
  ];
  const windows = specs.filter((spec) => isIn(WINDOWS, spec.kind));
  const unbounded =
    specs.some((spec) => spec.kind === "ppe") || windows.some((spec) => spec.values[0] === Number.POSITIVE_INFINITY);
  return {
    decays: distinct(specs.filter((spec) => isIn(DECAYED, spec.kind) && spec.kind !== "decayedExposures")),
    exposureDecays: distinct(of(["decayedExposures"])),
    exposures: of(EXPOSURE_KINDS).length > 0,
    previous: of(["recencySuccess", "recencyFailure"]).length > 0,
    first: specs.some((spec) => isIn(BASES, spec.kind)) || of(PRACTICE_KINDS).length > 0,
    practice: of(PRACTICE_KINDS).length > 0,
    leaks: distinct(of(["dash", "dashSuccess"])),
    times: unbounded || windows.length > 0,
    horizon: unbounded ? Number.POSITIVE_INFINITY : Math.max(0, ...windows.map((spec) => spec.values[0] ?? 0)),
  };
}

/** The decayed counts before any attempt. */
const NO_COUNTS = { successes: 0, failures: 0 };

/** What a feature reads: the learner's history and exposures at a skill, the time of the attempt and the practice so far. */
interface Reading {
  readonly history: SkillHistory | undefined;
  readonly exposure: ExposureHistory | undefined;
  readonly time: number;
  readonly practice: number;
  readonly successes: number;
  readonly failures: number;
  readonly attempts: number;
}

type Reader = (spec: FeatureSpec, reading: Reading) => number;

/** Returns the decay, exponent or window of a feature. */
const rate = (spec: FeatureSpec): number => spec.values[0] ?? 0;

const decayedSums = (spec: FeatureSpec, { history }: Reading): { successes: number; failures: number } =>
  (history === undefined ? undefined : entry(history.decayed, String(rate(spec)))) ?? NO_COUNTS;

/** The reader of a decayed feature, from the decayed counts and the prior of one success and one failure. */
const decayedReader =
  (value: (decay: number, attempts: number, successes: number, failures: number) => number): Reader =>
  (spec, reading) => {
    const { successes, failures } = decayedSums(spec, reading);
    return value(rate(spec), reading.attempts, successes, failures);
  };

/** Returns the time of the first attempt to now, or 0 before any. */
const age = ({ history, time }: Reading): number => (history?.first === undefined ? 0 : time - history.first);

/** Returns ln(1 + count) A⁻ᵈ, 0 at an age of 0 or less, where LKT gives infinity. */
const base = (count: number, ageNow: number, decay: number): number =>
  ageNow > 0 ? Math.log(1 + count) * ageNow ** -decay : 0;

/** Returns the age that compresses the time away from practice by `scale`, from the first attempt to now. */
function practiceAge(spec: FeatureSpec, reading: Reading): number {
  const { history, practice } = reading;
  const practiced = history?.firstPractice === undefined ? 0 : practice - history.firstPractice;
  return (age(reading) - practiced) * (spec.values[1] ?? 0) + practiced;
}

const basePractice =
  (count: (reading: Reading) => number): Reader =>
  (spec, reading) =>
    base(count(reading), practiceAge(spec, reading), rate(spec));

/** Returns the outcome of the last attempt and the recency Δt⁻ᵉ from it, or 0 before any. */
const lastOutcome = ({ history }: Reading): number => history?.previous ?? 0;

const recencyOf = (last: number | undefined, time: number, exponent: number): number => {
  const elapsed = last === undefined ? 0 : time - last;
  return elapsed === 0 ? 0 : elapsed ** -exponent;
};

/** Returns ln(1 + x), x the leaky count of attempts, or of successes, at the time of the attempt. */
const leakyReader =
  (pick: "attempts" | "successes"): Reader =>
  (spec, { history, time }) =>
    Math.log(1 + (history === undefined ? 0 : leakyAt(history, rate(spec), time)[pick]));

/** Returns ln(1 + the attempts, successes or failures in the last w units of time). */
const windowReader =
  (count: (outcome: number) => number): Reader =>
  (spec, { history, time }) => {
    const { times = [], outcomes = [] } = history ?? {};
    let sum = 0;
    for (let i = 0; i < times.length; i++) if (time - (times[i] ?? 0) <= rate(spec)) sum += count(outcomes[i] ?? 0);
    return Math.log(1 + sum);
  };

/**
 * The geometric sum 1 + d + ... + dᵐ⁻¹ of m terms, whose division of the decayed successes gives LKT's propdec2: the
 * attempts and three failed attempts before them.
 */
const geometric = (decay: number, terms: number): number => (decay === 1 ? terms : (1 - decay ** terms) / (1 - decay));

const READERS: Record<Kind, Reader> = {
  opportunities: (_, r) => r.attempts,
  successes: (_, r) => r.successes,
  failures: (_, r) => r.failures,
  logOpportunities: (_, r) => Math.log(1 + r.attempts),
  logSuccesses: (_, r) => Math.log(1 + r.successes),
  logFailures: (_, r) => Math.log(1 + r.failures),
  exposures: (_, r) => r.exposure?.count ?? 0,
  proportion: (_, r) => (r.attempts === 0 ? 0.5 : r.successes / r.attempts),
  netSuccesses: (_, r) => r.successes - r.failures,
  powerOpportunities: (spec, r) => r.attempts ** rate(spec),
  logit: (spec, r) => Math.log((rate(spec) + r.successes) / (rate(spec) + r.failures)),
  decayedOpportunities: decayedReader((_, __, successes, failures) => successes + failures),
  decayedSuccesses: decayedReader((_, __, successes) => successes),
  decayedFailures: decayedReader((_, __, ___, failures) => failures),
  decayedProportion: decayedReader((decay, attempts, successes, failures) => {
    const prior = decay ** attempts;
    return (prior + successes) / (prior + successes + (prior + failures));
  }),
  decayedProportionFailed: decayedReader((decay, attempts, successes) => successes / geometric(decay, attempts + 3)),
  decayedLogit: decayedReader(decayedLogit),
  decayedExposures: (spec, r) =>
    (r.exposure === undefined ? undefined : entry(r.exposure.decayed, String(rate(spec)))) ?? 0,
  recency: (spec, r) => recencyOf(r.history?.last, r.time, rate(spec)),
  recencyExposure: (spec, r) => recencyOf(r.exposure?.last, r.time, rate(spec)),
  recencySuccess: (spec, r) => lastOutcome(r) * recencyOf(r.history?.last, r.time, rate(spec)),
  recencyFailure: (spec, r) =>
    r.history === undefined ? 0 : (1 - lastOutcome(r)) * recencyOf(r.history.last, r.time, rate(spec)),
  base: (spec, r) => base(r.attempts, age(r), rate(spec)),
  baseSuccess: (spec, r) => base(r.successes, age(r), rate(spec)),
  baseFailure: (spec, r) => base(r.failures, age(r), rate(spec)),
  basePractice: basePractice((r) => r.attempts),
  basePracticeSuccess: basePractice((r) => r.successes),
  basePracticeFailure: basePractice((r) => r.failures),
  dash: leakyReader("attempts"),
  dashSuccess: leakyReader("successes"),
  ppe: (spec, r) => performancePrediction(r.history?.times ?? [], r.time, spec.values),
  windowOpportunities: windowReader(() => 1),
  windowSuccesses: windowReader((outcome) => outcome),
  windowFailures: windowReader((outcome) => 1 - outcome),
};

/**
 * Returns a feature's value for an attempt at `time`, from the learner's history and exposures with the skill before
 * it, and the practice time so far; with none, that of a skill not practiced. They hold what the model's features
 * need, as its state check and its fit ensure.
 */
export function featureValue(
  spec: FeatureSpec,
  history: SkillHistory | undefined,
  exposure: ExposureHistory | undefined,
  time: number,
  practice: number,
): number {
  const successes = history?.successes ?? 0;
  const failures = history?.failures ?? 0;
  return READERS[spec.kind](spec, {
    history,
    exposure,
    time,
    practice,
    successes,
    failures,
    attempts: successes + failures,
  });
}

/**
 * Returns the logit of the decayed proportion with a prior of one success and one failure decayed over every attempt:
 * each weighs dⁿ after n attempts.
 */
function decayedLogit(decay: number, attempts: number, successes: number, failures: number): number {
  const prior = decay ** attempts;
  const ratio = (prior + successes) / (prior + failures);
  if (ratio > 0 && ratio < Number.POSITIVE_INFINITY) return Math.log(ratio);
  // After a long run of one outcome, dⁿ underflows and the other count is 0: ln(dⁿ + 0) is n ln d, still finite.
  const logWeight = (count: number): number =>
    prior + count > 0 ? Math.log(prior + count) : attempts * Math.log(decay);
  return logWeight(successes) - logWeight(failures);
}

/**
 * Returns the Performance Prediction Equation at `time` from the times of the attempts so far: n^power T^-(decay + spacing
 * S). Where an earlier attempt is at the time of the attempt, so that its weight is infinite, the weighted age T is 1, as
 * LKT's NaN falls back to.
 */
function performancePrediction(
  times: readonly number[],
  time: number,
  [power = 0, decay = 0, spacing = 0, weight = 0]: readonly number[],
): number {
  const n = times.length;
  const ages = times.map((earlier) => time - earlier);
  const weights = ages.map((a) => a ** -weight);
  const total = weights.reduce((sum, w) => sum + w, 0);
  const mean = weights.reduce((sum, w, i) => sum + (w * (ages[i] ?? 0)) / total, 0);
  const weighted = n > 0 && Number.isFinite(total) && mean > 0 && Number.isFinite(mean) ? mean : 1;
  let spaced = 0;
  for (let k = 1; k < n; k++) {
    const gap = (times[k] ?? 0) - (times[k - 1] ?? 0);
    spaced += gap === 0 ? 0 : 1 / Math.log(gap + Math.E);
  }
  const space = n <= 1 ? 0 : spaced / (n - 1);
  return n ** power * weighted ** -(decay + spacing * space);
}
