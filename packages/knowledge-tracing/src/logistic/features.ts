import { checkArray, checkNonNegative, DataError, describe, isRecord } from "@measurekit/core";
import { entry } from "../data/record.js";
import type { ExposureHistory, SkillHistory } from "./state.js";

/**
 * A feature of a learner's history with a skill before an attempt, as the logistic knowledge tracing framework (LKT)
 * defines it, and the exposure features of instructional factors analysis. Counts and log counts take no parameter;
 * the decayed features take a decay d from 0 to 1, by which each earlier attempt weighs d times the next; recency takes
 * an exponent. The exposure features read the events without an outcome at the skill, and the others the events with
 * one.
 */
export type LogisticFeature =
  | "opportunities"
  | "successes"
  | "failures"
  | "logOpportunities"
  | "logSuccesses"
  | "logFailures"
  | "exposures"
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
      /** The logit of the decayed proportion of successes with decay d. */
      readonly decayedLogit: number;
    }
  | {
      /** Δt⁻ᵉ with exponent e, Δt the time since the last attempt; 0 at the first attempt. */
      readonly recency: number;
    }
  | {
      /** Exposures decayed by d: Σ dᵃ over the exposures, a the number of later exposures. */
      readonly decayedExposures: number;
    }
  | {
      /** Δt⁻ᵉ with exponent e, Δt the time since the latest exposure; 0 before any exposure. */
      readonly recencyExposure: number;
    };

const COUNTS = [
  "opportunities",
  "successes",
  "failures",
  "logOpportunities",
  "logSuccesses",
  "logFailures",
  "exposures",
] as const;
const DECAYED = [
  "decayedOpportunities",
  "decayedSuccesses",
  "decayedFailures",
  "decayedProportion",
  "decayedLogit",
  "decayedExposures",
] as const;
const RECENCIES = ["recency", "recencyExposure"] as const;

type Kind = (typeof COUNTS)[number] | (typeof DECAYED)[number] | (typeof RECENCIES)[number];

/** A checked feature: its kind, its parameter, and its label, the key of its coefficients. */
export interface FeatureSpec {
  readonly kind: Kind;
  /** The decay, or the exponent of recency; 0 for the counts. */
  readonly rate: number;
  /** `opportunities`, or with the parameter, `decayedLogit(0.8)`. */
  readonly label: string;
}

/** Returns the checked features, rejecting unknown ones, parameters out of range and repeated features. */
export function checkFeatures(features: readonly LogisticFeature[]): FeatureSpec[] {
  checkArray(features, "The option features must be an array");
  const specs = features.map(toSpec);
  const labels = new Set<string>();
  for (const { label } of specs) {
    if (labels.has(label)) throw new DataError(`The feature ${label} is listed twice.`);
    labels.add(label);
  }
  return specs;
}

function toSpec(feature: LogisticFeature): FeatureSpec {
  if (typeof feature === "string") {
    if (!isCount(feature)) throw unknown(feature);
    return { kind: feature, rate: 0, label: feature };
  }
  const entries = isRecord(feature) ? Object.entries(feature) : [];
  const [kind, rate] = entries[0] ?? [];
  if (entries.length !== 1 || kind === undefined) throw unknown(feature);
  if (isRecency(kind)) {
    checkNonNegative(rate, `The exponent of ${kind}`);
    return { kind, rate, label: `${kind}(${rate})` };
  }
  if (!isDecayed(kind)) throw unknown(feature);
  if (!(typeof rate === "number" && rate > 0 && rate <= 1)) {
    throw new DataError(`The decay of ${kind} must be a number above 0 and at most 1, found ${describe(rate)}.`);
  }
  return { kind, rate, label: `${kind}(${rate})` };
}

function isDecayed(kind: string): kind is (typeof DECAYED)[number] {
  return (DECAYED as readonly string[]).includes(kind);
}

function isRecency(kind: string): kind is (typeof RECENCIES)[number] {
  return (RECENCIES as readonly string[]).includes(kind);
}

function isCount(kind: string): kind is (typeof COUNTS)[number] {
  return (COUNTS as readonly string[]).includes(kind);
}

function unknown(feature: unknown): DataError {
  return new DataError(`Unknown feature ${JSON.stringify(feature)}; see LogisticFeature for the features.`);
}

/** Returns whether any feature reads the exposures, which a state tracks only then. */
export function usesExposures(specs: readonly FeatureSpec[]): boolean {
  return specs.some(
    (spec) => spec.kind === "exposures" || spec.kind === "decayedExposures" || spec.kind === "recencyExposure",
  );
}

/** Returns the decays the features of outcomes need running sums for, distinct, in order. */
export function decaysOf(specs: readonly FeatureSpec[]): number[] {
  return decaysOfKinds(specs, (kind) => kind !== "decayedExposures");
}

/** Returns the decays the exposure features need running sums for, distinct, in order. */
export function exposureDecaysOf(specs: readonly FeatureSpec[]): number[] {
  return decaysOfKinds(specs, (kind) => kind === "decayedExposures");
}

function decaysOfKinds(specs: readonly FeatureSpec[], include: (kind: Kind) => boolean): number[] {
  const decayed = specs.filter((spec) => !isRecency(spec.kind) && spec.rate > 0 && include(spec.kind));
  return [...new Set(decayed.map((spec) => spec.rate))];
}

/** The decayed counts before any attempt. */
const NO_COUNTS = { successes: 0, failures: 0 };

/**
 * Returns a feature's value for an attempt at `time`, from the learner's history and exposures with the skill before
 * it; with none, that of a skill not practiced. They hold the running sums of the model's decays, as its state check and
 * its fit ensure.
 */
export function featureValue(
  spec: FeatureSpec,
  history: SkillHistory | undefined,
  exposure: ExposureHistory | undefined,
  time: number,
): number {
  const successes = history?.successes ?? 0;
  const failures = history?.failures ?? 0;
  const attempts = successes + failures;
  switch (spec.kind) {
    case "opportunities":
      return attempts;
    case "successes":
      return successes;
    case "failures":
      return failures;
    case "logOpportunities":
      return Math.log(1 + attempts);
    case "logSuccesses":
      return Math.log(1 + successes);
    case "logFailures":
      return Math.log(1 + failures);
    case "exposures":
      return exposure?.count ?? 0;
    case "decayedExposures":
      return (exposure === undefined ? undefined : entry(exposure.decayed, String(spec.rate))) ?? 0;
    case "recency":
      return recency(history?.last, time, spec.rate);
    case "recencyExposure":
      return recency(exposure?.last, time, spec.rate);
    default: {
      const sums = (history === undefined ? undefined : entry(history.decayed, String(spec.rate))) ?? NO_COUNTS;
      return decayed(spec.kind, spec.rate, attempts, sums.successes, sums.failures);
    }
  }
}

/** Returns Δt⁻ᵉ since `last`. LKT gives 0 before any attempt and at an attempt at the same time as the last. */
function recency(last: number | undefined, time: number, exponent: number): number {
  const elapsed = last === undefined ? 0 : time - last;
  return elapsed === 0 ? 0 : elapsed ** -exponent;
}

/**
 * Returns a decayed feature from the decayed counts of successes and failures and the prior, one success and one
 * failure decayed over every attempt: each weighs dⁿ after n attempts.
 */
function decayed(kind: Kind, decay: number, attempts: number, successes: number, failures: number): number {
  const prior = decay ** attempts;
  switch (kind) {
    case "decayedOpportunities":
      return successes + failures;
    case "decayedSuccesses":
      return successes;
    case "decayedFailures":
      return failures;
    case "decayedProportion":
      return (prior + successes) / (prior + successes + (prior + failures));
    default: {
      const ratio = (prior + successes) / (prior + failures);
      if (ratio > 0 && ratio < Number.POSITIVE_INFINITY) return Math.log(ratio);
      // After a long run of one outcome, dⁿ underflows and the other count is 0: ln(dⁿ + 0) is n ln d, still finite.
      const logWeight = (count: number): number =>
        prior + count > 0 ? Math.log(prior + count) : attempts * Math.log(decay);
      return logWeight(successes) - logWeight(failures);
    }
  }
}
