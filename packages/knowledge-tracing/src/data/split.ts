import { checkIntegerFrom, checkRandom, DataError, type Random } from "@measurekit/core";
import { permutation } from "../math/permutation.js";
import type { Event } from "./event.js";
import { checkLog, type EventLog } from "./event-log.js";

/** A training log and a test log taken from one log, for evaluating a model on events it was not fitted to. */
export interface Fold {
  /** The events to fit the model to. */
  readonly train: EventLog;
  /** The events to predict. */
  readonly test: EventLog;
}

/**
 * Splits a log into `folds` folds by learner: each learner's events are in the test log of one fold and the training
 * logs of the others. Learners are assigned to folds at random, in equal numbers up to one.
 *
 * @throws `DataError` when the log is malformed, `folds` is not an integer of at least 2, or there are fewer learners
 * than folds.
 */
export function splitByLearner(log: EventLog, folds: number, random: Random): Fold[] {
  return splitByKey(log, folds, random, "learner", (event) => event.learner);
}

/**
 * Splits a log into `folds` folds by item, as {@link splitByLearner} does by learner: the test log of a fold holds the
 * events of items the model was not fitted to.
 *
 * @throws `DataError` when the log is malformed, `folds` is not an integer of at least 2, or there are fewer items than
 * folds.
 */
export function splitByItem(log: EventLog, folds: number, random: Random): Fold[] {
  return splitByKey(log, folds, random, "item", (event) => event.item);
}

/**
 * Splits a log by time into `folds + 1` consecutive blocks of about equal numbers of events, and returns `folds` folds:
 * fold k trains on blocks 1 to k and tests on block k + 1, so a model never sees events later than those it predicts.
 * A block boundary falls after the last event of its time, so events with equal times share a block.
 *
 * @throws `DataError` when the log is malformed, `folds` is not a positive integer, or the times do not divide into
 * `folds + 1` non-empty blocks.
 */
export function splitByTime(log: EventLog, folds: number): Fold[] {
  checkLog(log);
  checkFolds(folds, 1);
  const blocks = folds + 1;
  const bounds = [0];
  for (let block = 1; block < blocks; block++) {
    let bound = Math.max(Math.floor((block * log.length) / blocks), (bounds.at(-1) ?? 0) + 1);
    while (bound < log.length && log[bound]?.time === log[bound - 1]?.time) bound++;
    if (bound >= log.length) {
      throw new DataError(
        `The log's times do not divide into ${blocks} blocks for ${folds} folds: too few events have distinct times.`,
      );
    }
    bounds.push(bound);
  }
  bounds.push(log.length);
  return Array.from({ length: folds }, (_, fold) => ({
    train: log.slice(0, bounds[fold + 1]),
    test: log.slice(bounds[fold + 1], bounds[fold + 2]),
  }));
}

function splitByKey(log: EventLog, folds: number, random: Random, kind: string, key: (event: Event) => string): Fold[] {
  checkLog(log);
  checkRandom(random);
  checkFolds(folds, 2);
  const ids = [...new Set(log.map(key))];
  if (ids.length < folds) {
    throw new DataError(
      `Splitting by ${kind} into ${folds} folds needs at least ${folds} ${kind}s, found ${ids.length}.`,
    );
  }
  // Dealing the ids out in random order gives every fold the same number of them, up to one.
  const position = new Uint32Array(ids.length);
  permutation(ids.length, random).forEach((index, place) => {
    position[index] = place;
  });
  const foldOf = new Map(ids.map((id, index) => [id, (position[index] ?? 0) % folds]));
  return Array.from({ length: folds }, (_, fold) => ({
    train: log.filter((event) => foldOf.get(key(event)) !== fold),
    test: log.filter((event) => foldOf.get(key(event)) === fold),
  }));
}

function checkFolds(folds: number, minimum: number): void {
  checkIntegerFrom(folds, "The number of folds", minimum);
}
