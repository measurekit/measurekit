import { checkRandom, checkUnitInterval, type Random } from "@measurekit/core";
import { type Attempt, checkAttempts } from "../data/event.js";
import { type EventLog, eventLog } from "../data/event-log.js";
import { checkModel, type LearnerModel } from "../model.js";

/** A simulated log, with each learner's state after it. */
export interface Simulation<TState> {
  /** The attempts with their simulated outcomes, in time order. */
  readonly log: EventLog;
  /** Each learner's state after their events, by learner id. */
  readonly states: Readonly<Record<string, TState>>;
}

/**
 * Simulates the outcomes of attempts under a model.
 *
 * @remarks
 * In time order, each attempt is correct with the probability the model predicts from the learner's state, and the
 * state is updated with the drawn outcome. For a model whose prediction is the probability of a correct answer given
 * the learner's earlier outcomes, as BKT, logistic models, IRT and KST, this draws from the model itself: each
 * learner's latent knowledge is drawn along with the outcomes. Outcomes are 0 or 1.
 *
 * @throws `DataError` when the model is not a learner model, the attempts are not a non-empty array of attempts, the
 * model rejects one, or a prediction is not from 0 to 1.
 */
export function simulate<TState>(
  model: LearnerModel<TState>,
  attempts: readonly Attempt[],
  random: Random,
): Simulation<TState> {
  checkModel(model, "A simulation needs a learner model, such as bkt()");
  checkAttempts(attempts, "Attempt");
  checkRandom(random);
  const ordered = [...attempts].sort((x, y) => x.time - y.time);
  const states = new Map<string, TState>();
  const events = ordered.map((attempt) => {
    const state = states.get(attempt.learner) ?? model.initialState(attempt.learner);
    const p = model.predict(state, attempt);
    checkUnitInterval(p, `The model's prediction for "${attempt.learner}" at "${attempt.item}"`);
    const event = { ...attempt, outcome: random.uniform() < p ? 1 : 0 };
    states.set(attempt.learner, model.update(state, event));
    return event;
  });
  return { log: eventLog(events), states: Object.fromEntries(states) };
}
