import { checkBoolean, checkRandom, checkUnitInterval, type Random } from "@measurekit/core";
import { type Attempt, buildEventLog, checkAttempts, type EventInput, type EventLog } from "../data/internal.js";
import { checkModel, type LearnerModel } from "../model.js";

/** An attempt to simulate: with `observe: false` it is a learning opportunity, whose outcome is not drawn. */
export interface SimulatedAttempt extends Attempt {
  /**
   * Whether an outcome is drawn; `false` gives an event with `outcome: null`.
   *
   * @defaultValue `true`
   */
  readonly observe?: boolean;
}

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
 * An attempt with `observe: false`, such as a lesson, draws nothing: it is predicted by no one, and the model updates
 * with an event without an outcome.
 *
 * @param model - The model whose predictions give the probability of a correct outcome.
 * @param attempts - The attempts, in the order the learners make them.
 * @param random - The generator of the outcomes.
 * @returns The log with simulated outcomes and each learner's final state.
 * @throws `DataError` when the model is not a learner model, the attempts are not a non-empty array of attempts, the
 * model rejects one, or a prediction is not from 0 to 1.
 *
 * @example
 * ```ts
 * import { createRandom } from "@measurekit/knowledge-tracing";
 * import { bkt } from "@measurekit/knowledge-tracing/bkt";
 * import { simulate } from "@measurekit/knowledge-tracing/sim";
 *
 * const truth = { fractions: { prior: 0.3, learn: 0.15, forget: 0, guess: 0.2, slip: 0.1 } };
 * const attempts = Array.from({ length: 4 }, (_, learner) =>
 *   Array.from({ length: 3 }, (_, k) => ({
 *     learner: `l${learner}`,
 *     item: `q${k}`,
 *     skills: ["fractions"],
 *     time: learner * 10 + k,
 *   })),
 * ).flat();
 * const { log } = simulate(bkt({ parameters: { skills: truth } }), attempts, createRandom(42));
 * log.length; // 12
 * log.map(({ outcome }) => outcome); // [0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1]
 * ```
 */
export function simulate<TState>(
  model: LearnerModel<TState>,
  attempts: readonly SimulatedAttempt[],
  random: Random,
): Simulation<TState> {
  checkModel(model, "A simulation needs a learner model, such as bkt()");
  checkAttempts(attempts, "Attempt");
  attempts.forEach(({ observe }, index) => {
    if (observe !== undefined) checkBoolean(observe, `The observe of attempt ${index + 1}`);
  });
  checkRandom(random);
  const ordered = [...attempts].sort((x, y) => x.time - y.time);
  const states = new Map<string, TState>();
  const events = ordered.map(({ observe = true, ...attempt }): EventInput => {
    const state = states.get(attempt.learner) ?? model.initialState(attempt.learner);
    let outcome: number | null = null;
    if (observe) {
      const p = model.predict(state, attempt);
      checkUnitInterval(p, `The model's prediction for "${attempt.learner}" at "${attempt.item}"`);
      outcome = random.uniform() < p ? 1 : 0;
    }
    const event = { ...attempt, outcome };
    states.set(attempt.learner, model.update(state, event));
    return event;
  });
  return { log: buildEventLog(events), states: Object.fromEntries(states) };
}
