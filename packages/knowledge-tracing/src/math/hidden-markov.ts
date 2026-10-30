/**
 * One observed sequence of a hidden Markov model with K states, whose transitions and emissions may change from step
 * to step, as those of knowledge tracing models do with the item practiced. Matrices are stored by rows.
 */
export interface HiddenMarkovSequence {
  /** K, the number of hidden states. */
  readonly stateCount: number;
  /** P(state i at step 0): K entries. */
  readonly initial: Float64Array;
  /**
   * P(state j at step t + 1 | state i at step t) for t from 0 to T - 2: T - 1 matrices of K × K, by step, then i, then
   * j.
   */
  readonly transitions: Float64Array;
  /** P(observation at step t | state i at step t): T × K, by step, then i. Every step has a positive probability. */
  readonly emissions: Float64Array;
}

/** The posterior quantities of a sequence, by the scaled forward-backward algorithm. */
export interface ForwardBackward {
  /** ln P(observations). */
  readonly logLikelihood: number;
  /** P(state i at step t | all observations): T × K, by step, then i. */
  readonly posteriors: Float64Array;
  /** P(state i at step t, state j at step t + 1 | all observations): T - 1 matrices of K × K, as `transitions`. */
  readonly transitionPosteriors: Float64Array;
  /**
   * P(state i at step t | the observations before step t): T × K, by step, then i. These are the predictions of online
   * inference, made before each observation.
   */
  readonly predictions: Float64Array;
}

/**
 * Runs the forward-backward algorithm on one sequence: the forward pass normalizes the forward variables at each step,
 * with the log-likelihood the sum of the logs of the normalizers, and the backward pass smooths the normalized ones, so
 * that long sequences neither underflow nor overflow.
 *
 * @see {@link https://doi.org/10.1109/5.18626 | Rabiner (1989)}
 */
export function forwardBackward(sequence: HiddenMarkovSequence): ForwardBackward {
  const filtered = forward(sequence);
  return { logLikelihood: filtered.logLikelihood, ...backward(sequence, filtered), predictions: filtered.predictions };
}

/** The forward pass: the predictions and the normalized forward variables. */
interface Filtered {
  readonly predictions: Float64Array;
  /** P(state i at step t | the observations to step t). */
  readonly forward: Float64Array;
  /** Σ over steps of ln P(observation at step t | the observations before step t). */
  readonly logLikelihood: number;
}

function forward(sequence: HiddenMarkovSequence): Filtered {
  const { stateCount: k, initial, transitions, emissions } = sequence;
  const steps = emissions.length / k;
  const predictions = new Float64Array(steps * k);
  const filtered = new Float64Array(steps * k);
  let logLikelihood = 0;
  predictions.set(initial);
  for (let t = 0; t < steps; t++) {
    if (t > 0) propagate(filtered, (t - 1) * k, transitions, (t - 1) * k * k, predictions, t * k, k);
    let scale = 0;
    for (let j = 0; j < k; j++) {
      const joint = (predictions[t * k + j] ?? 0) * (emissions[t * k + j] ?? 0);
      filtered[t * k + j] = joint;
      scale += joint;
    }
    for (let j = 0; j < k; j++) filtered[t * k + j] = (filtered[t * k + j] ?? 0) / scale;
    logLikelihood += Math.log(scale);
  }
  return { predictions, forward: filtered, logLikelihood };
}

/**
 * Writes the state distribution after one transition, out(j) = Σ_i from(i) A(i, j), for the K states starting at the
 * offsets given.
 */
function propagate(
  from: Float64Array,
  fromOffset: number,
  transitions: Float64Array,
  transitionOffset: number,
  out: Float64Array,
  outOffset: number,
  k: number,
): void {
  for (let j = 0; j < k; j++) {
    let sum = 0;
    for (let i = 0; i < k; i++) sum += (from[fromOffset + i] ?? 0) * (transitions[transitionOffset + i * k + j] ?? 0);
    out[outOffset + j] = sum;
  }
}

/**
 * The backward pass, by smoothing the filtered probabilities: γ_(T-1) = α_(T-1), and for earlier steps
 * ξ_t(i, j) = α_t(i) A_t(i, j) / p_(t+1)(j) · γ_(t+1)(j) and γ_t(i) = Σ_j ξ_t(i, j), with p_(t+1)(j) = Σ_i
 * α_t(i) A_t(i, j) the forward pass's prediction. Every factor is a probability or a share of one, so it
 * neither overflows nor gives `NaN` for a sequence of thousands of steps, in which a state's filtered
 * probability underflows to 0.
 */
function backward(
  sequence: HiddenMarkovSequence,
  filtered: Filtered,
): Pick<ForwardBackward, "posteriors" | "transitionPosteriors"> {
  const { stateCount: k, transitions } = sequence;
  const { forward, predictions } = filtered;
  const steps = forward.length / k;
  const posteriors = new Float64Array(steps * k);
  const transitionPosteriors = new Float64Array(Math.max(steps - 1, 0) * k * k);
  posteriors.set(forward.subarray((steps - 1) * k), (steps - 1) * k);
  for (let t = steps - 2; t >= 0; t--) {
    for (let j = 0; j < k; j++) {
      const predicted = predictions[(t + 1) * k + j] ?? 0;
      // No probability reaches a state predicted at 0, so its pairs keep their 0.
      if (predicted === 0) continue;
      const smoothed = posteriors[(t + 1) * k + j] ?? 0;
      for (let i = 0; i < k; i++) {
        const pair = (((forward[t * k + i] ?? 0) * (transitions[t * k * k + i * k + j] ?? 0)) / predicted) * smoothed;
        transitionPosteriors[t * k * k + i * k + j] = pair;
        posteriors[t * k + i] = (posteriors[t * k + i] ?? 0) + pair;
      }
    }
  }
  return { posteriors, transitionPosteriors };
}
