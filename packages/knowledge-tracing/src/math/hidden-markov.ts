/**
 * The arrays of the forward-backward algorithm for hidden Markov models with K states, whose transitions and emissions
 * may change from step to step, as those of knowledge tracing models do with the item practiced. They are long enough
 * for the longest of several sequences, which a caller fills and reads one sequence at a time, so that the E-step of a
 * model allocates nothing for each sequence. A sequence of T steps takes the first entries of each. Matrices are stored
 * by rows.
 */
export interface HiddenMarkovBuffers {
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
  /**
   * P(state i at step t | the observations before step t): T × K, by step, then i. These are the predictions of online
   * inference, made before each observation.
   */
  readonly predictions: Float64Array;
  /** P(state i at step t | the observations to step t): T × K, by step, then i. */
  readonly filtered: Float64Array;
  /** P(state i at step t | all observations): T × K, by step, then i. */
  readonly posteriors: Float64Array;
  /** P(state i at step t, state j at step t + 1 | all observations): T - 1 matrices of K × K, as `transitions`. */
  readonly transitionPosteriors: Float64Array;
}

/** Returns the buffers of sequences of K states and at most `maxSteps` steps. */
export function createHiddenMarkovBuffers(stateCount: number, maxSteps: number): HiddenMarkovBuffers {
  const states = stateCount * maxSteps;
  const pairs = stateCount * stateCount * Math.max(maxSteps - 1, 0);
  return {
    stateCount,
    initial: new Float64Array(stateCount),
    transitions: new Float64Array(pairs),
    emissions: new Float64Array(states),
    predictions: new Float64Array(states),
    filtered: new Float64Array(states),
    posteriors: new Float64Array(states),
    transitionPosteriors: new Float64Array(pairs),
  };
}

/**
 * Runs the forward-backward algorithm on the first `steps` steps of the buffers' `initial`, `transitions` and
 * `emissions`, and writes the first entries of their `predictions`, `filtered`, `posteriors` and
 * `transitionPosteriors`: the forward pass normalizes the forward variables at each step, with the log-likelihood the
 * sum of the logs of the normalizers, and the backward pass smooths the normalized ones, so that long sequences neither
 * underflow nor overflow.
 *
 * @returns ln P(observations).
 * @see {@link https://doi.org/10.1109/5.18626 | Rabiner (1989)}
 */
export function smoothSequence(buffers: HiddenMarkovBuffers, steps: number): number {
  const logLikelihood = forward(buffers, steps);
  backward(buffers, steps);
  return logLikelihood;
}

/**
 * The forward pass: writes the predictions and the normalized forward variables, P(state i at step t | the
 * observations to step t), and returns Σ over steps of ln P(observation at step t | the observations before step t).
 */
function forward(buffers: HiddenMarkovBuffers, steps: number): number {
  const { stateCount: k, initial, transitions, emissions, predictions, filtered } = buffers;
  let logLikelihood = 0;
  for (let j = 0; j < k; j++) predictions[j] = initial[j] ?? 0;
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
  return logLikelihood;
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
function backward(buffers: HiddenMarkovBuffers, steps: number): void {
  const { stateCount: k, filtered, posteriors } = buffers;
  const last = (steps - 1) * k;
  for (let i = 0; i < k; i++) posteriors[last + i] = filtered[last + i] ?? 0;
  for (let t = steps - 2; t >= 0; t--) smoothStep(buffers, t);
}

/** Writes ξ_t and γ_t of one step from γ_(t+1). */
function smoothStep(buffers: HiddenMarkovBuffers, t: number): void {
  const { stateCount: k, transitions, filtered, predictions, posteriors, transitionPosteriors } = buffers;
  for (let i = 0; i < k; i++) posteriors[t * k + i] = 0;
  for (let j = 0; j < k; j++) {
    const predicted = predictions[(t + 1) * k + j] ?? 0;
    const smoothed = posteriors[(t + 1) * k + j] ?? 0;
    for (let i = 0; i < k; i++) {
      // No probability reaches a state predicted at 0, so its pairs are 0.
      const pair =
        predicted === 0
          ? 0
          : (((filtered[t * k + i] ?? 0) * (transitions[t * k * k + i * k + j] ?? 0)) / predicted) * smoothed;
      transitionPosteriors[t * k * k + i * k + j] = pair;
      posteriors[t * k + i] = (posteriors[t * k + i] ?? 0) + pair;
    }
  }
}
