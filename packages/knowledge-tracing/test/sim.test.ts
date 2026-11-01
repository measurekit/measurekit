import { describe, expect, it } from "vitest";
import {
  type Attempt,
  afm,
  bkt,
  createRandom,
  DataError,
  elo,
  type IrtItem,
  irt,
  type KstState,
  kst,
  type LearnerModel,
  logistic,
  type Random,
  type SimulatedAttempt,
  simulate,
  surmiseStructure,
} from "../src/index.js";
import { permutation } from "../src/math/permutation.js";
import { correlation, defined } from "./assertions.js";

// Recovery thresholds, fixed before the simulations were first run.
const BKT_PRIOR_THRESHOLD = 0.1;
const BKT_RATE_THRESHOLD = 0.05;
const AFM_INTERCEPT_THRESHOLD = 0.25;
const AFM_SLOPE_THRESHOLD = 0.05;
const PROFICIENCY_CORRELATION = 0.8;
const SD_THRESHOLD = 0.15;
const ELO_ABILITY_CORRELATION = 0.85;
const ELO_DIFFICULTY_CORRELATION = 0.95;
const IRT_CORRELATION = 0.9;
const KST_RATE_THRESHOLD = 0.05;
// The assessment's posterior is calibrated when the model is right: the share of learners whose most likely state is
// their true one matches the mean probability of that state, within 3 standard errors of a share.
const CALIBRATION_STANDARD_ERRORS = 3;

/** Returns an index drawn with the given probabilities. */
function draw(probabilities: readonly number[], random: Random): number {
  let u = random.uniform();
  for (let k = 0; k < probabilities.length - 1; k++) {
    u -= probabilities[k] ?? 0;
    if (u < 0) return k;
  }
  return probabilities.length - 1;
}

describe("simulate with attempts that are not observed", () => {
  const model = elo();
  const attempts: SimulatedAttempt[] = Array.from({ length: 12 }, (_, time) => ({
    learner: "ana",
    item: `q${time % 3}`,
    skills: [],
    time,
  }));

  it("gives an attempt with observe false no outcome, draws nothing for it, and updates the model with it", () => {
    const lessons = attempts.map((attempt) => (attempt.time % 4 === 0 ? { ...attempt, observe: false } : attempt));
    const { log, states } = simulate(model, lessons, createRandom(9));
    expect(log.map((event) => event.outcome === null)).toEqual(lessons.map((attempt) => attempt.observe === false));
    expect(log.every((event) => !("observe" in event))).toBe(true);
    // Elo ignores events without an outcome, so the others draw as they would without the lessons.
    const without = simulate(
      model,
      lessons.filter((attempt) => attempt.observe !== false),
      createRandom(9),
    );
    expect(log.filter((event) => event.outcome !== null).map((event) => event.outcome)).toEqual(
      without.log.map((event) => event.outcome),
    );
    expect(states).toEqual(without.states);
  });

  it("teaches the model with an unobserved attempt: BKT learns from a lesson before the first draw", () => {
    const learner = bkt({ parameters: { skills: { s: { prior: 0, learn: 1, forget: 0, guess: 0, slip: 0 } } } });
    const lesson = { learner: "ana", item: "video", skills: ["s"], time: 1, observe: false };
    const question = { learner: "ana", item: "q", skills: ["s"], time: 2 };
    expect(simulate(learner, [lesson, question], createRandom(1)).log.map((event) => event.outcome)).toEqual([null, 1]);
    expect(
      simulate(learner, [{ ...lesson, observe: true }, question], createRandom(1)).log.map((e) => e.outcome),
    ).toEqual([0, 1]);
  });

  it("rejects an observe that is not a boolean", () => {
    const attempt = JSON.parse('{"learner":"a","item":"q","skills":[],"time":1,"observe":"no"}');
    expect(() => simulate(model, [attempt], createRandom(1))).toThrow(
      'The observe of attempt 1 must be a boolean, found "no".',
    );
  });
});

describe("simulate", () => {
  it("reproduces with the seed and continues each learner's state", () => {
    const model = bkt({ parameters: { skills: { s: { prior: 0.3, learn: 0.2, forget: 0, guess: 0.2, slip: 0.1 } } } });
    const attempts: Attempt[] = Array.from({ length: 20 }, (_, time) => ({
      learner: time % 2 === 0 ? "ana" : "budi",
      item: "q",
      skills: ["s"],
      time: 20 - time,
    }));
    const first = simulate(model, attempts, createRandom(5));
    expect(simulate(model, attempts, createRandom(5))).toEqual(first);
    expect(first.log.map((event) => event.time)).toEqual(Array.from({ length: 20 }, (_, i) => i + 1));
    const ana = first.log.filter((event) => event.learner === "ana");
    expect(first.states.ana).toEqual(
      ana.reduce((state, event) => model.update(state, event), model.initialState("ana")),
    );
  });

  it("rejects malformed attempts and predictions that are not probabilities", () => {
    const model = bkt({ parameters: { skills: { s: { prior: 0.3, learn: 0.2, forget: 0, guess: 0.2, slip: 0.1 } } } });
    expect(() => simulate(model, [{ learner: "", item: "q", skills: ["s"], time: 0 }], createRandom(1))).toThrow(
      DataError,
    );
    const broken: LearnerModel<null, null> = {
      parameters: null,
      fit: () => {
        throw new Error("Not fitted.");
      },
      initialState: () => null,
      update: () => null,
      predict: () => 1.5,
    };
    expect(() => simulate(broken, [{ learner: "ana", item: "q", skills: [], time: 0 }], createRandom(1))).toThrow(
      'The model\'s prediction for "ana" at "q" must be a number from 0 to 1, found 1.5.',
    );
    expect(() => simulate(broken, JSON.parse("{}"), createRandom(1))).toThrow(
      "The attempts must be an array of attempts, found an object.",
    );
  });
});

describe("recovery from simulated learners", () => {
  const random = createRandom(20261029);

  it("recovers BKT's parameters", () => {
    const truth = {
      fractions: { prior: 0.3, learn: 0.15, forget: 0, guess: 0.2, slip: 0.1 },
      decimals: { prior: 0.5, learn: 0.25, forget: 0, guess: 0.15, slip: 0.08 },
    };
    const attempts = Array.from({ length: 800 }, (_, learner) =>
      Array.from({ length: 24 }, (_, k) => ({
        learner: `l${learner}`,
        item: `q${k % 6}`,
        skills: [k % 2 === 0 ? "fractions" : "decimals"],
        time: learner * 100 + k,
      })),
    ).flat();
    const { log } = simulate(bkt({ parameters: { skills: truth } }), attempts, random);
    const fit = bkt().fit(log);
    for (const [skill, expected] of Object.entries(truth)) {
      const found = defined(fit.model.parameters.skills[skill]);
      expect(Math.abs(found.prior - expected.prior), `${skill} prior`).toBeLessThanOrEqual(BKT_PRIOR_THRESHOLD);
      for (const rate of ["learn", "guess", "slip"] as const) {
        expect(Math.abs(found[rate] - expected[rate]), `${skill} ${rate}`).toBeLessThanOrEqual(BKT_RATE_THRESHOLD);
      }
    }
    expect(fit.logLikelihood).toBeGreaterThanOrEqual(
      bkt({ parameters: { skills: truth } }).fit(log).logLikelihood - 1e-9,
    );
  });

  it("recovers AFM's intercepts, slopes and proficiencies, and the spread of a random intercept", () => {
    const skills = { s1: [-0.5, 0.1], s2: [0, 0.2], s3: [0.3, 0.05], s4: [-1, 0.15] } as const;
    const thetas = Array.from({ length: 300 }, () => 0.8 * random.normal());
    const parameters = {
      skills: Object.fromEntries(
        Object.entries(skills).map(([skill, [intercept, slope]]) => [
          skill,
          { intercept, coefficients: { opportunities: slope } },
        ]),
      ),
      learners: Object.fromEntries(thetas.map((theta, l) => [`l${l}`, theta])),
    };
    const names = Object.keys(skills);
    const attempts = thetas.flatMap((_, l) =>
      Array.from({ length: 30 }, (_, k) => ({
        learner: `l${l}`,
        item: "q",
        skills: [names[Math.floor(random.uniform() * names.length)] ?? "s1"],
        time: l * 100 + k,
      })),
    );
    const { log } = simulate(afm({ parameters }), attempts, random);
    const fit = afm().fit(log);
    for (const [skill, [intercept, slope]] of Object.entries(skills)) {
      const found = defined(fit.model.parameters.skills[skill]);
      expect(Math.abs(found.intercept - intercept), `${skill} intercept`).toBeLessThanOrEqual(AFM_INTERCEPT_THRESHOLD);
      expect(Math.abs((found.coefficients.opportunities ?? 0) - slope), `${skill} slope`).toBeLessThanOrEqual(
        AFM_SLOPE_THRESHOLD,
      );
    }
    const learners = defined(fit.model.parameters.learners);
    const estimated = thetas.map((_, l) => learners[`l${l}`] ?? 0);
    expect(correlation(estimated, thetas)).toBeGreaterThanOrEqual(PROFICIENCY_CORRELATION);
    const sd = defined(afm({ proficiency: "random" }).fit(log).model.parameters.learnerSd);
    expect(Math.abs(sd - 0.8)).toBeLessThanOrEqual(SD_THRESHOLD);
  });

  it("recovers abilities and difficulties of Rasch learners with Elo", () => {
    const abilities = Array.from({ length: 300 }, () => random.normal());
    const difficulties = Array.from({ length: 40 }, () => random.normal());
    // The Rasch model as a logistic model: one skill per item, its intercept minus the difficulty.
    const rasch = logistic({
      proficiency: "fixed",
      parameters: {
        skills: Object.fromEntries(difficulties.map((d, i) => [`q${i}`, { intercept: -d, coefficients: {} }])),
        learners: Object.fromEntries(abilities.map((theta, l) => [`l${l}`, theta])),
      },
    });
    const attempts = abilities.flatMap((_, l) =>
      Array.from(permutation(difficulties.length, random), (i, k) => ({
        learner: `l${l}`,
        item: `q${i}`,
        skills: [`q${i}`],
        time: k * 1000 + l,
      })),
    );
    const fit = elo().fit(simulate(rasch, attempts, random).log);
    const found = abilities.map((_, l) => defined(fit.states[`l${l}`]).ability);
    const items = difficulties.map((_, i) => defined(fit.model.parameters.items[`q${i}`]).difficulty);
    expect(correlation(found, abilities)).toBeGreaterThanOrEqual(ELO_ABILITY_CORRELATION);
    expect(correlation(items, difficulties)).toBeGreaterThanOrEqual(ELO_DIFFICULTY_CORRELATION);
  });

  it("recovers IRT abilities by EAP", () => {
    const items: Record<string, IrtItem> = Object.fromEntries(
      Array.from({ length: 30 }, (_, i) => [
        `q${i}`,
        { a: 0.8 + 1.2 * random.uniform(), b: random.normal(), c: i % 3 === 0 ? 0.15 : 0 },
      ]),
    );
    const thetas = Array.from({ length: 200 }, () => random.normal());
    const model = irt({ parameters: { items } });
    const estimates = thetas.map((theta, l) => {
      // A learner of known ability: a prior concentrated at it.
      const learner = irt({ parameters: { items }, prior: { mean: theta, sd: 1e-3 } });
      const attempts = Object.keys(items).map((item, time) => ({ learner: `l${l}`, item, skills: [], time }));
      const { log } = simulate(learner, attempts, random);
      return model.ability(log.reduce((state, event) => model.update(state, event), model.initialState(`l${l}`))).theta;
    });
    expect(correlation(estimates, thetas)).toBeGreaterThanOrEqual(IRT_CORRELATION);
  });

  const structure = surmiseStructure({
    a: [],
    b: ["a"],
    c: ["a"],
    d: ["b"],
    e: ["b", "c"],
    f: [],
    g: ["f"],
    h: ["e", "g"],
  });
  const weights = structure.states.map(() => 0.2 + random.uniform());
  const total = weights.reduce((sum, w) => sum + w, 0);
  const truth = {
    probabilities: weights.map((w) => w / total),
    items: Object.fromEntries(
      structure.domain.map((item) => [
        item,
        { guess: 0.05 + 0.15 * random.uniform(), slip: 0.05 + 0.15 * random.uniform() },
      ]),
    ),
  };

  // Whether every state with the element added, or removed, is a state: the BLIM then cannot tell the item's guess, or
  // slip, from the states' probabilities (Stefanutti et al., 2012), so only the other rates are recovered.
  const keys = new Set(structure.states.map((state) => [...state].sort().join()));
  const graded = (change: (state: readonly string[]) => string[]) =>
    structure.states.every((state) => keys.has(change(state).sort().join()));
  const forward = (element: string) => graded((state) => [...new Set([...state, element])]);
  const backward = (element: string) => graded((state) => state.filter((e) => e !== element));

  it("recovers the BLIM's identifiable guesses and slips, at a log-likelihood at least the truth's", () => {
    // The work is the 8 events of each learner, simulated once and replayed by two fits. 20,000 learners recover the
    // rates to 0.015, within half of the threshold, where 15,000 reach 0.029. A generator of its own keeps that result
    // independent of the tests before this one.
    const own = createRandom(20261029);
    const attempts = Array.from({ length: 20_000 }, (_, l) =>
      structure.domain.map((item, i) => ({ learner: `l${l}`, item, skills: [], time: l * 10 + i })),
    ).flat();
    const { log } = simulate(kst({ structure, over: "items", parameters: truth }), attempts, own);
    const fit = kst({ structure, over: "items" }).fit(log);
    for (const [item, rates] of Object.entries(truth.items)) {
      const found = defined(fit.model.parameters.items[item]);
      if (!forward(item)) {
        expect(Math.abs(found.guess - rates.guess), `${item} guess`).toBeLessThanOrEqual(KST_RATE_THRESHOLD);
      }
      if (!backward(item)) {
        expect(Math.abs(found.slip - rates.slip), `${item} slip`).toBeLessThanOrEqual(KST_RATE_THRESHOLD);
      }
    }
    expect(["a", "f"].every(forward) && ["d", "h"].every(backward)).toBe(true);
    const atTruth = kst({ structure, over: "items", parameters: truth, maxIterations: 1 }).fit(log);
    expect(fit.logLikelihood).toBeGreaterThanOrEqual(atTruth.logLikelihood);
  });

  it("finds learners' knowledge states by adaptive assessment as often as its posterior expects", () => {
    const model = kst({ structure, over: "items", parameters: truth });
    let found = 0;
    let expected = 0;
    const learners = 500;
    for (let l = 0; l < learners; l++) {
      // Draw the learner's state from π, then answer each question from it.
      const state = defined(structure.states[draw(truth.probabilities, random)]);
      let open: Attempt[] = structure.domain.map((item) => ({ learner: "ana", item, skills: [], time: 0 }));
      let posterior: KstState = model.initialState("ana");
      for (let q = model.nextQuestion(posterior, open); q !== null; q = model.nextQuestion(posterior, open)) {
        const { guess, slip } = defined(truth.items[q.item]);
        const outcome = random.uniform() < (state.includes(q.item) ? 1 - slip : guess) ? 1 : 0;
        posterior = model.update(posterior, { ...q, outcome });
        open = open.filter((candidate) => candidate !== q);
      }
      const likely = model.mostLikelyState(posterior);
      expected += likely.probability;
      if ([...likely.state].sort().join() === [...state].sort().join()) found += 1;
    }
    const tolerance = CALIBRATION_STANDARD_ERRORS * Math.sqrt(0.25 / learners);
    expect(Math.abs(found - expected) / learners).toBeLessThanOrEqual(tolerance);
  });
});

describe("simulate input from JavaScript", () => {
  it("rejects a model that is not a learner model", () => {
    expect(() => simulate(JSON.parse("[]"), [], createRandom(1))).toThrow(
      "A simulation needs a learner model, such as bkt(), found an array.",
    );
  });

  it("rejects a random number generator that is not one", () => {
    const model = bkt({ parameters: { skills: { s: { prior: 0.3, learn: 0.2, forget: 0, guess: 0.2, slip: 0.1 } } } });
    expect(() => simulate(model, [{ learner: "ana", item: "q", skills: ["s"], time: 0 }], JSON.parse("null"))).toThrow(
      "A random number generator must be one from createRandom, found null.",
    );
  });
});
