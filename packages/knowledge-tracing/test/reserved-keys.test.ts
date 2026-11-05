import { describe, expect, it } from "vitest";
import { bkt } from "../src/bkt/index.js";
import { buildEventLog, type Event } from "../src/data/index.js";
import { elo } from "../src/elo/index.js";
import type { LearnerModel } from "../src/index.js";
import { irt } from "../src/irt/index.js";
import { kst } from "../src/kst/index.js";
import { logistic } from "../src/logistic/index.js";
import { defined } from "./assertions.js";

/** The names that reach a member of every object instead of an entry. */
const NAMES = ["__proto__", "constructor", "prototype"] as const;

const name = (index: number): string => defined(NAMES[index % NAMES.length]);

const EVENTS: readonly Event[] = Array.from({ length: 180 }, (_, k) => ({
  learner: name(k),
  item: name(k + Math.floor(k / 3)),
  skills: [name(k + Math.floor(k / 9))],
  time: k + 1,
  outcome: (k * k + k) % 5 < 3 ? 1 : 0,
}));
const LOG = buildEventLog(EVENTS);

/** Returns a value as storage returns it, through JSON. */
function stored<T>(value: T): T {
  return JSON.parse(JSON.stringify(value));
}

/** Expects every learner's stored state of a model built from stored parameters to take an event and to predict. */
function expectReplayed<TState, TParameters>(model: LearnerModel<TState, TParameters>): void {
  for (const learner of NAMES) {
    const state = stored(model.update(model.initialState(learner), { ...defined(EVENTS[0]), learner }));
    const prediction = model.predict(state, { learner, item: name(1), skills: [name(2)], time: 2 });
    expect(Number.isFinite(prediction), learner).toBe(true);
  }
}

describe("ids named as members of an object", () => {
  it("are data of a fit of BKT, of its stored parameters and states, and of its option fixed", () => {
    const { model } = bkt({ resourceLearning: true, itemGuessSlip: true, restarts: 1 }).fit(LOG);
    expect(Object.keys(model.parameters.skills).sort()).toEqual([...NAMES].sort());
    const parameters = stored(model.parameters);
    expectReplayed(bkt({ resourceLearning: true, itemGuessSlip: true, parameters }));
    const fixed = JSON.parse('{ "constructor": { "learn": 0.2, "resources": { "__proto__": { "learn": 0.3 } } } }');
    const held = bkt({ resourceLearning: true, fixed, restarts: 1 }).fit(LOG).model.parameters.skills;
    expect(Object.hasOwn(held, "constructor")).toBe(true);
  });

  it("are data of a fit of Elo, of its stored parameters and of its stored states", () => {
    const { model } = elo({ skills: true }).fit(LOG);
    expect(Object.keys(model.parameters.items).sort()).toEqual([...NAMES].sort());
    expectReplayed(elo({ skills: true, parameters: stored(model.parameters) }));
  });

  it("are data of the items of online IRT", () => {
    const items = Object.fromEntries(NAMES.map((id) => [id, { a: 1, b: 0, c: 0 }]));
    expectReplayed(irt(stored({ items })));
  });

  it("are data of a fit of KST and of its stored parameters", () => {
    const structure = {
      domain: [...NAMES],
      states: [[], ["__proto__"], ["__proto__", "constructor"], [...NAMES]],
    };
    const { model } = kst(structure, { over: "items" }).fit(LOG);
    expect(Object.keys(model.parameters.items).sort()).toEqual([...NAMES].sort());
    expectReplayed(kst(structure, { over: "items", parameters: stored(model.parameters) }));
  });

  it("are data of a logistic model's covariates and exposures, whose stored records hold them", () => {
    const events = EVENTS.map((event, k) => ({ ...event, covariates: JSON.parse(`{ "constructor": ${k % 2} }`) }));
    const options = {
      features: ["opportunities", "exposures"],
      covariates: ["constructor"],
      proficiency: "none",
      ridge: 0.5,
    } as const;
    const { model } = logistic(options).fit(buildEventLog(events));
    const parameters = stored(model.parameters);
    const replayed = logistic({ ...options, parameters });
    const state = stored(replayed.initialState("constructor"));
    const after = stored(replayed.update(state, { ...defined(events[0]), learner: "constructor" }));
    const attempt = {
      learner: "constructor",
      item: "prototype",
      skills: ["__proto__"],
      time: 2,
      covariates: JSON.parse('{ "constructor": 1 }'),
    };
    expect(Number.isFinite(replayed.predict(after, attempt))).toBe(true);
  });
});
