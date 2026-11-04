// A practice log of 60 learners on 12 items and 3 skills, simulated with a seed. Compares learner-item Elo,
// concept-specific Elo and Bayesian knowledge tracing by cross-validation by learner and prints the learning curve of
// each skill; then fits Elo to the log and uses it online: predict, store the learner's state as JSON, resume it and
// update it; finally ranks the items of the skills BKT does not count as mastered, and finds when each learner first
// answered three in a row.
// Run after `pnpm build`: node examples/knowledge-tracing/practice-session.ts
import {
  bkt,
  buildEventLog,
  compareModels,
  createRandom,
  type EloState,
  elo,
  learningCurve,
  recommend,
  splitByLearner,
  streakMastery,
} from "@measurekit/knowledge-tracing";

const random = createRandom(20261029);
const skills = ["fractions", "decimals", "percentages"];
const qMatrix = Object.fromEntries(
  Array.from({ length: 12 }, (_, i) => [`q${i + 1}`, skills.slice(i % 3, (i % 3) + 1)]),
);
const events = Array.from({ length: 60 }, (_, learner) => {
  const ability = random.normal();
  // Each learner practices 15 items at random; practice raises the chance of a correct answer.
  return Array.from({ length: 15 }, (_, attempt) => {
    const item = `q${1 + Math.floor(random.uniform() * 12)}`;
    const chance = 1 / (1 + Math.exp(-(ability + 0.15 * attempt - 0.3)));
    const outcome = random.uniform() < chance ? 1 : 0;
    return { learner: `learner${learner + 1}`, item, time: learner * 100 + attempt, outcome };
  });
}).flat();
const log = buildEventLog(events, { qMatrix });

const folds = splitByLearner(log, 5, random);
const models = { elo: elo(), "concept elo": elo({ skills: true }), bkt: bkt() };
for (const { name, metrics } of compareModels(models, folds)) {
  console.log(name, metrics);
}
for (const [skill, points] of Object.entries(learningCurve(log).skills)) {
  console.log(
    skill,
    points.slice(0, 5).map((point) => point.errorRate.toFixed(2)),
  );
}

// Online: learner 7 returns and answers q2. The state goes to storage as JSON between requests.
const { model, states } = elo().fit(log);
const stored = JSON.stringify(states.learner7 ?? model.initialState("learner7"));
const state: EloState = JSON.parse(stored);
const attempt = { learner: "learner7", item: "q2", skills: qMatrix.q2 ?? [], time: 10_000 };
console.log("P(correct):", model.predict(state, attempt).toFixed(3));
const next = model.update(state, { ...attempt, outcome: 0 });
console.log("P(correct) after an incorrect answer:", model.predict(next, attempt).toFixed(3));

// Practice: the items of the skills BKT does not count as mastered, ranked by Elo's predicted success.
const tracer = bkt().fit(log);
const known = tracer.states.learner7 ?? tracer.model.initialState("learner7");
const open = Object.entries(qMatrix)
  .filter(([, itemSkills]) => itemSkills.some((skill) => tracer.model.mastery(known, skill) < 0.95))
  .map(([item, itemSkills]) => ({ learner: "learner7", item, skills: itemSkills, time: 10_001 }));
const [best] = recommend(model, next, open);
console.log("Next item:", best?.attempt.item, best?.prediction.toFixed(3));
console.log("First three correct in a row, by skill:", streakMastery(log).learner7);
