import { test } from "vitest";
import { elo, recommend } from "../src/index.js";
import { benchmark } from "./budgets.js";

const CANDIDATES = 500;
const attempts = Array.from({ length: CANDIDATES }, (_, i) => ({
  learner: "ana",
  item: `q${i}`,
  skills: [],
  time: 1,
}));

test("ranks 500 candidate items by their distance to the band of success", async (context) => {
  const model = elo({
    parameters: {
      items: Object.fromEntries(attempts.map(({ item }, i) => [item, { difficulty: (i - 250) / 100, count: 20 }])),
    },
  });
  const state = model.initialState("ana");
  await benchmark(context, "practice.recommend", () => recommend(model, state, attempts));
});
