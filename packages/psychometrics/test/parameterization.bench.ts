import { test } from "vitest";
import { type IrtItem, twoParameterLogistic, writeMirtIrt, writeMirtSlopeIntercept } from "../src/index.js";
import { defined } from "./assertions.js";
import { benchmark } from "./budgets.js";
import { type BenchIrtFixture, readFixture } from "./fixtures.js";

const fixture = readFixture<BenchIrtFixture>("bench-irt.json");

/** A pool of 10,000 items: the 40 items of the IRT benchmarks, 250 times. */
const POOL = 250;
const items: IrtItem[] = Array.from({ length: POOL }, () =>
  fixture.items.a.map((a, i) => ({
    model: twoParameterLogistic,
    parameters: { a, b: defined(fixture.items.b[i]), c: 0, d: 1 },
  })),
).flat();

test("converts the 2PL items of a pool of 10,000 to the forms of mirt", async (context) => {
  await benchmark(context, "parameterization.writeMirtSlopeIntercept", () =>
    items.map((item) => writeMirtSlopeIntercept(item)),
  );
  await benchmark(context, "parameterization.writeMirtIrt", () => items.map((item) => writeMirtIrt(item)));
});
