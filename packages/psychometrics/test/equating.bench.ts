import { createRandom } from "@measurekit/core";
import { test } from "vitest";
import {
  bootstrapEquating,
  type CalibratedForm,
  chainedEquipercentileEquating,
  equipercentileEquating,
  frequencyEstimationEquating,
  haebaraLinking,
  irtObservedScoreEquating,
  kernelEquating,
  linearEquating,
  linkCovariance,
  linkedEquatingErrors,
  logLinearPresmoothing,
  stockingLordLinking,
  transformItem,
  trueScoreEquating,
  tuckerEquating,
} from "../src/equating/index.js";
import { type IrtItem, threeParameterLogistic, twoParameterLogistic } from "../src/irt/index.js";
import { defined } from "./assertions.js";
import { benchmark } from "./budgets.js";
import { readFixture } from "./fixtures.js";

interface BenchEquatingFixture {
  randomGroups: { from: number[]; to: number[] };
  neat: { from: number[][]; to: number[][] };
  irt: { forms: { ids: string[]; a: number[]; b: number[]; c: number[] }[]; common: string[]; points: number };
}
const fixture = readFixture<BenchEquatingFixture>("bench-equating.json");

// biome-ignore-start lint/style/useNamingConvention: the keys are those of the fixture file.
interface BenchEquatingLinksFixture {
  forms: {
    T1: {
      items: { id: string; a: number; b: number }[];
      covariance: { parameters: { item: string; name: string }[]; matrix: number[][] };
    };
    T2: {
      items: { id: string; a: number; b: number }[];
      covariance: { parameters: { item: string; name: string }[]; matrix: number[][] };
    };
    T3: {
      items: { id: string; a: number; b: number }[];
      covariance: { parameters: { item: string; name: string }[]; matrix: number[][] };
    };
  };
  points: number;
}
// biome-ignore-end lint/style/useNamingConvention: the keys are those of the fixture file.
const linksFixture = readFixture<BenchEquatingLinksFixture>("bench-equating-links.json");

const SEED = 20261029;
/** The replications of the bootstrap of the standard errors. */
const REPLICATIONS = 100;
const quadrature = { method: "gauss-hermite", points: fixture.irt.points } as const;

const { randomGroups, neat } = fixture;

test("equates two forms of 40 items taken by about 4,300 persons each, in random groups", async (context) => {
  await benchmark(context, "equating.linear", () => linearEquating(randomGroups.from, randomGroups.to));
  await benchmark(context, "equating.equipercentile", () => equipercentileEquating(randomGroups.from, randomGroups.to));
  await benchmark(context, "equating.presmoothing", () => logLinearPresmoothing(randomGroups.from, 3));
  const sample = { design: "random-groups" as const, from: randomGroups.from, to: randomGroups.to };
  await benchmark(context, "equating.bootstrapLinear", () =>
    bootstrapEquating(sample, (s) => linearEquating(s.from, s.to), createRandom(SEED), { replications: REPLICATIONS }),
  );
});

test("equates two forms of 36 items with an anchor of 12 taken by about 1,650 persons each, in nonequivalent groups", async (context) => {
  await benchmark(context, "equating.tucker", () => tuckerEquating(neat.from, neat.to));
  await benchmark(context, "equating.frequencyEstimation", () => frequencyEstimationEquating(neat.from, neat.to));
  await benchmark(context, "equating.chainedEquipercentile", () => chainedEquipercentileEquating(neat.from, neat.to));
});

test("links and equates two forms of 3PL items that share 10 items", async (context) => {
  const [first, second] = fixture.irt.forms;
  const form = (parameters: (typeof fixture.irt.forms)[number]) =>
    parameters.ids.map((id, i): IrtItem & { id: string } => ({
      id,
      model: threeParameterLogistic,
      parameters: { a: defined(parameters.a[i]), b: defined(parameters.b[i]), c: defined(parameters.c[i]), d: 1 },
    }));
  const from = form(defined(first));
  const to = form(defined(second));
  const common = (items: readonly (IrtItem & { id: string })[]) =>
    fixture.irt.common.map((id) => defined(items.find((item) => item.id === id)));
  const options = { quadrature };
  await benchmark(context, "equating.stockingLordLinking", () =>
    stockingLordLinking(common(from), common(to), options),
  );
  await benchmark(context, "equating.haebaraLinking", () => haebaraLinking(common(from), common(to), options));
  const linking = stockingLordLinking(common(from), common(to), options);
  const linked = from.map((item) => transformItem(item, linking));
  await benchmark(context, "equating.trueScore", () => trueScoreEquating(linked, to));
  await benchmark(context, "equating.irtObservedScore", () => irtObservedScoreEquating(linked, to));
});

test("equates two forms of 40 items in random groups by kernel equating, with its standard errors", async (context) => {
  await benchmark(context, "equating.kernel", () =>
    kernelEquating({ design: "equivalent-groups", from: randomGroups.from, to: randomGroups.to, degrees: [3, 3] }),
  );
});

test("finds the standard errors of links between separate calibrations of forms of 20 items", async (context) => {
  const form = (name: "T1" | "T2" | "T3"): CalibratedForm => {
    const { items, covariance } = linksFixture.forms[name];
    return {
      items: items.map(({ id, a, b }) => ({ id, model: twoParameterLogistic, parameters: { a, b, c: 0, d: 1 } })),
      covariance,
    };
  };
  const forms = [form("T3"), form("T2"), form("T1")];
  const linkQuadrature = { method: "gauss-hermite", points: linksFixture.points } as const;
  await benchmark(context, "equating.linkCovariance", () =>
    linkCovariance(forms, { linking: "haebara", quadrature: linkQuadrature }),
  );
  await benchmark(context, "equating.linkedErrors", () =>
    linkedEquatingErrors(form("T1"), form("T2"), { linking: "mean-mean", quadrature: linkQuadrature }),
  );
});
