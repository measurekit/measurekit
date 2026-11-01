import {
  checkOptions,
  checkPositiveInteger,
  checkRandom,
  checkRecord,
  DataError,
  describe,
  type LogisticItem,
  type Random,
  type ThresholdItem,
} from "@measurekit/core";
import { checkFunction } from "../checks.js";
import {
  checkItem,
  checkModel,
  fourParameterLogistic,
  generalizedPartialCredit,
  gradedResponse,
  type IrtItem,
  type IrtModel,
  type NominalParameters,
  nominalResponse,
  oneParameterLogistic,
  partialCredit,
  type RatingScaleParameters,
  ratingScale,
  threeParameterLogistic,
  twoParameterLogistic,
} from "../irt/internal.js";
import {
  type Distribution,
  logNormalDistribution,
  normalDistribution,
  STANDARD_NORMAL,
  uniformDistribution,
} from "./distributions.js";

/** Options of {@link simulateItems}. */
export interface ItemSimulationOptions {
  /**
   * The number of categories of every item: 2 for the 1PL to the 4PL, an integer from 2 for the other models.
   *
   * @defaultValue 2 for the 1PL to the 4PL, 3 for the others, as catR's `nrCat`
   */
  readonly categoryCount?: number;
  /**
   * Distributions of the model's parameters by name, such as `{ a: uniformDistribution(0.5, 2) }`, in place of the
   * defaults.
   *
   * @defaultValue catR's, as {@link simulateItems} lists them
   */
  readonly distributions?: Readonly<Record<string, Distribution>>;
}

// The defaults of catR's genDichoMatrix (aPrior, bPrior, cPrior, dPrior) and genPolyMatrix, where items without a
// slope in catR's form have a = 1.
const ONE: Distribution = () => 1;
const LOGISTIC_SLOPE = normalDistribution(1, 0.2);
const POLYTOMOUS_SLOPE = logNormalDistribution(0, 0.1225);
const LOWER_ASYMPTOTE = uniformDistribution(0, 0.25);
const UPPER_ASYMPTOTE = uniformDistribution(0.75, 1);

/** How simulation draws the parameters of a model's items. */
interface ItemDraws<TParameters> {
  readonly model: IrtModel<TParameters>;
  /** The names of the parameters drawn, which `options.distributions` may give. */
  readonly names: readonly string[];
  /**
   * Returns the parameters of `count` items with `categoryCount` categories, each parameter drawn from its distribution
   * in `given` or from its default. Parameters the model's items share are drawn once.
   */
  draw(count: number, categoryCount: number, given: Distributions, random: Random): TParameters[];
}

type Distributions = Readonly<Record<string, Distribution>>;

/** {@link ItemDraws} without the type of the parameters: it returns their vectors, checked by the model. */
interface VectorDraws {
  readonly names: readonly string[];
  vectors(count: number, categoryCount: number, given: Distributions, random: Random): Float64Array[];
}

const DRAWS = new Map<IrtModel, VectorDraws>([
  [oneParameterLogistic, vectorDraws(logisticDraws(oneParameterLogistic, ONE))],
  [twoParameterLogistic, vectorDraws(logisticDraws(twoParameterLogistic, LOGISTIC_SLOPE))],
  [threeParameterLogistic, vectorDraws(logisticDraws(threeParameterLogistic, LOGISTIC_SLOPE))],
  [fourParameterLogistic, vectorDraws(logisticDraws(fourParameterLogistic, LOGISTIC_SLOPE))],
  [gradedResponse, vectorDraws(gradedDraws())],
  [generalizedPartialCredit, vectorDraws(partialCreditDraws(generalizedPartialCredit, POLYTOMOUS_SLOPE))],
  [partialCredit, vectorDraws(partialCreditDraws(partialCredit, ONE))],
  [ratingScale, vectorDraws(ratingScaleDraws())],
  [nominalResponse, vectorDraws(nominalDraws())],
]);

/**
 * Draws the parameters of `count` items of one of the built-in models, each from its distribution in
 * `options.distributions` or from its default, catR's.
 *
 * @remarks
 * a ~ N(1, 0.2²), b ~ N(0, 1), c ~ U(0, 0.25) and d ~ U(0.75, 1) for the 1PL to the 4PL; a ~ logN(0, 0.1225²) and
 * thresholds b ~ N(0, 1) for the GPCM and the GRM, in increasing order for the GRM; locations b and steps ~ N(0, 1) for
 * the RSM; and for the NRM, slopes a ~ logN(0, 0.1225²) and intercepts c ~ N(0, 1) of the categories from 1, with 0 for
 * category 0. The parameters the model's items share, a of the 1PL, PCM and RSM and the RSM's steps, are drawn once, a
 * = 1 by default. Items come in the form calibration reports, with the RSM's steps and the NRM's slopes and intercepts
 * summing to 0.
 *
 * @param model - The model of the items.
 * @param count - The number of items.
 * @param random - The generator of the parameters.
 * @param options - The number of categories and the distribution of each parameter.
 * @returns The items, each with its parameters.
 * @throws `DataError` when the model is not one of the built-in models, `count` is not a positive integer, `random`
 * is not a random number generator, the number of categories is invalid for the model, a distribution is not a
 * function or is given for a parameter the model does not draw, or the draws give an invalid item.
 */
export function simulateItems<TParameters>(
  model: IrtModel<TParameters>,
  count: number,
  random: Random,
  options: ItemSimulationOptions = {},
): IrtItem<TParameters>[] {
  checkModel(model, "The model must be one of the built-in models, such as twoParameterLogistic");
  const draws = DRAWS.get(model);
  if (draws === undefined) {
    throw new DataError(`Simulation supports the built-in models only, found the ${describe(model.name)} model.`);
  }
  checkPositiveInteger(count, "The number of items");
  checkRandom(random);
  checkOptions(options);
  const { categoryCount = Math.min(model.maxCategories, 3), distributions = {} } = options;
  if (!(Number.isSafeInteger(categoryCount) && categoryCount >= 2 && categoryCount <= model.maxCategories)) {
    throw new DataError(
      model.maxCategories === 2
        ? `${model.name} items have 2 categories, found ${describe(categoryCount)}.`
        : `The option categoryCount must be an integer from 2, found ${describe(categoryCount)}.`,
    );
  }
  checkRecord(distributions, "The option distributions must be an object of distributions by parameter name");
  for (const [name, distribution] of Object.entries(distributions)) {
    if (!draws.names.includes(name)) {
      throw new DataError(
        `The ${model.name} model draws ${draws.names.join(", ")}; found a distribution for "${name}".`,
      );
    }
    checkFunction(distribution, `The distribution of ${name} must be a function of a random number generator`);
  }
  return draws
    .vectors(count, categoryCount, distributions, random)
    .map((vector) => ({ model, parameters: model.fromVector(vector) }));
}

function vectorDraws<TParameters>(draws: ItemDraws<TParameters>): VectorDraws {
  const { model, names } = draws;
  return {
    names,
    vectors: (count, categoryCount, given, random) =>
      draws.draw(count, categoryCount, given, random).map((parameters, i) => {
        checkItem({ model, parameters }, `Simulated item ${i + 1}`);
        return model.toVector(parameters);
      }),
  };
}

function logisticDraws(model: IrtModel<LogisticItem>, slope: Distribution): ItemDraws<LogisticItem> {
  const parameters = model.parameters(2);
  const names = parameters.map(({ name }) => name);
  return {
    model,
    names,
    draw(count, _, given, random) {
      const a = given.a ?? slope;
      const b = given.b ?? STANDARD_NORMAL;
      const c = given.c ?? LOWER_ASYMPTOTE;
      const d = given.d ?? UPPER_ASYMPTOTE;
      const shared = parameters[0]?.shared === true ? a(random) : undefined;
      return Array.from({ length: count }, () => ({
        a: shared ?? a(random),
        b: b(random),
        c: names.includes("c") ? c(random) : 0,
        d: names.includes("d") ? d(random) : 1,
      }));
    },
  };
}

function gradedDraws(): ItemDraws<ThresholdItem> {
  return {
    model: gradedResponse,
    names: ["a", "b"],
    draw(count, categoryCount, given, random) {
      const a = given.a ?? POLYTOMOUS_SLOPE;
      const b = given.b ?? STANDARD_NORMAL;
      return Array.from({ length: count }, () => {
        const slope = a(random);
        // The GRM's thresholds increase with a > 0 and decrease with a < 0.
        const thresholds = drawValues(b, categoryCount - 1, random).sort((x, y) => (slope < 0 ? y - x : x - y));
        return { a: slope, b: thresholds };
      });
    },
  };
}

function partialCreditDraws(model: IrtModel<ThresholdItem>, slope: Distribution): ItemDraws<ThresholdItem> {
  return {
    model,
    names: ["a", "b"],
    draw(count, categoryCount, given, random) {
      const a = given.a ?? slope;
      const b = given.b ?? STANDARD_NORMAL;
      const shared = model.parameters(categoryCount)[0]?.shared === true ? a(random) : undefined;
      return Array.from({ length: count }, () => ({
        a: shared ?? a(random),
        b: drawValues(b, categoryCount - 1, random),
      }));
    },
  };
}

function ratingScaleDraws(): ItemDraws<RatingScaleParameters> {
  return {
    model: ratingScale,
    names: ["a", "b", "steps"],
    draw(count, categoryCount, given, random) {
      const a = (given.a ?? ONE)(random);
      const steps = drawValues(given.steps ?? STANDARD_NORMAL, categoryCount - 1, random);
      const b = given.b ?? STANDARD_NORMAL;
      return Array.from({ length: count }, () => ({ a, b: b(random), steps }));
    },
  };
}

function nominalDraws(): ItemDraws<NominalParameters> {
  return {
    model: nominalResponse,
    names: ["a", "c"],
    draw(count, categoryCount, given, random) {
      const a = given.a ?? POLYTOMOUS_SLOPE;
      const c = given.c ?? STANDARD_NORMAL;
      return Array.from({ length: count }, () => ({
        a: [0, ...drawValues(a, categoryCount - 1, random)],
        c: [0, ...drawValues(c, categoryCount - 1, random)],
      }));
    },
  };
}

function drawValues(distribution: Distribution, count: number, random: Random): number[] {
  return Array.from({ length: count }, () => distribution(random));
}
