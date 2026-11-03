import { checkRecord, DataError, describe, type LogisticItem, mean, type ThresholdItem } from "@measurekit/core";
import {
  centered,
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
  checkMultidimensionalItem,
  checkMultidimensionalModel,
  type MultidimensionalGradedParameters,
  type MultidimensionalItem,
  type MultidimensionalLogisticParameters,
  type MultidimensionalModel,
  multidimensionalGradedResponse,
  multidimensionalThreeParameterLogistic,
  multidimensionalTwoParameterLogistic,
} from "../multidimensional/internal.js";

/**
 * One item's coefficients as a row of mirt's `coef(model, simplify = TRUE)$items` gives them, without the columns of
 * other item types: names to values, such as `{ a: 1.2, b: -0.3, g: 0.2, u: 1 }` or `{ a1: 1.1, d1: 2.3, d2: -0.4 }`.
 *
 * @remarks
 * Values may be `undefined` so that rows of a JSON array of several item types type-check; a name a form uses must have
 * a finite value.
 */
export interface MirtCoefficients {
  /** The coefficient of each name. */
  readonly [name: string]: number | undefined;
}

/**
 * Returns an item's coefficients in mirt's IRT form, as `coef(model, IRTpars = TRUE, simplify = TRUE)$items` gives
 * them for the item's model.
 *
 * @param item - The item.
 * @returns The coefficients by name, in the IRT form.
 * @throws `DataError` when the item's parameters are invalid, or its model is not one of the built-in models.
 */
export function toMirtIrt(item: IrtItem): MirtCoefficients {
  return toMirt(item, "irt");
}

/**
 * Returns an item's coefficients in mirt's slope-intercept form, as `coef(model, IRTpars = FALSE, simplify =
 * TRUE)$items` gives them for the item's model.
 *
 * @param item - The item.
 * @returns The coefficients by name, in the slope-intercept form.
 * @throws `DataError` when the item's parameters are invalid or have no slope-intercept form, or its model is not one
 * of the built-in models.
 */
export function toMirtSlopeIntercept(item: IrtItem): MirtCoefficients {
  return toMirt(item, "slopeIntercept");
}

/**
 * Returns the item of a model with mirt's coefficients in the IRT form.
 *
 * @param coefficients - The coefficients of one item, as `coef(model, IRTpars = TRUE, simplify = TRUE)$items` gives
 *   them.
 * @param model - The model of the item.
 * @returns The item.
 * @throws `DataError` when the coefficients lack a name of the form, have a name it does not use, or give invalid
 * parameters, or the model is not one of the built-in models.
 */
export function fromMirtIrt<TParameters>(
  coefficients: MirtCoefficients,
  model: IrtModel<TParameters>,
): IrtItem<TParameters> {
  return fromMirt(coefficients, model, "irt");
}

/**
 * Returns the item of a model with mirt's coefficients in the slope-intercept form.
 *
 * @param coefficients - The coefficients of one item, as `coef(model, IRTpars = FALSE, simplify = TRUE)$items` gives
 *   them.
 * @param model - The model of the item.
 * @returns The item.
 * @throws `DataError` when the coefficients lack a name of the form, have a name it does not use, or give invalid
 * parameters, or the model is not one of the built-in models.
 */
export function fromMirtSlopeIntercept<TParameters>(
  coefficients: MirtCoefficients,
  model: IrtModel<TParameters>,
): IrtItem<TParameters> {
  return fromMirt(coefficients, model, "slopeIntercept");
}

/**
 * Returns a multidimensional item's coefficients in mirt's slope-intercept form, as `coef(model, simplify =
 * TRUE)$items` gives them for the item's model: the slopes `a1` to `aD`, and `d`, `g` and `u` for a logistic item or
 * `d1` to `d(K-1)` for a graded one.
 *
 * @param item - The multidimensional item.
 * @returns The coefficients by name, in the slope-intercept form.
 * @throws `DataError` when the item's parameters are invalid, or its model is not one of the built-in multidimensional
 * models.
 */
export function toMirtMultidimensional(item: MultidimensionalItem): MirtCoefficients {
  checkMultidimensionalItem(item, "The item");
  return multidimensionalFormOf(item.model).toMirt(item.parameters);
}

/**
 * Returns the multidimensional item of a model with mirt's coefficients in the slope-intercept form.
 *
 * @param coefficients - The coefficients of one item, as `coef(model, simplify = TRUE)$items` gives them.
 * @param model - The model of the item.
 * @returns The item.
 * @throws `DataError` when the coefficients lack a name of the form, have a name it does not use, or give invalid
 * parameters, or the model is not one of the built-in multidimensional models.
 */
export function fromMirtMultidimensional<TParameters>(
  coefficients: MirtCoefficients,
  model: MultidimensionalModel<TParameters>,
): MultidimensionalItem<TParameters> {
  checkRecord(coefficients, "The mirt coefficients must be an object of numbers by name");
  checkMultidimensionalModel(
    model,
    "The model must be one of the built-in multidimensional models, such as multidimensionalTwoParameterLogistic",
  );
  const parameters = readParameters(
    coefficients,
    multidimensionalFormOf(model),
    `The mirt coefficients of a multidimensional ${model.name} item`,
    "form",
  ) as TParameters;
  checkMultidimensionalItem({ model, parameters }, `The multidimensional ${model.name} item of the mirt coefficients`);
  return { model, parameters };
}

/** A model's conversions to and from mirt's coefficient forms. */
interface MirtForms {
  readonly irt: MirtForm;
  readonly slopeIntercept: MirtForm;
}

/** The conversion of a model's parameters to and from one of mirt's coefficient forms. */
interface MirtForm {
  /** Returns the coefficients of valid parameters of the model. */
  toMirt(parameters: unknown): MirtCoefficients;
  /** Returns the parameters given by coefficients, read with `read`. */
  fromMirt(read: CoefficientReader): unknown;
}

/** Reads coefficients by name, each a finite number. */
interface CoefficientReader {
  /** Returns the coefficient `name`. */
  value(name: string): number;
  /** Returns the coefficients `prefix` + `first`, `prefix` + (`first` + 1), ... up to the first name not given. */
  series(prefix: string, first: number): number[];
}

const LOGISTIC: MirtForms = {
  irt: {
    toMirt: ({ a, b, c, d }: LogisticItem) => ({ a, b, g: c, u: d }),
    fromMirt: (read) => ({ a: read.value("a"), b: read.value("b"), c: read.value("g"), d: read.value("u") }),
  },
  slopeIntercept: {
    toMirt: ({ a, b, c, d }: LogisticItem) => ({ a1: a, d: -a * b, g: c, u: d }),
    fromMirt(read) {
      const a1 = nonzeroSlope(read);
      return { a: a1, b: -read.value("d") / a1, c: read.value("g"), d: read.value("u") };
    },
  },
};

const THRESHOLD_IRT: MirtForm = {
  toMirt: ({ a, b }: ThresholdItem) => ({ a, ...numbered("b", 1, b) }),
  fromMirt: (read) => ({ a: read.value("a"), b: read.series("b", 1) }),
};

const GRADED: MirtForms = {
  irt: THRESHOLD_IRT,
  slopeIntercept: {
    toMirt: ({ a, b }: ThresholdItem) => ({
      a1: a,
      ...numbered(
        "d",
        1,
        b.map((threshold) => -a * threshold),
      ),
    }),
    fromMirt(read) {
      const a1 = nonzeroSlope(read);
      return { a: a1, b: read.series("d", 1).map((intercept) => -intercept / a1) };
    },
  },
};

const PARTIAL_CREDIT: MirtForms = {
  irt: THRESHOLD_IRT,
  slopeIntercept: {
    toMirt({ a, b }: ThresholdItem) {
      const intercepts = [0];
      for (const threshold of b) intercepts.push((intercepts.at(-1) ?? 0) - a * threshold);
      return {
        a1: a,
        ...numbered(
          "ak",
          0,
          intercepts.map((_, k) => k),
        ),
        ...numbered("d", 0, intercepts),
      };
    },
    fromMirt(read) {
      const a1 = nonzeroSlope(read);
      const scores = read.series("ak", 0);
      const intercepts = read.series("d", 0);
      if (!(scores.every((score, k) => score === k) && intercepts.length === scores.length && intercepts[0] === 0)) {
        throw new DataError(
          `The (generalized) partial credit model has mirt's scores ak0, ak1, ... equal to 0, 1, ... and d0 = 0, one d per score, found ak = [${scores.join(", ")}], d = [${intercepts.join(", ")}].`,
        );
      }
      return { a: a1, b: intercepts.slice(1).map((intercept, j) => ((intercepts[j] ?? 0) - intercept) / a1) };
    },
  },
};

// mirt's rsm item type has one form: P(k) ∝ exp(a1 (k θ - b1 - ... - bk) + k c), whose thresholds are b_j - c / a1.
const RATING_SCALE_FORM: MirtForm = {
  toMirt: ({ a, b, steps }: RatingScaleParameters) => ({ a1: a, ...numbered("b", 1, steps), c: -a * b }),
  fromMirt(read) {
    const a1 = nonzeroSlope(read);
    const shift = read.value("c") / a1;
    const thresholds = read.series("b", 1).map((step) => step - shift);
    const b = mean(thresholds);
    return { a: a1, b, steps: thresholds.map((threshold) => threshold - b) };
  },
};

const NOMINAL: MirtForms = {
  irt: {
    toMirt: ({ a, c }: NominalParameters) => ({ ...numbered("a", 1, centered(a)), ...numbered("c", 1, centered(c)) }),
    fromMirt: (read) => ({ a: read.series("a", 1), c: read.series("c", 1) }),
  },
  slopeIntercept: {
    // mirt identifies the nominal model by ak0 = 0, ak(K-1) = K - 1 and d0 = 0.
    toMirt({ a, c }: NominalParameters) {
      const last = a.length - 1;
      const a1 = ((a[last] ?? 0) - (a[0] ?? 0)) / last;
      if (a1 === 0) {
        throw new DataError(
          "An NRM item whose first and last categories have equal slopes has no slope-intercept form in mirt, which fixes ak0 = 0 and ak(K-1) = K - 1.",
        );
      }
      return {
        a1,
        ...numbered(
          "ak",
          0,
          a.map((slope) => (slope - (a[0] ?? 0)) / a1),
        ),
        ...numbered(
          "d",
          0,
          c.map((intercept) => intercept - (c[0] ?? 0)),
        ),
      };
    },
    fromMirt(read) {
      const a1 = read.value("a1");
      return { a: centered(read.series("ak", 0).map((score) => a1 * score)), c: centered(read.series("d", 0)) };
    },
  },
};

const FORMS = new Map<IrtModel, MirtForms>([
  [oneParameterLogistic, LOGISTIC],
  [twoParameterLogistic, LOGISTIC],
  [threeParameterLogistic, LOGISTIC],
  [fourParameterLogistic, LOGISTIC],
  [gradedResponse, GRADED],
  [generalizedPartialCredit, PARTIAL_CREDIT],
  [partialCredit, PARTIAL_CREDIT],
  [ratingScale, { irt: RATING_SCALE_FORM, slopeIntercept: RATING_SCALE_FORM }],
  [nominalResponse, NOMINAL],
]);

const MULTIDIMENSIONAL_LOGISTIC: MirtForm = {
  toMirt: ({ a, intercept, c, d }: MultidimensionalLogisticParameters) => ({
    ...numbered("a", 1, a),
    d: intercept,
    g: c,
    u: d,
  }),
  fromMirt: (read) => ({ a: read.series("a", 1), intercept: read.value("d"), c: read.value("g"), d: read.value("u") }),
};

const MULTIDIMENSIONAL_GRADED: MirtForm = {
  toMirt: ({ a, intercepts }: MultidimensionalGradedParameters) => ({
    ...numbered("a", 1, a),
    ...numbered("d", 1, intercepts),
  }),
  fromMirt: (read) => ({ a: read.series("a", 1), intercepts: read.series("d", 1) }),
};

const MULTIDIMENSIONAL_FORMS = new Map<MultidimensionalModel, MirtForm>([
  [multidimensionalTwoParameterLogistic, MULTIDIMENSIONAL_LOGISTIC],
  [multidimensionalThreeParameterLogistic, MULTIDIMENSIONAL_LOGISTIC],
  [multidimensionalGradedResponse, MULTIDIMENSIONAL_GRADED],
]);

function multidimensionalFormOf(model: MultidimensionalModel): MirtForm {
  const form = MULTIDIMENSIONAL_FORMS.get(model);
  if (form === undefined) {
    throw new DataError(`Only the built-in models convert to and from mirt, found the ${describe(model.name)} model.`);
  }
  return form;
}

function formsOf(model: IrtModel): MirtForms {
  const forms = FORMS.get(model);
  if (forms === undefined) {
    throw new DataError(
      `Only the built-in models other than the GGUM convert to and from mirt, found the ${describe(model.name)} model.`,
    );
  }
  return forms;
}

function toMirt(item: IrtItem, form: keyof MirtForms): MirtCoefficients {
  checkItem(item, "The item");
  return formsOf(item.model)[form].toMirt(item.parameters);
}

function fromMirt<TParameters>(
  coefficients: MirtCoefficients,
  model: IrtModel<TParameters>,
  form: keyof MirtForms,
): IrtItem<TParameters> {
  checkRecord(coefficients, "The mirt coefficients must be an object of numbers by name");
  checkModel(model, "The model must be one of the built-in models, such as twoParameterLogistic");
  const parameters = readParameters(
    coefficients,
    formsOf(model)[form],
    `The mirt coefficients of a ${model.name} item`,
    form === "irt" ? "IRT form" : "slope-intercept form",
  ) as TParameters;
  checkItem({ model, parameters }, `The ${model.name} item of the mirt coefficients`);
  return { model, parameters };
}

/**
 * Returns the parameters that a form reads from the coefficients, which must use all of them. The form builds them from
 * untyped coefficients, so the caller's model checks them before they are typed.
 *
 * @param subject - Starts the messages, such as `The mirt coefficients of a 2PL item`.
 * @param formName - Names the form in the message about coefficients it does not use.
 * @throws `DataError` when a coefficient the form reads is missing or not finite, or the coefficients have a name it does
 * not use.
 */
function readParameters(coefficients: MirtCoefficients, form: MirtForm, subject: string, formName: string): unknown {
  const used = new Set<string>();
  const value = (name: string): number => {
    const coefficient = coefficients[name];
    if (!(typeof coefficient === "number" && Number.isFinite(coefficient))) {
      throw new DataError(`${subject} need a finite ${name}, found ${describe(coefficient)}.`);
    }
    used.add(name);
    return coefficient;
  };
  const series = (prefix: string, first: number): number[] => {
    const values: number[] = [];
    // The first name of a series is required: reading it throws when it is missing.
    values.push(value(`${prefix}${first}`));
    for (let k = first + 1; `${prefix}${k}` in coefficients; k++) values.push(value(`${prefix}${k}`));
    return values;
  };
  const parameters = form.fromMirt({ value, series });
  const unusedNames = Object.keys(coefficients).filter((name) => !used.has(name));
  if (unusedNames.length > 0) {
    throw new DataError(`${subject} have names its ${formName} does not use: ${unusedNames.join(", ")}.`);
  }
  return parameters;
}

/** Returns the slope a1, which a conversion divides by. */
function nonzeroSlope(read: CoefficientReader): number {
  const a1 = read.value("a1");
  if (a1 === 0) throw new DataError("An item with slope a1 = 0 has no IRT parameters, which divide by a1.");
  return a1;
}

/** Returns the values named `prefix` + `first`, `prefix` + (`first` + 1), ... */
function numbered(prefix: string, first: number, values: readonly number[]): Record<string, number> {
  return Object.fromEntries(values.map((value, k) => [`${prefix}${k + first}`, value]));
}
