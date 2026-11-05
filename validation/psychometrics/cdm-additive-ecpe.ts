// How closely calibrateCdm agrees with GDINA's additive models on ECPE, whose 2,922 persons take EM 700 to 1,100 cycles to a
// tolerance of 1e-10, 9 to 13 seconds a model under coverage: the tests compare the same models on sim30GDINA, 1,000
// persons, instead. The case is ECPE of cdm-additive.json, by cdm.R: the A-CDM, the LLM, the R-RUM and a mix of all six
// models per item, each held to the tolerances of the tests: the success probabilities, coefficients, guessing, slip and
// profile proportions within 1e-5, the log-likelihood at least GDINA's less 1e-8, as GDINA's M-steps stop short of the
// maximum, and the parameters that AIC and BIC count within 1e-6 of GDINA's.
//
// It writes results/cdm-additive-ecpe.json beside it as each fit finishes: the sizes, the tolerances, each model's largest
// differences, iterations and seconds. It takes about 10 seconds, and exits with 1 when a fit misses a tolerance. Run after
// `pnpm build`: node validation/psychometrics/cdm-additive-ecpe.ts

import { readFileSync } from "node:fs";
import { type CdmModel, calibrateCdm } from "@measurekit/psychometrics/cdm";
import { buildResponseMatrix } from "@measurekit/psychometrics/data";
import { timed, writeResults } from "../study.ts";

interface Fit {
  readonly model: string | readonly string[];
  readonly items: readonly (readonly number[])[];
  readonly coefficients: readonly (readonly number[])[];
  readonly guessing: readonly number[];
  readonly slip: readonly number[];
  readonly proportions: readonly number[];
  readonly logLikelihood: number;
  readonly aic: number;
  readonly bic: number;
}
interface Case {
  readonly name: string;
  readonly itemIds: readonly string[];
  readonly responses: readonly (readonly number[])[];
  readonly qmatrix: Readonly<Record<string, readonly string[] | null>>;
  readonly fits: readonly Fit[];
}
const { cases } = JSON.parse(
  readFileSync(new URL("../../packages/psychometrics/test/fixtures/cdm-additive.json", import.meta.url), "utf8"),
) as { readonly cases: readonly Case[] };
const ecpe = cases.find(({ name }) => name === "ECPE");
if (ecpe === undefined) throw new Error("cdm-additive.json has no case ECPE.");

const TOLERANCES = { parameters: 1e-5, logLikelihood: 1e-8, criteria: 1e-6 };
const MODELS: readonly CdmModel[] = ["gdina", "dina", "dino", "acdm", "llm", "rrum"];
/** The model of a fixture's name, which cdm.R writes in lower case. */
const cdmModel = (name: string): CdmModel => {
  const model = MODELS.find((candidate) => candidate === name);
  if (model === undefined) throw new Error(`Unknown model ${name}.`);
  return model;
};
const models = (model: Fit["model"]): CdmModel | CdmModel[] =>
  typeof model === "string" ? cdmModel(model) : model.map(cdmModel);
const qMatrix = Object.fromEntries(
  Object.entries(ecpe.qmatrix).flatMap(([id, attributes]) => (attributes === null ? [] : [[id, attributes]])),
);
const matrix = buildResponseMatrix(ecpe.responses, { itemIds: [...ecpe.itemIds] });

const results = {
  source:
    "calibrateCdm of GDINA's realdata_ECPE, 2,922 persons, 28 items and 3 attributes, to a tolerance of 1e-10, against GDINA's A-CDM, LLM, R-RUM and a per-item mix of all six models.",
  sizes: { persons: ecpe.responses.length, items: ecpe.itemIds.length, attributes: 3 },
  tolerances: TOLERANCES,
  fits: [] as object[],
};
let missed = false;

for (const fit of ecpe.fits) {
  const { value: result, seconds } = timed(() =>
    calibrateCdm(matrix, qMatrix, { model: models(fit.model), tolerance: 1e-10, maxIterations: 10_000 }),
  );
  let parameters = 0;
  result.items.forEach((item, j) => {
    item.probabilities.forEach((p, l) => {
      parameters = Math.max(parameters, Math.abs(p - (fit.items[j]?.[l] ?? Number.NaN)));
    });
    item.coefficients.forEach((value, i) => {
      parameters = Math.max(parameters, Math.abs(value - (fit.coefficients[j]?.[i] ?? Number.NaN)));
    });
    parameters = Math.max(
      parameters,
      Math.abs(item.guessing - (fit.guessing[j] ?? Number.NaN)),
      Math.abs(item.slip - (fit.slip[j] ?? Number.NaN)),
    );
  });
  result.profiles.forEach(({ proportion }, c) => {
    parameters = Math.max(parameters, Math.abs(proportion - (fit.proportions[c] ?? Number.NaN)));
  });
  const logLikelihood = result.logLikelihood - fit.logLikelihood;
  const criteria = Math.max(
    Math.abs(result.aic + 2 * result.logLikelihood - (fit.aic + 2 * fit.logLikelihood)),
    Math.abs(result.bic + 2 * result.logLikelihood - (fit.bic + 2 * fit.logLikelihood)),
  );
  const within =
    result.converged &&
    parameters <= TOLERANCES.parameters &&
    logLikelihood >= -TOLERANCES.logLikelihood &&
    criteria <= TOLERANCES.criteria;
  missed ||= !within;
  results.fits.push({
    model: fit.model,
    parameters,
    logLikelihoodAboveGdina: logLikelihood,
    criteria,
    within,
    iterations: result.iterations,
    seconds,
  });
  writeResults(import.meta.url, results);
}

if (missed) {
  console.error("A fit missed a tolerance: see results/cdm-additive-ecpe.json.");
  process.exitCode = 1;
}
