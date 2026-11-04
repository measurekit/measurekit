# @measurekit/core

The shared foundation of the measurekit packages: errors, argument checks, a seeded random number generator and numerics.
Install @measurekit/psychometrics or @measurekit/knowledge-tracing.

It holds the errors the packages throw, the checks of their arguments, a seeded random number generator, the numerics and
the types that both packages use. [`@measurekit/psychometrics`](https://measurekit.wakan.dev/_measurekit/psychometrics/)
and [`@measurekit/knowledge-tracing`](https://measurekit.wakan.dev/_measurekit/knowledge-tracing/) depend on it and
re-export `createRandom`, `Random`, `DataError` and `MeasurekitError`, so their users do not import it themselves.

```bash
npm install @measurekit/core
```

## Quick start

A check throws `DataError` with a sentence that names the value and what was expected, such as
`A seed must be a safe integer, found 1.5.`

```ts
import { createRandom, DataError } from "@measurekit/core";

const random = createRandom(42);
random.uniform(); // => 0.66, a draw from [0, 1), the same for every run with seed 42
let refusal: unknown;
try {
  createRandom(1.5);
} catch (error) {
  refusal = error;
}
refusal instanceof DataError; // => true
```

## Architecture

```mermaid
flowchart BT
  C["@measurekit/core<br/>DataError, checks, createRandom, numerics"]
  P["@measurekit/psychometrics"] --> C
  K["@measurekit/knowledge-tracing"] --> C
```

It has no dependencies, reads no clock, network or environment, and runs in Node.js, browsers, Deno, Bun and edge runtimes
such as Cloudflare Workers.

## API

| Group    | Exports                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| -------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Errors   | `MeasurekitError`, the base class; `DataError`, for invalid input                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| Random   | `createRandom(seed)`, a seeded xoshiro128\*\* generator, and its interface `Random`; `shuffle`, a uniform permutation in place                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| Checks   | `describe`; `isNonEmptyString`, `isPositive`, `isRecord`, `isUnitInterval`; `checkRecord`, `checkArray`, `checkOptions`, `checkMethod`, `checkRandom`, `checkBoolean`, `checkChoice`, `checkFinite`, `checkPositive`, `checkNonNegative`, `checkPositiveInteger`, `checkIntegerFrom`, `checkUnitInterval`, `checkSeed`, `checkStopping`                                                                                                                                                                                                                                    |
| Numerics | `normalQuantile`, `logistic`, `logLogistic`, `logisticPair`, `normalizeLogs`, `cholesky`, `bfgs` with `GradientFunction`, `ValueAndGradient`, `OptimizerOptions` and `OptimizationResult` for minimization, `backtrack` with `BacktrackOptions` and `ARMIJO` for line searches, `stepAlong` with `DescentPoint` for a step of a descent, `iterateToConvergence` with `Stopping` and `Convergence`, `logGamma`, `chiSquareUpperTail`, `studentLowerTail`, `kappaOfPairs` with `KappaOptions` and `KappaWeights`, `dot`, `largestChange`, `maxAbs`, `mean`, `multiplyVector` |
| Items    | `LogisticItem`, `ThresholdItem` and `NominalParameters`, the parameters of a logistic, a graded or partial credit, and a nominal response item; `logisticProbabilities`, `gradedProbabilities`, `partialCreditProbabilities`, `nominalProbabilities`, `softmax` and `stridedSoftmax`; `hasValidAsymptotes`, `hasOrderedThresholds`                                                                                                                                                                                                                                         |
| Types    | `QMatrix`, the attributes or skills of each item by item id; `ThetaDistribution`, nodes of theta with weights that sum to 1                                                                                                                                                                                                                                                                                                                                                                                                                                                |

## Supported

The routines are validated by the fixtures of the packages that use them: the normal quantile, the logistic function and
the chi-square upper tail against R's `qnorm`, `plogis` and `pchisq`, and the kappa of pairs against `irr::kappa2`, within
the tolerances of the [psychometrics](https://measurekit.wakan.dev/_measurekit/psychometrics/validation/) and
[knowledge tracing](https://measurekit.wakan.dev/_measurekit/knowledge-tracing/validation/) validation. The tests of core
cover their edge cases.

[Supported](https://measurekit.wakan.dev/_measurekit/core/supported/) lists what the package holds and what each part is
checked against, and [evidence](https://measurekit.wakan.dev/_measurekit/core/evidence/) what shows that it is right.

## Why the numerics are written here

The numerics are written here rather than taken from `ml-matrix` 6.15.1 or the `@stdlib/stats-base-dists-*` packages
0.3.1, the libraries closest to them, measured on the same inputs on 2026-11-04, Node.js 26, Apple M5:

| Need                                                                                                        | `@measurekit/core`                                        | `ml-matrix` 6.15.1                                                                      | `@stdlib` 0.3.1                                                                                              |
| ----------------------------------------------------------------------------------------------------------- | --------------------------------------------------------- | --------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| Cholesky factor of a correlation matrix of order 25 to 200                                                  | 0.0034 to 0.32 ms, on one `Float64Array` by rows          | The same factor in 3.3 to 4.2 times the time, on a matrix of one `Float64Array` per row | None                                                                                                         |
| Normal quantile of 10⁶ probabilities                                                                        | 5.4 ms; R's `qnorm` to the last digit at 1e-12 and 1e-300 | None                                                                                    | 15.9 ms; 3.1e-6 from `qnorm` at 1e-12, and `-Infinity` at 1e-300                                             |
| Descent loops that report `converged` and `iterations`: BFGS, Armijo's line search, a fixed-point iteration | `bfgs`, `backtrack`, `iterateToConvergence`               | None                                                                                    | None                                                                                                         |
| Form                                                                                                        | ESM, no dependency, synchronous                           | An ESM entry over a CommonJS build, two dependencies                                    | CommonJS; the distribution function and the quantile of the normal bring 148 `@stdlib` packages and 9 others |

Each routine is held to R within the tolerances of the libraries' validation, which a dependency would have to meet as
well. What one library alone needs, such as an eigendecomposition, stays in its own `math/`
([psychometrics](https://measurekit.wakan.dev/_measurekit/psychometrics/architecture/#why-the-numerics-are-written-here)).

## Documentation

[Choosing](https://measurekit.wakan.dev/_measurekit/core/choosing/) says when the package fits, when another library fits better, and what each has
that the package lacks.

[Performance](https://measurekit.wakan.dev/_measurekit/core/performance/) gives the speed of the numerics against base R
and stats, and what the entry point adds to a bundle.

## License

[Apache-2.0](https://github.com/measurekit/measurekit/blob/main/packages/core/LICENSE). Copyright Fariz Wakan. See
[NOTICE](https://github.com/measurekit/measurekit/blob/main/packages/core/NOTICE).
