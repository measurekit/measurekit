# Architecture

## Principles

- **Correct first.** Every statistic matches a named R reference within a documented tolerance
  ([validation](validation.md)), and the fixtures are committed.
- **Portable.** No platform API (`node:*`, `fetch`, DOM, `process`) and one runtime dependency,
  `@measurekit/core`, which has none and holds the errors, argument checks and random number generator that
  both packages share. ESM, strict TypeScript.
- **Deterministic.** The library never calls `Math.random`, which a Biome plugin enforces. Randomness comes from a
  `Random` passed in, so a seed reproduces a result. Results are identical within one JavaScript engine version:
  `Math.log`, `Math.exp` and the other `Math` functions may differ in their last bits between engines and between
  versions of one engine, such as V8's, which can move results by a few units in the last place.
- **Data in, data out.** Functions take plain arrays and objects and return plain, typed result objects. No I/O, no
  logging, no localized text: labels are string unions with documented, configurable thresholds.
- **Defined results.** Invalid input throws `DataError`, also input of another type than documented, as plain
  JavaScript or JSON may pass it: `null` or a number for an array or for options, a string for a number, or a model,
  rule, estimator or random number generator that is not an object or function of its interface. An option set to
  `null` is invalid, not absent. A statistic that is undefined for the data is `null` in a table and a `DataError`
  from a function that returns only that statistic; results never contain `NaN`. Iterative methods report
  convergence instead of throwing.
- **Extensible.** Where methods have alternatives, such as models or estimators, they are interfaces, and built-ins
  implement the same interfaces as user code.
- **Lean.** A method is added when a real use needs it, and ships validated. No speculative options.

## Modules

Each module is a folder of `packages/psychometrics/src/` with an `index.ts` that lists its public API.
[AGENTS.md](../../AGENTS.md#layout) shows the layout of the repository and
the [README](../../packages/psychometrics/README.md#api) the contents of each module. The package exports `.` and
one subpath per module: `@measurekit/psychometrics/cat`,
`@measurekit/psychometrics/cdm`, `@measurekit/psychometrics/change`, `@measurekit/psychometrics/ctt`,
`@measurekit/psychometrics/data`, `@measurekit/psychometrics/dif`, `@measurekit/psychometrics/dimensionality`,
`@measurekit/psychometrics/equating`, `@measurekit/psychometrics/explanatory`, `@measurekit/psychometrics/fit`,
`@measurekit/psychometrics/generalizability`, `@measurekit/psychometrics/irt`, `@measurekit/psychometrics/mokken`,
`@measurekit/psychometrics/multidimensional`, `@measurekit/psychometrics/parameterization`,
`@measurekit/psychometrics/rater`, `@measurekit/psychometrics/response-time`, `@measurekit/psychometrics/scaling`,
`@measurekit/psychometrics/sim` and `@measurekit/psychometrics/standard-setting`. Public names are unique across modules, so `.` re-exports all of them.
Only the random number generator is public in `math/`.

## Dependencies between modules

Modules form layers; a module imports only from lower layers, never from its own or a higher one.

| Layer | Modules                                                                            |
| ----- | ---------------------------------------------------------------------------------- |
| 1     | `@measurekit/core`, `checks.ts`                                                    |
| 2     | `math/`, `data/`                                                                   |
| 3     | `ctt/`, `change/`, `irt/`, `dimensionality/`, `generalizability/`, `mokken/`       |
| 4     | `cdm/`, `fit/`, `equating/`, `response-time/`, `explanatory/`, `multidimensional/` |
| 5     | `sim/`, `dif/`, `standard-setting/`, `scaling/`, `rater/`, `parameterization/`     |
| 6     | `cat/`                                                                             |

Analysis modules (`ctt/`, `change/`, `irt/`, `dimensionality/`, `generalizability/` and `mokken/`) build on `data/` and
`math/` or on nothing, not on each other. `sim/` builds on `irt/`, `cdm/` and `multidimensional/` to simulate their models, `parameterization/`
on `irt/` and `multidimensional/` to convert their items, `cat/` on `sim/` for the same response simulation, on `cdm/` for
the profile estimates of diagnostic tests and on `multidimensional/` for the items, estimates and information of
multidimensional tests, `dif/` and `standard-setting/` on `equating/` for linking and score distributions, `scaling/` on
`equating/` for the linear transformation of a theta scale, and `rater/` on `irt/` for the Rasch model of its many
facets and on `fit/` for the fit of their elements. `explanatory/` builds on `irt/` for the calibrated items it
regresses, and `multidimensional/` on `irt/` for its priors, compressed responses and parameter bounds.
[AGENTS.md](../../AGENTS.md#dependencies) lists the imports each module may use; Biome enforces them, the absence of
Node modules and platform globals in `src/`, and that code outside `packages/` imports `@measurekit/psychometrics` by
name.

## Numerics

- Numerics work on `Float64Array` and do not allocate in inner loops.
- Functions in `math/` assume valid input; public functions validate before calling them.
- Scores are integers, so sums are exact: covariances come from Σxy, Σx and Σy, a variance is exactly 0 when the
  values are equal, and no statistic depends on the order of persons. Zero-variance checks compare with 0.
- Calibration estimates bounded parameters on an unbounded scale (logit between two bounds), so the optimizers work
  without constraints.
- Calibration stores the answered responses by person, with each response's item index, whether the input is a matrix
  or records: memory and E-step time grow with the responses. Each item's log-probabilities at the nodes sit in one
  array, so a response adds its column with a fixed stride.
