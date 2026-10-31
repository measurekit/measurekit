# AGENTS.md

Conventions for people and coding agents working in this repository. Read this before changing anything. Most
rules are enforced by `pnpm lint` (Biome and cspell), `pnpm typecheck`, `pnpm knip` and `pnpm docs:api`; the rest
are checked in review.

measurekit holds two libraries for JavaScript and TypeScript, which do not import each other, and the core that they
share:

- `@measurekit/psychometrics` is a psychometrics library, validated against R.
  [docs/psychometrics/architecture.md](docs/psychometrics/architecture.md) states its design principles.
- `@measurekit/knowledge-tracing` is a knowledge tracing library, validated against pyBKT and R.
  [docs/knowledge-tracing/architecture.md](docs/knowledge-tracing/architecture.md) states its design principles and the
  learner model interface.
- `@measurekit/core` holds what both libraries use: their errors, the checks of their arguments, the random number
  generator and common numerics. It is the only dependency of the libraries, and it has none.

Follow the principles of the package you change. The rules below apply to every package unless a section names one.

## Terms

Use these words with these meanings in code, comments and docs.

### Psychometrics

| Term               | Meaning                                                                                                                    |
| ------------------ | -------------------------------------------------------------------------------------------------------------------------- |
| person             | One test taker: a row of a response matrix.                                                                                |
| item               | One scored question or task: a column of a response matrix.                                                                |
| response           | A person's score on an item: an integer from 0, or `null` when missing.                                                    |
| choice             | The option label a person chose on a multiple-choice item, before scoring.                                                 |
| option             | One alternative of a multiple-choice item. The **key** is the correct option; the others are distractors.                  |
| total score        | The sum of a person's responses.                                                                                           |
| complete case      | A person who answered every item.                                                                                          |
| p-value            | The mean score of a dichotomous item. Never a significance level.                                                          |
| upper, lower group | The persons with the highest and lowest total scores, `groupFraction` of them each.                                        |
| category           | One possible response to an item: an integer from 0. A dichotomous item has categories 0 and 1.                            |
| theta              | A person's ability: on the N(0, 1) scale, and on the Rasch scale, a = 1, in joint maximum likelihood.                      |
| IRT parameters     | mirt's IRT form, D = 1: `a` discrimination, `b` difficulty or thresholds, `c`, `d` asymptotes. See `interop/`.             |
| cut score          | The total score, or theta, at or above which a person meets a standard.                                                    |
| attribute          | A skill or piece of knowledge a cognitive diagnosis model classifies as mastered or not; knowledge tracing calls it skill. |
| profile, Q-matrix  | A profile marks each attribute mastered (1) or not (0); a Q-matrix lists each item's attributes by item id.                |
| record             | One response in long format, `{ person, item, response }`, for sparse data.                                                |
| group              | A set of persons with its own distribution of theta in calibration; in DIF, `"reference"` or `"focal"`.                    |
| testlet            | Items that share a stimulus, such as a reading passage; the testlet model adds one effect per person and testlet.          |
| pool, form, panel  | A pool holds the items an adaptive test selects from; a form is a fixed set of items; a panel has stages of modules.       |
| response time      | Seconds a person spent on an item, aligned with the response: `times` of a matrix, `time` of a record.                     |
| subject, rater     | In rater agreement, what is rated, such as an essay, and who rates it: a row and a column of ratings.                      |

### Knowledge tracing

| Term                | Meaning                                                                                                                         |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| learner             | One person who practices; a model keeps one state per learner.                                                                  |
| item                | One question or task a learner attempts.                                                                                        |
| skill               | A knowledge component that an item practices; psychometrics calls it attribute. An item practices zero or more skills.          |
| attempt             | A learner's try at an item: learner, item, skills and time. Models predict attempts.                                            |
| event               | An attempt with its outcome: one entry of a log.                                                                                |
| outcome             | From 0 to 1: 1 correct, 0 incorrect, and between them partial credit.                                                           |
| log                 | Events in time order.                                                                                                           |
| Q-matrix            | The skills of each item, by item id: the shape of the `QMatrix` of psychometrics, so one moves between the packages.            |
| opportunity         | The number of times a learner has practiced a skill, the current event included.                                                |
| state               | One learner's knowledge as a model tracks it: plain data, updated one event at a time.                                          |
| parameters          | What a model estimates from a log, such as item difficulties: plain data.                                                       |
| prediction          | P(correct) for an attempt, made before its outcome is known.                                                                    |
| fold                | A training log and a test log taken from one log.                                                                               |
| ability, difficulty | θ and d on the logit scale: P(correct) = logistic(θ - d). Elo estimates them; IRT's θ is the position on a calibration's scale. |
| mastery             | P(the learner has mastered a skill), for models that define it: P(L) in BKT.                                                    |
| prior group         | Learners who share a BKT prior for each skill.                                                                                  |
| proficiency         | θ, a learner's general ability in a logistic model: none, a fixed effect or a random intercept.                                 |
| feature             | A logistic model's function of a learner's history with a skill before an attempt.                                              |
| band                | The range of predicted success that practice aims at when ranking items.                                                        |
| knowledge state     | In KST, the set of skills or items a learner masters; a structure lists the possible ones.                                      |
| fringe              | The elements a knowledge state can lose (inner) or gain (outer) while staying a state.                                          |

## Layout

```
packages/core/                the published package @measurekit/core
  src/index.ts          re-exports everything below
  src/errors.ts         MeasurekitError, DataError
  src/checks.ts         checks of one value: describe; the predicates isNonEmptyString, isPositive, isRecord,
                        isUnitInterval; and checkRecord, checkArray, checkOptions, checkMethod, checkRandom,
                        checkBoolean, checkChoice, checkFinite, checkPositive, checkNonNegative,
                        checkPositiveInteger, checkUnitInterval, checkSeed, checkStopping
  src/random.ts         Random and createRandom
  src/normal.ts         normalQuantile
  src/logistic.ts       logistic, logLogistic
  src/normalize-logs.ts normalizeLogs
  src/cholesky.ts       cholesky
  src/vectors.ts        dot, maxAbs, mean
  test/<file>.test.ts   one file per concern; the numerics are validated by the fixtures of the packages
packages/psychometrics/       the published package @measurekit/psychometrics
  src/index.ts          re-exports every module, and DataError, MeasurekitError and createRandom of core
  src/checks.ts         checkFunction, the one check of a value that only this package uses
  src/data/             response matrices and records, scoring
  src/math/             numerics
  src/ctt/              classical test theory
  src/irt/              item response theory: models, calibration, ability estimation, information, reliability
  src/sim/              simulation of thetas, item parameters and responses
  src/cat/              adaptive and multistage tests, their simulation, exposure control, form assembly
  src/cdm/              cognitive diagnosis: G-DINA, DINA, DINO, Q-matrix validation, nonparametric classification
  src/fit/              item and person fit, local dependence
  src/dif/              differential item functioning
  src/equating/         linking of IRT scales, observed-score equating with and without common items, IRT true-score
                        and observed-score equating, bootstrap and delta-method standard errors, the anchor screen,
                        presmoothing
  src/standard-setting/ cut scores by the modified Angoff, Bookmark and Hofstee methods; classification accuracy
  src/rater/            rater agreement: kappa, intraclass correlations
  src/response-time/    response times: the lognormal model, speed, aberrant times, rapid guessing
  src/interop/          conversion of item parameters to and from mirt
  test/<module>.test.ts one file per src folder; fixtures/ holds R results
packages/knowledge-tracing/   the published package @measurekit/knowledge-tracing
  src/index.ts          re-exports every module, the model interface, and DataError, MeasurekitError and
                        createRandom of core
  src/model.ts          LearnerModel and ModelFit, the interface every model implements, and checkModel
  src/data/             events, logs, Q-matrices, learner checks, splits, the replay of a log through a model
  src/math/             numerics
  src/evaluate/         metrics, calibration, cross-validation, model comparison, learning curves
  src/elo/              learner-item and concept-specific Elo
  src/bkt/              Bayesian knowledge tracing with pyBKT's extensions
  src/logistic/         AFM, PFA and LKT features, with fixed or random proficiency
  src/irt/              online IRT: the posterior of ability on a grid, from calibrated item parameters
  src/kst/              knowledge structures, the basic local independence model, IITA, adaptive assessment
  src/practice/         the next item by predicted success, mastery criteria, attempts to a streak
  src/sim/              outcomes simulated from any learner model
  test/<module>.test.ts one file per src folder; fixtures/ holds the reference results
docs/<package>/         guides of each package; TypeDoc publishes them with the API reference
validation/<package>/   R and Python scripts that write the fixtures; validation/README.md lists them
examples/<package>/     runnable examples, which import the package by name
biome/                  Biome GritQL plugins
.changeset/             the pending release: one changeset per package
```

Each package folder also holds `README.md` (the page of the package on npm), `LICENSE`, `NOTICE`, `package.json`,
`typedoc.json` and its `tsconfig` files.

## Commands

```bash
pnpm install
pnpm lint             # Biome and cspell; `pnpm format` fixes formatting
pnpm typecheck        # tsc over the sources, tests and examples of every package
pnpm test             # Vitest, one project per package; `pnpm test --project psychometrics` runs one
pnpm knip             # unused files, exports and dependencies
pnpm build            # every package: tsc -b after deleting dist/, so that no stale output ships
pnpm check:packages   # builds, then publint and Are the Types Wrong on each packed package
pnpm docs:api         # TypeDoc into docs/api, a section per package; fails on undocumented exports
```

`pnpm --filter @measurekit/psychometrics <script>` runs a script of one package. Run all of the commands before
finishing a change. Development needs Node.js 22.18+ and pnpm 12.

## Releases

Changesets version and publish the packages, each on its own version. A release has one changeset per package in
`.changeset/`; changes made before the release extend it. To release, a pull request from a `changeset-release/` branch
runs `pnpm changeset version`, which bumps the versions, writes `CHANGELOG.md` and consumes the changesets. After it is
merged, the maintainer runs the Release workflow by hand: it builds and publishes the versions not yet on npm. No
workflow opens pull requests, and Dependabot version updates are off. Changesets, docs and TSDoc describe current
behavior only, without "now", "no longer", "renamed" or "previously".

## Dependencies

- No package uses a platform API: no `node:*`, `fetch`, DOM or `process`. `@measurekit/core` has no dependencies, and
  the libraries depend on it alone. The libraries do not import each other.
- A module imports only the modules in its row; every module may import `@measurekit/core`. Biome enforces the rows,
  and imports never form cycles.
- Code goes into `@measurekit/core` only if both libraries use it. What one library uses stays in that library, so that
  core does not become a place for everything. A generic routine that both use, such as a Cholesky factorization, is
  written once, in core.
- Code outside `packages/` imports a package by name.

Psychometrics: every module of `cat/`, `equating/`, `irt/` and `sim/` may also import `checks.ts`.

| Module                     | Imports                               |
| -------------------------- | ------------------------------------- |
| `math/`, `data/`, `rater/` | none                                  |
| `ctt/`                     | `data/`, `math/`                      |
| `irt/`                     | `data/`, `math/`                      |
| `interop/`                 | `irt/`                                |
| `cdm/`                     | `data/`, `irt/`, `math/`              |
| `fit/`                     | `data/`, `irt/`, `math/`              |
| `response-time/`           | `data/`, `irt/`, `math/`              |
| `equating/`                | `irt/`, `math/`                       |
| `sim/`                     | `data/`, `irt/`, `cdm/`, `math/`      |
| `dif/`                     | `data/`, `irt/`, `equating/`, `math/` |
| `standard-setting/`        | `irt/`, `equating/`, `math/`          |
| `cat/`                     | `irt/`, `sim/`, `cdm/`, `math/`       |

Knowledge tracing: the model families never import each other, nor does anything import `evaluate/`, `practice/` or
`sim/`: they use models only through `LearnerModel`.

| Module                                                                        | Imports                      |
| ----------------------------------------------------------------------------- | ---------------------------- |
| `math/`                                                                       | none                         |
| `data/`                                                                       | `math/`                      |
| `model.ts`                                                                    | the types of `data/`         |
| `elo/`, `bkt/`, `logistic/`, `irt/`, `kst/`, `evaluate/`, `practice/`, `sim/` | `data/`, `math/`, `model.ts` |

## Naming

- Files: kebab-case. One concept per file, named after it. Group files in folders by module.
- Values and functions: camelCase. Module-level constant data (thresholds, coefficient tables): SCREAMING_SNAKE.
- Types: PascalCase. Type parameters: `T` or `T` plus a name.
- Acronyms are words: `kr20`, `bfgs`, `auc`, `rmse`. Other statistics are named in full: `standardErrorOfMeasurement`,
  `logLoss`, not `sem`, `ll`. `QMatrix` keeps the field's spelling, with a Biome exception.
- Errors are `DataError` and its base class `MeasurekitError`, both of `@measurekit/core`, which the
  libraries re-export. A package that needs an error of its own extends `MeasurekitError`.
- Function options are `<Name>Options`; options shared by several functions are named after what they configure
  (`GroupOptions`), and a method's are named after it (`NextQuestionOptions`). Results are nouns: `ItemStatistics`,
  `SplitHalf`, `CrossValidation`, `LearningCurve`.
- Psychometrics labels are string unions in kebab-case (`"very-good"`); their thresholds are `<NAME>_THRESHOLDS`.
- A knowledge tracing model family has a factory named after it, `elo(options)`, that returns a `LearnerModel`, or
  a type extending it when the family adds members: `EloModel`. Its files: `<family>.ts` for the factory, `state.ts` and
  `parameters.ts` for the plain data and their checks, `estimation.ts` for a fit's numerics.
- Iterative methods take `tolerance` and `maxIterations` and report `converged` and `iterations`; random starts take
  `restarts` and `seed`. Defaults are named constants, such as `DEFAULT_TOLERANCE`.

## Code

- TypeScript strict, ESM, named exports only, `.js` extensions in relative imports, `import type` for types.
- No `as` casts except where untyped data enters and is validated next. No non-null assertions.
- Public functions validate their input and throw `DataError` with a full sentence that says what is wrong and
  what is expected, e.g. `A seed must be a safe integer, found 1.5.` Input may come from plain JavaScript or JSON, so
  every argument is checked for its type before it is read, once, where the public function reads it, with a `check`
  function of `@measurekit/core`: options with `checkOptions`, arrays with `checkArray`, objects with `checkRecord`,
  an object with a method with `checkMethod`, a function with `checkFunction` (`checks.ts` of psychometrics), a
  `Random` with `checkRandom`, a number with `checkFinite`, `checkPositive`, `checkNonNegative`,
  `checkPositiveInteger` or `checkUnitInterval`, `tolerance` and `maxIterations` with `checkStopping`, a seed
  with `checkSeed`, a boolean with `checkBoolean` and a string option with
  `checkChoice`: `<Label> must be a positive number, found <value>.` An option's label is `The option <name>`. A
  field of a record is tested with a predicate of core or the language, such as `isUnitInterval` or
  `Number.isFinite`, and the message names the record; a predicate of core is never written out by hand. Other
  messages show values with `describe`. The messages read alike in both libraries, as one set of checks writes them.
- Psychometrics checks items with `checkItem` or `checkItemList` and an IRT model with `checkModel`, all of
  `irt/items.ts`. Knowledge tracing checks a learner model passed in with `checkModel` of `model.ts`, lists of attempts
  with `checkAttempts`, and the model's predictions with `checkUnitInterval`. A field's message names its record:
  `Event 3 needs an outcome from 0 to 1, found 2.`
- Options and optional fields are destructured with their defaults, `const { tolerance = DEFAULT_TOLERANCE } =
  options`, so that a value set to `null` is checked and rejected, never taken as absent.
- Missing responses are `null`. Each function documents which persons a statistic uses.
- A statistic undefined for the data is `null` in tables and a `DataError` from single-value functions. Never
  return `NaN`. Iterative methods report `converged` and `iterations` instead of throwing.
- Knowledge tracing models are immutable: `update` returns a new state and `fit` a new model. States and parameters are
  plain data that JSON serializes; model methods check them, as they may come from storage.
- Never use `Math.random`; take a `Random`. The GritQL plugin `biome/no-math-random.grit` makes `pnpm lint` fail on
  `Math.random` anywhere in `packages/*/src`.
- Numerics work on `Float64Array` and do not allocate in inner loops.
- Numerics stay defined at extreme parameters. A weight that underflows to 0 is skipped before it divides; a loop that
  grows or shrinks a value until a test passes also stops when the value is not finite. In psychometrics, a response's
  probability enters a log-likelihood as at least `MIN_PROBABILITY` of `irt/probability.ts`, so that no likelihood is
  0; knowledge tracing throws `DataError` for outcomes of probability 0.
- A number with a meaning is a named constant with its source, such as the reference's name for it.
- No speculative options, abstractions or modules. Add a method only when a real use needs it, with its validation.

## Comments and docs

- Every export has a TSDoc comment: types and constants a noun phrase, functions a third-person verb phrase
  ("Returns ...", "Builds ..."), properties a short phrase. Formulas use plain notation: `k / (k - 1)`. TypeDoc renders
  the comments, so they use its tags:
  - The first sentence is the summary. A longer explanation, such as a method, a formula or the edge cases, follows
    under `@remarks`; a description of more than three lines has one.
  - `@defaultValue` gives the default of every option and every parameter that has one, as code (`@defaultValue
    \`1e-4\``) or, when the default is a behavior, a short phrase. The summary never states a default. A parameter
    whose initializer is in the signature shows it in TypeDoc; one whose default is computed in the body has
    `@defaultValue` on the function, naming the parameter. An option whose default differs between functions that take
    it has an options type of its own.
  - `@throws` lists the errors.
  - `@see {@link https://doi.org/... | Author (year)}` cites the literature, one tag per work, with the DOI where the
    work has one. The summary and `@remarks` name no year.
  - `@example` goes only on a public entry point where an example truly helps. It shows its imports, and the values in
    its comments come from running it.
  - No other tags and no boilerplate.
- `//` comments only explain why something non-obvious is done, or cite a source. No commented-out code.
- US English. Docs state facts in short sentences; prefer a table or list to prose. Code samples are complete: they
  show their imports. Values shown in samples come from running them. No marketing language, emojis or filler. A
  section of a source is cited in words ("section 3.1").
- Each document has one purpose: a package's `README.md` for its users, `docs/<package>/` for its guides,
  `CONTRIBUTING.md` for the contribution process, `AGENTS.md` for conventions, `validation/README.md` for running the
  reference scripts.

## Validation and tests

- Every statistic has a reference: a function of an established R or Python package, or a base R computation written
  out in the script. The script writes inputs and expected values to
  `packages/<package>/test/fixtures/<module>-<case>.json`; the test compares within the tolerance in
  `docs/<package>/validation.md`. Never type expected values by hand except for edge cases small enough to derive
  exactly.
- The numerics of `@measurekit/core`, such as `normalQuantile` and `logistic`, are validated by the fixtures of the
  packages that use them; core's tests cover its edge cases and the rest of its behavior.
- Psychometrics `sim/` has no R reference, as R's generator differs from `createRandom`: seeded tests check its
  distributions, and a calibration recovers the items it simulates.
- A knowledge tracing model without a package of record, such as Elo, has its equations written out in R and a seeded
  simulation recovery study, with thresholds fixed before the simulation is first run.
- Use public datasets (mirt, psych, difR, pyBKT) or simulations with a fixed seed.
- One test file per source folder, named after it: `ctt.test.ts` for `src/ctt/`. `test/helpers.ts` holds shared
  helpers, such as `expectClose` in psychometrics, which compares with a reference value, and `online` in knowledge
  tracing, which runs a log through any model as an app would. A file's `describe` blocks follow the order of the
  folder's source files or the sections of the module's guide, with a closing `properties` block in psychometrics;
  within a block, the comparisons with the reference and the main behavior come first, then the options, then errors and
  edge cases.
- Every public function has a test of input of other types, as JavaScript or JSON pass it: `null` options and
  arguments, a string for a number, an object for an array. Pass such input with `JSON.parse`, not a cast.
- Properties that hold for any input (alpha at most 1, probabilities from 0 to 1, invariance to the order of persons,
  states left unchanged by `update`, states that continue identically after a JSON round trip, reproducible seeds) are
  tested with fast-check.
- Edge cases are tested for every function: constant items, persons and items without responses, one item, two persons;
  an empty log, one learner, events out of time order, items and skills that are unknown, outcomes all correct or all
  wrong, and each model's limits, such as BKT parameters at their bounds or a posterior mode beyond the IRT grid.
- Test names are plain sentences about behavior: `it("shares the places at a tied boundary equally")`.
- A bug fix comes with a test that fails without it, in the test file of the fixed folder.

## Where things go

- A statistic: a file in its module folder, exported from the module's `index.ts`, with a row in the module's guide
  in `docs/<package>/`, a reference in `validation/<package>/<module>.R` (or `.py`), a fixture, tests, and a tolerance
  row in `docs/<package>/validation.md` if it needs its own.
- A knowledge tracing model family: a module whose factory returns a `LearnerModel`, with its state and parameters as
  plain data and their checks, its steps for `replay` in `fit`, and the property tests of the interface.
- A module: a folder in `src/` with `index.ts`, a subpath in the package's `package.json` `exports`, an entry point in
  its `typedoc.json`, `export *` in `src/index.ts`, a test file, a guide in `docs/<package>/`, a reference script if it
  computes statistics, import rules in `biome.json` and the table above, and a row in the API table of the package's
  `README.md`.
- A new package: a folder in `packages/` with the files listed under Layout, a project in `vitest.config.ts`, an
  entry in the root `typedoc.json`, a changeset, a row in the package table of the root `README.md` and, for a library,
  `docs/<package>/`.
- Numerics shared by modules: `math/`, internal unless users need them; numerics both libraries use: `@measurekit/core`.
- Checks shared by modules: of one value in `@measurekit/core` or, if only that library uses it, `checks.ts` of the
  library; otherwise with the data they check, such as
  `data/dichotomous.ts` and `irt/items.ts` in psychometrics, `data/event.ts` and `model.ts` in knowledge tracing.
- An example: `examples/<package>/<name>.ts`, importing the package by name, with a header comment that says what it
  shows and how to run it.
