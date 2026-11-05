# AGENTS.md

Conventions for people and coding agents working in this repository. Read this before changing anything. `pnpm lint`,
`pnpm check:standards`, `pnpm check:duplication`, `pnpm typecheck`, `pnpm knip` and `pnpm docs:api` enforce most of the rules; this file holds the
ones they cannot.

measurekit holds two libraries for JavaScript and TypeScript, which do not import each other, and the core that they
share. Follow the principles of the package you change.

- `@measurekit/psychometrics`: psychometrics, validated against R and RSMTool. Principles: [architecture](docs/psychometrics/architecture.md).
- `@measurekit/knowledge-tracing`: knowledge tracing, validated against pyBKT and R. Principles and the learner model
  interface: [architecture](docs/knowledge-tracing/architecture.md).
- `@measurekit/core`: what both use, their errors, the checks of their arguments, the random number generator and the
  common numerics. It has no dependencies, and the libraries depend on it alone.

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
| IRT parameters     | mirt's IRT form, D = 1: `a` discrimination, `b` difficulty or thresholds, `c`, `d` asymptotes. See `parameterization/`.    |
| cut score          | The total score, or theta, at or above which a person meets a standard.                                                    |
| attribute          | A skill or piece of knowledge a cognitive diagnosis model classifies as mastered or not; knowledge tracing calls it skill. |
| profile, Q-matrix  | A profile marks each attribute mastered (1) or not (0); a Q-matrix lists each item's attributes by item id.                |
| record             | One response in long format, `{ person, item, response }`, for sparse data.                                                |
| group              | A set of persons with its own distribution of theta in calibration; in DIF, `"reference"` or `"focal"`.                    |
| person weight      | A person's survey weight in calibration and norms, `personWeights`: never a node's weight of theta (`weights`).            |
| bundle             | Items that DIF studies as one, by the sum of their scores, such as those of a reading passage.                             |
| occasion           | One administration of the same items to the same persons; a longitudinal calibration gives each its own trait.             |
| testlet            | Items that share a stimulus, such as a reading passage; the testlet model adds one effect per person and testlet.          |
| pool, form, panel  | A pool holds the items an adaptive test selects from; a form is a fixed set of items; a panel has stages of modules.       |
| content area       | A group of items a blueprint counts, such as a domain or an item type; an item is in several (`contentAreas`).             |
| unit               | An item, or a testlet that has not begun, which an adaptive test administers whole.                                        |
| response time      | Seconds a person spent on an item, aligned with the response: `times` of a matrix, `time` of a record.                     |
| subject, rater     | In rater agreement, what is rated, such as an essay, and who rates it: a row and a column of ratings.                      |
| station            | A task of a performance test, such as an OSCE's: an examiner gives each person a checklist mark and a global rating.       |
| plausible value    | A random draw from a person's posterior distribution of theta; a statistic of several sets of them describes a group.      |
| anchor             | An item taken to work the same in two groups, forms or calibrations: a link's common items, DIF's matching items, `fixed`. |
| link               | The `ScaleTransformation` `{ slope, intercept }` that puts the thetas and items of one calibration on another's scale.     |
| exposure rate      | The share of persons who are administered an item in an adaptive test; knowledge tracing's exposure is another thing.      |

### Knowledge tracing

| Term                | Meaning                                                                                                                         |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| learner             | One person who practices; a model keeps one state per learner.                                                                  |
| item                | One question or task a learner attempts.                                                                                        |
| skill               | A knowledge component that an item practices; psychometrics calls it attribute. An item practices zero or more skills.          |
| attempt             | A learner's try at an item: learner, item, skills, time, and its resource and covariates. Models predict attempts.              |
| event               | An attempt with its outcome and its attempt metadata (hints, duration, attempt number): one entry of a log.                     |
| outcome             | From 0 to 1: 1 correct, 0 incorrect, and between them partial credit; `null` for a learning opportunity with nothing observed.  |
| resource            | What the learn and forget rates of an attempt belong to: a lesson, a video or an item, which is the default.                    |
| exposure            | An event without an outcome at a skill, which logistic models count apart from the events with one.                             |
| log                 | Events in time order.                                                                                                           |
| Q-matrix            | The skills of each item, by item id: the `QMatrix` of core, which psychometrics shares, so one moves between the packages.      |
| opportunity         | The number of times a learner has practiced a skill, the current event included.                                                |
| state               | One learner's knowledge as a model tracks it: plain data, updated one event at a time.                                          |
| parameters          | What a model estimates from a log, such as item difficulties: plain data.                                                       |
| prediction          | P(correct) for an attempt, made before its outcome is known.                                                                    |
| fold                | A training log and a test log taken from one log.                                                                               |
| ability, difficulty | θ and d on the logit scale: P(correct) = logistic(θ - d). Elo estimates them; IRT's θ is the position on a calibration's scale. |
| mastery             | P(the learner has mastered a skill), for models that define it: P(L) in BKT.                                                    |
| prior group         | Learners who share a BKT prior for each skill.                                                                                  |
| proficiency         | θ, a learner's general ability in a logistic model: none, a fixed effect or a random intercept.                                 |
| feature             | A logistic model's function of a learner's history with a skill, an item or all items before an attempt.                        |
| band                | The range of predicted success that practice aims at when ranking items.                                                        |
| knowledge state     | In KST, the set of skills or items a learner masters; a structure lists the possible ones.                                      |
| fringe              | The elements a knowledge state can lose (inner) or gain (outer) while staying a state.                                          |

## Layout

| Path                                      | Holds                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| ----------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `packages/core/`                          | A flat `src/`, a file for each concept: errors, checks, random, normal, logistic, item-probabilities, normalize-logs, cholesky, iteration, line-search, descent, bfgs, chi-square, student-t, kappa, vectors, q-matrix, theta-distribution; `continued-fraction.ts` is not exported                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| `packages/psychometrics/`                 | `src/<module>/` for data, math, ctt, change, dimensionality, irt, multidimensional, cdm, fit, dif, equating, standard-setting, explanatory, mokken, generalizability, scaling, rater, response-time, similarity, parameterization, sim and cat; `src/checks.ts` holds `checkFunction` and `checkOpenUnitInterval`, which only this package uses; `src/index.ts` re-exports `createRandom`, `DataError`, `MeasurekitError` and the types of core that the package's API names, and nothing else                                                                                                                                                                                                                                                                                                                             |
| `packages/knowledge-tracing/`             | `src/<module>/` for data, math, evaluate, elo, bkt, logistic, irt, kst, practice and sim; `src/model.ts` holds `LearnerModel`, `ModelFit` and `checkModel`, which every model implements; `src/index.ts` exports `LearnerModel` and `ModelFit`, and re-exports `createRandom`, `DataError`, `MeasurekitError` and the types of core that the package's API names                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| `packages/<package>/`                     | `src/<module>/index.ts`, the public API of a module, `internal.ts`, what the package shares, and `math/`, which no subpath exports, with `internal.ts` alone; `test/<module>.test.ts`, one for each source folder, with `fixtures/` of reference results; `README.md`, `LICENSE`, `NOTICE`, `package.json`, `typedoc.json` and the tsconfig files                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| `docs/<package>/`                         | The guides, which TypeDoc publishes with the API reference; `supported.md`, `evidence.md` and `choosing.md`; and `performance.md`, the benchmarks, the sizes and the peak memory of the package; core has these pages and no guide                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| `docs/typedoc-seo.ts`, `docs/favicon.svg` | The plugin of TypeDoc that makes the docs site from `scripts/family.ts`, and the icon of the site; `.github/workflows/docs.yml` deploys the site to GitHub Pages and stays disabled until the repository is public                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| `validation/<package>/`                   | The R and Python scripts that write the fixtures, and the TypeScript studies too long for the tests, which write `results/`; `validation/README.md` lists them. `validation/` is a package of the workspace, so that the studies import the packages by name                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| `validation/benchmarks/`                  | The inputs of the benchmarks and the timing of their references                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| `examples/<package>/`                     | Runnable examples; `examples/test/` runs each of them, the samples of the guides and the examples of the TSDoc                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| `compat/`                                 | The packages on Node.js, Deno, Bun and workerd, built as they ship: `scenarios.ts` holds a calibration and a BKT fit, `check.ts` bundles them and compares each runtime's results with Node.js's                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| `scripts/`                                | `check-standards.ts`, `check-rules.ts`, `check-licenses.ts`, `check-release.ts` and `check-tarballs.ts` with their tests; `prepare.ts` installs the git hooks in a clone and skips them in a copy; `bench-harness.ts` is what the benchmarks of the packages share; `entry-sizes.ts` bundles each entry point alone, with code splitting, against the budgets of `size-budgets.ts`; `peak-memory.ts` measures the readers of untrusted input in processes of a bounded heap; `guide-samples.ts` runs the samples of the guides and the examples of the TSDoc, and `stated-values.ts` compares the results they show; `test/workflows.test.ts` holds each job of the workflows to a timeout and each job of `ci.yml` but `check` to the `changes` job, and `test/changes.test.ts` runs that job's program on sample changes |
| `scripts/family.ts`                       | The identity of the family, which the standards check, the docs site and `llms.txt` read                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `biome/`, `.changeset/`                   | GritQL lint rules; the pending release, one changeset for each package                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |

## Commands

| Command                  | Does                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `pnpm install`           | Installs the dependencies and, in a git clone, the git hooks (`scripts/prepare.ts`)                                                                                                                                                                                                                                                                                                                                                                  |
| `pnpm lint`              | Biome and cspell; `pnpm format` fixes formatting                                                                                                                                                                                                                                                                                                                                                                                                     |
| `pnpm check:standards`   | The rules of STANDARDS.md that a script can check, over the whole repository                                                                                                                                                                                                                                                                                                                                                                         |
| `pnpm check:duplication` | jscpd over `packages/*/src`; fails on a repeat that [the table below](#repeats-left-on-purpose) does not list                                                                                                                                                                                                                                                                                                                                        |
| `pnpm typecheck`         | `tsc` over the sources, tests, examples and scripts                                                                                                                                                                                                                                                                                                                                                                                                  |
| `pnpm test`              | Vitest, a project for each package, the scripts and the examples; `--project psychometrics` runs one                                                                                                                                                                                                                                                                                                                                                 |
| `pnpm test:coverage`     | The tests of each package, held to the coverage floor of `vitest.config.ts`: branches over the package, statements, functions and lines over each module; psychometrics' take 36 minutes                                                                                                                                                                                                                                                             |
| `pnpm bench`             | Builds, then the benchmarks of every package, one package at a time, against the budgets of `test/budgets.ts`                                                                                                                                                                                                                                                                                                                                        |
| `pnpm knip`              | Unused files, exports and dependencies                                                                                                                                                                                                                                                                                                                                                                                                               |
| `pnpm build`             | Every package, from a clean `dist/`                                                                                                                                                                                                                                                                                                                                                                                                                  |
| `pnpm check:packages`    | Builds, then publint and Are the Types Wrong on each packed package                                                                                                                                                                                                                                                                                                                                                                                  |
| `pnpm check:release`     | Fails when a package of a changeset is private, lacks `publishConfig.access`, or is at 0.0.0 with less than `major`, or notes that tell of an earlier state                                                                                                                                                                                                                                                                                          |
| `pnpm check:tarballs`    | Builds, packs every package, installs the tarballs into a scratch project, imports every subpath and type-checks it with `bundler` and `node16`                                                                                                                                                                                                                                                                                                      |
| `pnpm check:rules`       | Removes each rule of `scripts/check-rules.ts` from a copy of the code; its test must fail (exit 1 when it does not); a run that takes too long is `TIMED OUT` (exit 2), never guarded or unguarded                                                                                                                                                                                                                                                   |
| `pnpm check:licenses`    | Fails on a production dependency whose license is not MIT, Apache-2.0, BSD or ISC                                                                                                                                                                                                                                                                                                                                                                    |
| `pnpm check:audit`       | `pnpm audit` of the production dependencies at any severity and of all at high; needs the network                                                                                                                                                                                                                                                                                                                                                    |
| `pnpm compat:runtimes`   | Builds, then bundles a calibration and a BKT fit and runs them on Node.js, Deno, Bun and workerd; their results must match Node.js's (`compat/README.md`); needs Deno and Bun, or name the runtimes: `pnpm compat:runtimes node workerd`                                                                                                                                                                                                             |
| `pnpm docs:api`          | Builds, then TypeDoc into `docs/api`; fails on undocumented exports. The plugin `docs/typedoc-seo.ts` makes the site of `SITE` (the `structure-dir` router): each page titled `<heading> · <package>` with a description from its own text and a canonical URL, the sitemap of those URLs, a Markdown copy beside each README and guide, `llms.txt` by `scripts/llms.ts`, `robots.txt`, `CNAME`, `404.html` and the lines of the marks in the footer |
| `pnpm check:site`        | After `pnpm docs:api`, the test of the built site (`scripts/check-site.ts`): a title of its own, a description other than TypeDoc's and one canonical URL, listed by the sitemap, on every page; the Markdown copy of every README and guide; the parts of `llms.txt`; every link to the site from the READMEs, the copies and `llms.txt` reaching a file of it                                                                                      |

`pnpm --filter @measurekit/psychometrics <script>` runs a script of one package. Run all of the commands before
finishing a change. Development needs Node.js 22.18+ and pnpm 12.

CI's `changes` job finds what a push reaches since the last run that passed, and only the jobs that read it run; `check`
always does. A change to a package reaches the packages that use it, the examples, `compat` and the tests of the scripts; the
guides and the README reach the examples alone, and a performance page no job; the lock file, the root configuration and `ci.yml` reach every job. A newer push
to the same ref cancels the run in progress, and `changes` compares with the last run that passed, so what a cancelled run did
not check is checked by the newer one. Every job has a `timeout-minutes` of about three times its last green run.

Per-push CI stays short, as Actions minutes are limited: `rules.yml` runs the slow checks each Monday, on `workflow_dispatch`
and from `release.yml` before it publishes: `rules`, and `coverage`, the floor of psychometrics, whose tests take 36 minutes
under coverage. A builder runs the rules of its package before a merge (`pnpm check:rules packages/<package>`).

## Releases

Changesets version and publish the packages, each on its own version. A release has one changeset per package in
`.changeset/`; changes made before the release extend it. A pull request from a `changeset-release/` branch runs
`pnpm changeset version`, which bumps the versions, writes `CHANGELOG.md` and consumes the changesets. After it is
merged, the maintainer runs the Release workflow by hand from the public repository: it runs `rules.yml`, and when the
rules and the coverage floor of psychometrics pass, `pnpm release`, which builds and runs `changeset publish` and publishes
the versions not yet on npm with provenance, the first version of each package with the short-lived token of the secret
`NPM_TOKEN` and later ones through each package's trusted publisher; then `scripts/github-release.ts` creates one GitHub
release named by the date with the versions published and their changelog entries (`node scripts/github-release.ts
--dry-run` prints it and creates nothing). It uses no changesets/action. No workflow opens pull requests, and Dependabot
version updates are off. Changesets, docs and TSDoc describe current behavior only. A package at 0.0.0 takes `major` in its
first changeset, so that it is released as 1.0.0; `pnpm check:release` fails on a `minor` or a `patch` for it, on notes
of it that tell of an earlier state, which the reader of 1.0.0 never had ("(was `", "no longer", "renamed"), and on a
package of a changeset that is private or lacks `publishConfig.access`.

At the public release the owner sets what a repository file cannot: the repository's description (the family's sentence of
`scripts/family.ts`), topics and homepage, the social preview (`.github/social-preview.png`, drawn from
`.github/social-preview.svg`), the organization's profile (`.github/profile/README.md`, copied into the organization's
`.github` repository), the domain of `SITE` with Pages and `docs.yml`, and Zenodo, which reads `CITATION.cff`. Context7 reads
`context7.json`, whose description and rules `pnpm check:standards` holds to the family's sentence and `FACTS`.

## Dependencies

- No package's `src/` uses a platform API: no `node:*`, `fetch`, DOM or `process`. `@measurekit/core` has no
  dependencies, and the libraries depend on it alone. The libraries do not import each other.
- A module imports only the modules in its row; every module may import `@measurekit/core`. Biome enforces the rows,
  and imports never form cycles.
- Code goes into `@measurekit/core` only if both libraries use it. What one library uses stays in that library, so that
  core does not become a place for everything. A generic routine that both use, such as a Cholesky factorization, is
  written once, in core.
- Every dependency of every `package.json` of the workspace is an exact version, development tools too, so that every
  machine and CI run builds the same output; `scripts/test/versions.test.ts` fails on a range. `@types/node` stays on the
  oldest supported Node.js major, and Dependabot ignores its major versions.
- A version is installed only when it is at least 3 days old: `pnpm-workspace.yaml` sets `minimumReleaseAge: 4320`, which
  pnpm checks against the lock file on every install.
- `pnpm check:audit` lets one advisory below high pass, of a development dependency that reads no untrusted input
  (2026-11-05): minimatch 10.2.6 (GHSA-g3hv-75wg-m249), which TypeDoc 0.28.20, the latest, runs on the entry points of
  `typedoc.json`, and whose repair, minimatch 11, makes private the `matchOne` that TypeDoc calls. The production
  dependencies have none.
- Code outside `packages/` imports a package by name.
- A module imports another only through its `index.ts` or its `internal.ts`, which lists what the modules and tests of
  the package share and is never exported from the package. `pnpm check:standards` fails on a deep path.

Psychometrics: every module of `cat/`, `cdm/`, `change/`, `equating/`, `fit/`, `irt/`, `multidimensional/`, `rater/`,
`scaling/`, `sim/`, `similarity/` and `standard-setting/` may also import `checks.ts`.

| Module              | Imports                                               |
| ------------------- | ----------------------------------------------------- |
| `math/`, `data/`    | none                                                  |
| `change/`           | none                                                  |
| `ctt/`              | `data/`, `math/`                                      |
| `dimensionality/`   | `data/`, `math/`                                      |
| `irt/`              | `data/`, `math/`                                      |
| `parameterization/` | `irt/`, `multidimensional/`                           |
| `cdm/`              | `data/`, `irt/`, `math/`                              |
| `fit/`              | `data/`, `irt/`, `sim/`, `math/`                      |
| `response-time/`    | `data/`, `irt/`, `math/`                              |
| `similarity/`       | `data/`, `irt/`, `math/`                              |
| `equating/`         | `irt/`, `math/`                                       |
| `sim/`              | `data/`, `irt/`, `cdm/`, `multidimensional/`, `math/` |
| `dif/`              | `data/`, `irt/`, `equating/`, `math/`                 |
| `standard-setting/` | `irt/`, `equating/`, `math/`                          |
| `explanatory/`      | `data/`, `irt/`, `math/`                              |
| `multidimensional/` | `data/`, `irt/`, `math/`                              |
| `generalizability/` | `data/`, `math/`                                      |
| `mokken/`           | `data/`, `math/`                                      |
| `scaling/`          | `irt/`, `equating/`, `math/`                          |
| `rater/`            | `data/`, `irt/`, `fit/`, `math/`                      |
| `cat/`              | `irt/`, `sim/`, `cdm/`, `multidimensional/`, `math/`  |

Knowledge tracing: the model families never import each other, nor does anything import `evaluate/`, `practice/` or
`sim/`: they use models only through `LearnerModel`.

| Module                                                                        | Imports                      |
| ----------------------------------------------------------------------------- | ---------------------------- |
| `math/`                                                                       | none                         |
| `data/`                                                                       | none                         |
| `model.ts`                                                                    | the types of `data/`         |
| `elo/`, `bkt/`, `logistic/`, `irt/`, `kst/`, `evaluate/`, `practice/`, `sim/` | `data/`, `math/`, `model.ts` |

## Deviations from the standards

Where the code departs from `STANDARDS.md`, with the reason:

- Section 4, "Small and stable surface": the root of each library re-exports the core names the package's API uses:
  `createRandom`, `DataError` and `MeasurekitError`, and every type of core that its declarations name, such as `Random`,
  `QMatrix` and `ThresholdItem`. These names have two entries, core's and the root's. A user installs one library, and
  pnpm does not let a program import a dependency of a dependency, so a user who names one of them would otherwise have
  to install core; `instanceof DataError` holds for both entries, as they are one module. `scripts/test/root-types.test.ts`
  holds each root to exactly these names. The roots hold nothing else but knowledge tracing's `LearnerModel` and
  `ModelFit`, which no module owns, and no module entry re-exports core: every other name has one entry, its module's
  subpath.

## Naming

- Acronyms are words: `kr20`, `bfgs`, `auc`, `rmse`. Other statistics are named in full: `standardErrorOfMeasurement`,
  `logLoss`. `QMatrix` keeps the field's spelling, with a Biome exception.
- A function that acts opens with its verb; one that computes a named quantity takes the quantity's name, as R and SciPy
  do (`cronbachAlpha`, `mantelHaenszel`), and a model, estimator or procedure the literature names keeps its name
  (`bkt`, `calibrate`, `linearEquating`). The verbs have one meaning across the families (STANDARDS section 4):

  | Verb            | Meaning                                                            | Example                                  |
  | --------------- | ------------------------------------------------------------------ | ---------------------------------------- |
  | `build`         | Assembles plain data from parts.                                   | `buildResponseMatrix`, `buildEventLog`   |
  | `create`        | Returns an object with behavior or state.                          | `createAdaptiveTest`, `createRandom`     |
  | `read`, `write` | Turns the form that another tool defines into the model, and back. | `readMirtIrt`, `writeMirtIrt`            |
  | `select`        | Chooses among candidates.                                          | `selectFirstAttempts`                    |
  | `check`         | Asserts and throws `DataError`.                                    | `checkArray`                             |
  | `validate`      | Judges a value and returns findings as data.                       | `validateQMatrix`                        |
  | `is`, `has`     | Predicates.                                                        | `isUnitInterval`, `hasOrderedThresholds` |

- Errors are `DataError` and its base class `MeasurekitError`, both of `@measurekit/core`, which the libraries
  re-export. A package that needs an error of its own extends `MeasurekitError`.
- Function options are `<Name>Options`; options shared by several functions are named after what they configure
  (`GroupOptions`), and a method's after it (`NextQuestionOptions`). Results are nouns: `ItemStatistics`, `SplitHalf`.
- Psychometrics labels are string unions in kebab-case (`"very-good"`); their thresholds are `<NAME>_THRESHOLDS`.
- A knowledge tracing model family has a factory named after it, `elo(options)`, that returns a `LearnerModel` or a type
  extending it (`EloModel`); a family that cannot work without a value takes it first, as `irt(parameters, options)` and
  `kst(structure, options)` do. Its files: `<family>.ts` for the factory, `state.ts` and `parameters.ts` for the plain data
  and their checks, `estimation.ts` for a fit's numerics.
- Public iterative methods take `tolerance` and `maxIterations` and report `converged` and `iterations`; random starts
  take `restarts` and `seed`. Defaults are named constants, such as `DEFAULT_TOLERANCE`. An internal iteration that a
  public method runs inside its own, such as `symmetricEigen` or `poissonFit`, may stop at named constants with their
  source instead, such as `MAX_ITERATIONS`, and reports `converged` to its caller where it can fail to.

## Code

- No `as` casts except where untyped data enters and is validated next.
- A record whose keys are names the code knows, fields or parameter names, goes through `checkRecord`, which refuses
  an own key `__proto__`, `prototype` or `constructor`. A record whose keys are the user's identifiers, such as learners,
  items, facets or covariates, goes through `checkRecordById`: every key is data, read with `Object.hasOwn`,
  `Object.entries` or a `Map` and written with `Object.fromEntries`, with a test that an id named `constructor` works.
- A public function validates its input once, where it reads it, and throws `DataError` with a full sentence that says
  what is wrong and what is expected (`A seed must be a safe integer, found 1.5.`). Input may come from plain JavaScript
  or JSON, so every argument is checked for its type first, with the checks of `@measurekit/core` (`checkOptions`,
  `checkFields`, `checkArray`, `checkRecord`, `checkRecordById`, `checkMethod`, `checkRandom`, `checkBoolean`, `checkChoice`, `checkFinite`,
  `checkPositive`, `checkNonNegative`, `checkPositiveInteger`, `checkIntegerFrom`, `checkUnitInterval`, `checkSeed`,
  `checkStopping`) and `checkFunction` and `checkOpenUnitInterval` of psychometrics. An option's label is `The option
  <name>`: `<Label> must be a positive number, found <value>.` A field of a record is tested with a predicate of core or
  of the language, such as `isUnitInterval` or `Number.isFinite`, and the message names the record
  (`Event 3 needs an outcome from 0 to 1, found 2.`); values are shown with `describe`. A list is checked with
  `checkArray`, which refuses holes, or read by index, as `forEach`, `map` and `every` skip a hole unseen.
- Psychometrics checks items with `checkItem` or `checkItemList` and a model with `checkModel`, all of `irt/items.ts`.
  Knowledge tracing checks a learner model with `checkModel` of `model.ts`, attempts with `checkAttempts`, and a
  model's predictions with `checkUnitInterval`.
- A public function returns plain data: every array and record in a result is `readonly`, at the top level as inside
  objects, so that a caller who sorts in place copies first. `pnpm check:standards` fails on a return type that a caller
  could write to.
- Options are destructured with their defaults, `const { tolerance = DEFAULT_TOLERANCE } = options`, so that a value set
  to `null` is rejected, never taken as absent.
- A public function checks its options with `checkOptions(options, NAMES)`, and a record of settings inside them, such
  as a prior, a quadrature or bounds, with `checkFields(record, NAMES, label)`: `NAMES` is a constant
  `FieldNames<TheOptions>` beside the type, which the type checker holds complete, and a name not in it throws
  `DataError`. A helper that functions with different options share takes the caller's names or leaves the check to
  them. Records of data (items, events, states, parameters, results passed back) are not checked this way, as they may
  carry a host's fields.
- Missing responses are `null`, and each function documents which persons a statistic uses. A statistic undefined for
  the data is `null` in tables and a `DataError` from a function that returns only it; never `NaN`. Iterative methods
  report `converged` and `iterations` instead of throwing.
- Knowledge tracing models are immutable: `update` returns a new state and `fit` a new model. States and parameters are
  plain data that JSON serializes, and model methods check them, as they may come from storage. A state or parameters
  stored by a 1.x release stay readable by every later 1.x release: a new field is optional, its absence means the old
  behavior, and a change that cannot be optional is a major version (`test/model.test.ts` reads
  `test/fixtures/stored-state.json`, which no build regenerates).
- An event with `outcome: null` is a learning opportunity: `update` accepts it, `predict` is never called for it, a fit
  gives it a `null` prediction and no likelihood term, and `n` of the information criteria counts the events with an
  outcome. Only an explicit `null` means that nothing was observed: a missing outcome throws `DataError`.
- Numerics work on `Float64Array`, do not allocate in inner loops and stay defined at extreme parameters: a weight that
  underflows to 0 is skipped before it divides, and a loop that grows or shrinks a value until a test passes also stops
  when the value is not finite. A response's probability enters a psychometrics log-likelihood as at least
  `MIN_PROBABILITY` of `irt/probability.ts`, through `logProbability`; knowledge tracing throws `DataError` for outcomes
  of probability 0.
- Never spread an array whose length the input sets into the arguments of a call (`push`, `unshift`, `append`, `prepend`,
  `replaceChildren`, `Math.max`, `Math.min`): above about 100,000 values that throws a `RangeError`. Loop over the values, or
  spread into an array literal. `pnpm lint` fails on it in `packages/*/src` (`biome/no-spread-arguments.grit`).
- A number with a meaning is a named constant with its source, such as the reference's name for it.
- A capability is built by the rule of `docs/<package>/architecture.md#sources`. The guide lists the rest under "Not included", a table that opens with
  `| Not included | Why |`, one row for each method, `| Method | Reason: why |`, with Reason one of No fit, No reference
  or Outside the sources. No speculative options, abstractions or modules.

## Repeats left on purpose

`pnpm check:duplication` runs jscpd over `packages/*/src` with `--min-tokens 40 --min-lines 4`, the settings of the three
repositories, and compares the clones with the table below: it fails on a clone whose files no one row holds together, and on
a file of a row that no longer repeats anything. A new clone is merged into a helper of the module that owns the concept, or
listed here with its reason. A file is written `<package>/<path below src>`, and `{a,b}` stands for several files of a folder.

| Repeat                                                              | Where                                                                                                                                                                                                                                                                                                                         | Why it stays                                                                                                                                                                                                                                                                           |
| ------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Import lists, signatures and the checks that open sibling functions | `psychometrics/cat/{bounds,priority-index}.ts`; `psychometrics/equating/{neat,neat-equipercentile}.ts`; `psychometrics/irt/{joint-maximum-likelihood,conditional-likelihood,ability}.ts`; `psychometrics/multidimensional/{calibration,em,scoring,mhrm,mhrm-run,likelihood-mode}.ts`; `psychometrics/irt/calibration.ts`      | No logic                                                                                                                                                                                                                                                                               |
| A value and its gradient, or two sides of one sum                   | `psychometrics/equating/characteristic-curve.ts`, `psychometrics/irt/{expectation,graded,latent,logistic}.ts`, `psychometrics/multidimensional/{graded,logistic,mhrm-standard-errors}.ts`, `psychometrics/math/noncentral-chi-square.ts`, `psychometrics/dif/mantel-test.ts`, `knowledge-tracing/irt/irt.ts`, `core/kappa.ts` | One form for both takes a mode parameter and reuses nothing                                                                                                                                                                                                                            |
| Descent loops                                                       | `core/bfgs.ts`, `psychometrics/math/optimize.ts`                                                                                                                                                                                                                                                                              | BFGS and Newton differ in direction and stopping; the line search and the matrix-vector product are in core                                                                                                                                                                            |
| The likelihood of a pattern                                         | `psychometrics/irt/ability.ts` and `psychometrics/multidimensional/pattern.ts`                                                                                                                                                                                                                                                | The model methods take theta as a number in one and as a vector in the other                                                                                                                                                                                                           |
| Sums over a person's answers                                        | `psychometrics/fit/{item-fit,person-fit}.ts`; `psychometrics/equating/{bivariate,neat-equipercentile}.ts`                                                                                                                                                                                                                     | The loops accumulate different sums                                                                                                                                                                                                                                                    |
| Priors of an item's entries                                         | `psychometrics/multidimensional/{m-step,mhrm-moments}.ts`                                                                                                                                                                                                                                                                     | Each adds a different term, on the own or the unbounded scale                                                                                                                                                                                                                          |
| Dimension names by item id                                          | `psychometrics/multidimensional/{bifactor-structure,design}.ts`                                                                                                                                                                                                                                                               | The messages name the option that is wrong                                                                                                                                                                                                                                             |
| Fields of results of one estimation                                 | `psychometrics/explanatory/lltm.ts` and `psychometrics/rater/many-facet-rasch.ts`                                                                                                                                                                                                                                             | Each result documents its fields for its own function                                                                                                                                                                                                                                  |
| Counting pairs of categories                                        | `core/kappa.ts`                                                                                                                                                                                                                                                                                                               | A table where the range of the categories is no wider than the pairs, a map otherwise: one loop for both takes a call per pair                                                                                                                                                         |
| The likelihood of a pattern at one theta                            | `psychometrics/cat/classification-stopping.ts` and `psychometrics/irt/ability.ts`                                                                                                                                                                                                                                             | The SPRT sums the items it holds at the bounds of each cut; the estimators' sum is a closure over checked responses                                                                                                                                                                    |
| The fields of a limited-information fit                             | `psychometrics/fit/limited-information.ts` and `psychometrics/cdm/model-fit.ts`                                                                                                                                                                                                                                               | No logic: the same statistics, described for IRT and for diagnosis models                                                                                                                                                                                                              |
| A person's answered items, twice                                    | `psychometrics/fit/root-mean-square-deviation.ts`                                                                                                                                                                                                                                                                             | One pass adds their log-probabilities to the posterior, the next the posterior to their counts                                                                                                                                                                                         |
| The checks that open a limited analysis                             | `psychometrics/mokken/scores.ts` and `psychometrics/cdm/model-fit.ts`                                                                                                                                                                                                                                                         | No logic                                                                                                                                                                                                                                                                               |
| The correlation of two items by two passes                          | `psychometrics/dimensionality/pearson.ts` and `psychometrics/multidimensional/starting-slopes.ts`                                                                                                                                                                                                                             | The matrix of `math/` skips the persons who lack a response and gives NaN where undefined; the starting slopes read the first `count` values of buffers they reuse for every pair of items, which a call of the matrix's routine on a `subarray` would allocate a view for, and give 0 |

## Comments and docs

- Every export has a TSDoc comment: types and constants a noun phrase, functions a third-person verb phrase ("Returns
  ...", "Builds ..."), properties a short phrase. Formulas use plain notation: `k / (k - 1)`. The tags:
  - The first sentence is the summary. A longer explanation, such as a method, a formula or the edge cases, follows under
    `@remarks`; a description of more than three lines has one.
  - `@param name - ...` describes every parameter of a function, and `@returns` its result where the summary does not say
    it: the shape, the units, the range and when it is `null`. A function that returns nothing has no `@returns`; a
    destructured parameter has one `@param`.
  - `@defaultValue` gives the default of every option and every parameter that has one, as code (`@defaultValue
    \`1e-4\``) or, when the default is a behavior, a short phrase. The summary never states a default. A parameter whose
    initializer is in the signature shows it in TypeDoc; one whose default is computed in the body has `@defaultValue` on
    the function, naming the parameter. An option whose default differs between functions has an options type of its own.
  - `@throws` lists the errors.
  - `@see {@link https://doi.org/... | Author (year)}` cites the literature, one tag per work, with the DOI where it has
    one. The summary and `@remarks` name no year.
  - `@example` goes on the main function of each module, as `pnpm check:standards` demands; others may have one. It is
    complete, with its imports, runs alone and shows at least one result, as the samples of the guides do.
  - No other tags and no boilerplate.
- `//` comments only explain why something non-obvious is done, or cite a source. No commented-out code.
- The guide of a module, every page of `docs/<package>/` but `architecture`, `choosing`, `coming-from-pybkt`, `coming-from-r`,
  `evidence`, `performance`, `storage`, `supported` and `validation`,
  has `## Quick start` first, a minimal sample of the main call, and ends with `## Errors`, `## Not included` and
  `## Differences from the references`, in that order, before `## References`; a section with nothing to say says "None"
  and why. `pnpm check:standards` fails on a guide without the four sections (`guide-sections`) and on a Not included
  section that is not the table above (`not-included`).
- Docs lead with a picture, then tables, then prose. Diagrams are Mermaid in Markdown: GitHub renders them, and
  `typedoc-plugin-mermaid`, pinned in `typedoc.json` with the Mermaid version it loads, renders them on the docs site.
- US English; short sentences, a table or list rather than prose; complete samples with their imports; a source's section
  is cited in words ("section 3.1"). No marketing, emojis or filler.
- A sample shows what a statement gives as `// => value`, on the statement's line or the line under it, with an
  explanation after a `,`, `;` or `:`: `model.mastery(next, "fractions"); // => 0.73, P(L)`. A declaration of one name is
  checked by the value it declares, and a value that spans lines goes on in the comment lines under it until its brackets
  close. `scripts/stated-values.ts` compares the value: a number at the digits shown (an integer exactly, so a computed
  zero is `0.0`), an array element by element, with `...` for the elements left out, an object by its keys unless `...`
  stands among them. Results follow statements of the top level: one inside a block is never picked up, and fails the test
  as a value in a plain comment on a statement's line does.
- `examples/test/guides.test.ts` runs every `ts` sample of the guides and READMEs and `examples/test/tsdoc.test.ts` every
  `@example` of the TSDoc, on the sources, through `scripts/guide-samples.ts`; each fails when a sample throws, gives a
  value other than a result it shows, or shows a result that no statement picks up or a value without `=>`. A fence `ts
  continues` runs after the sample before it and shares its variables, and `ts no-run` marks code that needs a worker or a
  package the repository does not have, and shows no result.
- The README of a package is its npm page: it opens with its description as its first sentence and the install line, and has
  the sections `## Quick start`, `## Architecture`, with its diagram, since npm shows a diagram as its source, `## API`,
  `## Supported`, `## Documentation` and `## License`; the root README opens with the family's sentence and has
  `## Which package`, with each package's description, `## Documentation`, `## Development`, `## Layout` and `## License`, in
  that order (`readme-sections`). The root README and the README of each package link absolutely: a guide or a package at its
  page on the docs site, any other file at its address in the repository.
- Every package has `supported.md`, `evidence.md` and `choosing.md` (`pnpm check:standards` fails without one);
  psychometrics has `coming-from-r.md` and knowledge tracing `coming-from-pybkt.md`. A number on them is one that a file of
  the repository holds (a fixture, a reference script, `validation.md`, `performance.md`) and states with its commit or date;
  a number with no such file is left out. `choosing.md` names, for each comparator read at a stated version and date, what
  it has that the package lacks, with a link to the guide whose Not included table gives the reason.
- Every guide's `#` heading names its subject ("Events and logs", not "Data"), since the site titles the page with it, and
  every entry point of a package is a `@module` named by its import path, so that its pages sit apart from the guides.
  `llms.txt` and the Markdown copies are generated by `pnpm docs:api` and never written by hand.
- `scripts/family.ts` holds the identity of the family: `SITE`, the base of the docs site (`https://measurekit.wakan.dev/`,
  the owner's domain), the family's sentence, each package's description and keywords, and the marks of others with their
  holders. A package's `description`, `keywords`, `homepage` and `author` and the first sentence of its README come from its
  row, and `scripts/check-identity.ts` holds them to it in `pnpm check:standards`; change a sentence in `family.ts` first. The
  plugin `docs/typedoc-seo.ts` reads the same file.
- Marks of others are named nominatively (`STANDARDS.md` section 2, "Names"): the plain name, and no claim word (compliant,
  conformant, certified, official, approved, endorsed) in a README, a NOTICE, a guide or a description; a quotation stays in
  quotation marks. ETS's is the one mark the texts name, for its DIF categories and its thresholds of automated scores; the
  R packages, pyBKT, RSMTool and the other products a text compares with are named plainly, with no line. Every README and
  NOTICE that names ETS carries, each on a line of its own, its mark's line and the line that measurekit is not affiliated
  with it, as `linesFor` of `scripts/identity.ts` writes them; the symbol of a mark appears only inside those lines.
  `pnpm check:standards` names any line that is missing or not needed.
- One purpose for each document: a package's `README.md` for its users, `docs/<package>/` for its guides,
  `CONTRIBUTING.md` for the process, `AGENTS.md` for conventions, `SECURITY.md` for deployment security,
  `validation/README.md` for running the reference scripts and the benchmarks.

## Evidence and tests

- Every statistic has a reference: a function of an established R or Python package, or a base R computation written out
  in the script. The script writes inputs and expected values to `packages/<package>/test/fixtures/<module>-<case>.json`,
  from public datasets (mirt, psych, difR, pyBKT) or simulations with a fixed seed, and the test compares within the
  tolerance in `docs/<package>/validation.md`. Expected values are never typed by hand, except for edge cases small enough
  to derive exactly.
- A test reads a fixture with `readFixture<T>("name.json")` of its package's `test/fixtures.ts` and declares `T` for the
  fields it reads; a type that several files use goes in `fixtures.ts`. It never imports a JSON file: the type checker
  builds a type for every element, and `pnpm lint` fails on the import. `resolveJsonModule` is off in
  `tsconfig.base.json`: TypeScript 6 turns it on under `NodeNext` when the option is absent.
- The numerics of `@measurekit/core` are validated by the fixtures of the packages that use them; core's tests cover its
  edge cases. Psychometrics `sim/` has no R reference, as R's generator differs from `createRandom`: seeded tests check its
  distributions, and a calibration recovers the items it simulates. A knowledge tracing model without a package of
  record, such as Elo, has its equations written out in R and a seeded recovery study, with thresholds fixed before the
  simulation is first run.
- One test file for each source folder, named after it, and `model.test.ts` for knowledge tracing's `model.ts`. The
  bifactor models of `multidimensional/`, whose tests take minutes, have `bifactor.test.ts` and `bifactor.bench.ts`, so
  that they run beside the folder's other tests. `test/assertions.ts` holds the shared assertions, such as
  `expectClose`, which compares with a reference value; `test/logs.ts` of knowledge tracing holds `online`, which runs a
  log through any model as an app would; psychometrics' `test/arbitraries.ts` holds the fast-check generators of items
  and `test/fixture-items.ts` reads the items of the fixtures. A file's `describe` blocks follow the order of the folder's source files or the
  guide's sections, with a closing `properties` block in psychometrics; within a block, the comparisons with the reference
  and the main behavior come first, then the options, then errors and edge cases. Titles are plain sentences about
  behavior: `it("shares the places at a tied boundary equally")`.
- Every public function has a test of input of other types, as JavaScript or JSON pass it: `null` options and arguments,
  a string for a number, an object for an array, passed with `JSON.parse`, not a cast.
- Properties that hold for any input (alpha at most 1, probabilities from 0 to 1, invariance to the order of persons,
  states left unchanged by `update`, states that continue identically after a JSON round trip, reproducible seeds) are
  tested with fast-check, from the fixed seed that `test/setup.ts` of each package gives every property. A generator of distinct doubles is `fc.uniqueArray(..., { comparator: "SameValueZero" })`, as
  the default comparison draws -0 and 0 as two values; the GritQL plugin `biome/unique-doubles-same-value-zero.grit`
  makes `pnpm lint` fail otherwise.
- Edge cases are tested for every function: constant items, persons and items without responses, one item, two persons;
  an empty log, one learner, events out of time order, unknown items and skills, outcomes all correct or all wrong, and
  each model's limits, such as BKT parameters at their bounds or a posterior mode beyond the IRT grid.
- A module has benchmarks of its hot path in `test/<module>.bench.ts`, the size of the input in the title of each, a budget
  in `test/budgets.ts` and, where a reference does the task, the same task timed in
  `validation/benchmarks/<package>/<module>.R`, or a `<module>-<topic>.R` beside it, on the same data. Each entry point has
  a size budget in `scripts/size-budgets.ts`, and the readers of untrusted input of the largest peaks, at the limits or
  past the size of their input, a case with a heap and a budget in `scripts/peak-memory.ts`. `docs/<package>/performance.md`
  gives the times against the references, the sizes and the peak memory of the readers of untrusted input, and
  `validation/README.md` the method of the benchmarks, how to refresh the pages and how the budgets were measured.
- A bug fix comes with a test that fails without it, in the test file of the fixed folder. A rule that keeps untrusted
  input from being misread, or from costing what its size does not, has an entry in `scripts/check-rules.ts`.
- A limit is a constant `MAX_*` that the message of its `DataError` names, or the default `DEFAULT_MAX_*` of an option
  `max*` that a `DataError` names, with a row in [Default limits](SECURITY.md#default-limits) that states its value:
  `pnpm check:standards` fails on a limit whose value the section does not state (`security-limit`).

## Where things go

- A statistic: a file in its module folder, exported from the module's `index.ts`, with a row in the module's guide, a
  reference in `validation/<package>/<module>.R` (or `.py`), a fixture, tests, and a tolerance row in
  `docs/<package>/validation.md` if it needs its own.
- A knowledge tracing model family: a module whose factory returns a `LearnerModel`, with its state and parameters as
  plain data and their checks, its steps for `fit`, and the property tests of the interface.
- A module: a folder in `src/` with `index.ts` (and `internal.ts` for what others share), a subpath in the package's
  `exports`, an entry point in its `typedoc.json`, a test file, a guide, a reference script
  if it computes statistics, benchmarks of its hot path with their budgets and, where a reference does the task, its
  timing in `validation/benchmarks/`, import rules in `biome.json` and the table above, and a row in the API table of the
  package's `README.md`.
- A package: a folder in `packages/` with the files listed under Layout, a project and a benchmark project in
  `vitest.config.ts`, an entry in the root `typedoc.json`, a changeset, a row in the root `README.md` and, for a library,
  `docs/<package>/`.
- Numerics shared by modules: `math/`, internal unless users need them; numerics both libraries use: `@measurekit/core`.
- Checks shared by modules: of one value in `@measurekit/core` or, if only that library uses it, `checks.ts` of the
  library; otherwise with the data they check, such as `data/dichotomous.ts` and `irt/items.ts` in psychometrics,
  `data/event.ts` and `model.ts` in knowledge tracing.
- An example: `examples/<package>/<name>.ts`, importing the package by name, with a header comment that says what it shows
  and how to run it. `pnpm test --project examples` runs every example on the sources and fails when one throws.
