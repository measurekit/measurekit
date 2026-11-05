# Architecture of psychometrics

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
  `null` is invalid, not absent, and an option a function does not take is refused with its name, as are the fields a
  record of settings does not have, such as a prior or a quadrature: a misspelled name would otherwise leave a default
  in place. Records of data, such as items and a calibration's testlets, may carry fields of their own. A statistic that is undefined for the data is `null` in a table and a `DataError`
  from a function that returns only that statistic; results never contain `NaN`. Iterative methods report
  convergence instead of throwing.
- **Extensible.** Where methods have alternatives, such as models or estimators, they are interfaces, and built-ins
  implement the same interfaces as user code.
- **Complete at the first version.** Each module has a fixed list of authoritative [sources](#sources), which the owner set. A capability is
  built when those sources treat it as standard practice, a usable reference exists and it fits a library. "Not included"
  holds the rest, one method per line, each No fit, No reference or Outside the sources. A new capability afterwards means
  changing the list.

## Sources

A capability is built when the sources of its module treat it as standard practice, a usable reference exists and it fits a library. Every method that is not built is listed under "Not included" in the module's guide, with the first test it fails.

```mermaid
flowchart LR
  C["A capability of the module's domain"] --> Q1{"Do the sources treat it<br/>as standard practice?"}
  Q1 -- no --> O["Not included: Outside the sources"]
  Q1 -- yes --> Q2{"Is there a usable<br/>reference?"}
  Q2 -- no --> R["Not included: No reference"]
  Q2 -- yes --> Q3{"Does it fit<br/>a library?"}
  Q3 -- no --> F["Not included: No fit"]
  Q3 -- yes --> B["Built, and validated against the reference"]
```

| Module             | Sources                                                                                                                                                                                                                                                                                            | Covers                                                                                                          |
| ------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| `ctt`              | [McDonald (1999)](https://doi.org/10.4324/9781410601087); [Thissen and Wainer (2001)](https://doi.org/10.4324/9781410604729-16); [Sinharay, Puhan and Haberman (2011)](https://doi.org/10.1111/j.1745-3992.2011.00208.x)                                                                           | Reliability and item analysis; scoring and augmented scores; subscores                                          |
| `dimensionality`   | [Reckase (2009)](https://doi.org/10.1007/978-0-387-89976-3), chapter 7; [Fabrigar et al. (1999)](https://doi.org/10.1037/1082-989X.4.3.272)                                                                                                                                                        | The structure of test data; factor analysis                                                                     |
| `irt`              | [van der Linden (2016)](https://doi.org/10.1201/9781315374512), Models; [van der Linden (2017)](https://doi.org/10.1201/b19166), Statistical Tools; [Lord (1980)](https://doi.org/10.4324/9780203056615); [van der Linden (2017)](https://doi.org/10.1201/9781315117430), Applications, chapter 14 | Unidimensional models, estimation, tests of the model; survey weights and plausible values                      |
| `multidimensional` | [Reckase (2009)](https://doi.org/10.1007/978-0-387-89976-3), chapters 4, 6, 8, 9; [van der Linden (2016)](https://doi.org/10.1201/9781315374512), Models; [Andersen (1985)](https://doi.org/10.1007/BF02294143)                                                                                    | Models, estimation, rotation, linking; two-tier, covariates and multiple groups; repeated testings              |
| `cat`              | [van der Linden and Glas (2010)](https://doi.org/10.1007/978-0-387-85461-8); [Magis, Yan and von Davier (2017)](https://doi.org/10.1007/978-3-319-69218-0); [Reckase (2009)](https://doi.org/10.1007/978-0-387-89976-3), chapter 10                                                                | Adaptive and multistage testing, multidimensional selection                                                     |
| `cdm`              | [von Davier and Lee (2019)](https://doi.org/10.1007/978-3-030-05584-4), chapters 7, 13, 15, 16, 29                                                                                                                                                                                                 | G-DINA, item fit, adaptive diagnosis, identifiability, the GDINA package                                        |
| `fit`              | [van der Linden (2017)](https://doi.org/10.1201/b19166), Statistical Tools; [van der Linden (2017)](https://doi.org/10.1201/9781315117430), Applications, chapter 6                                                                                                                                | Tests of model fit; person fit                                                                                  |
| `dif`              | [Holland and Wainer (1993)](https://doi.org/10.4324/9780203357811); [Penfield and Camilli (2006)](https://doi.org/10.1016/S0169-7161(06)26005-X); [Lord (1980)](https://doi.org/10.4324/9780203056615-20), "Study of item bias"                                                                    | DIF methods; Lord's chi-square and its purification                                                             |
| `equating`         | [Kolen and Brennan (2014)](https://doi.org/10.1007/978-1-4939-0317-7), chapters 1 to 8; [Brennan et al. (2009)](https://education.uiowa.edu/casma/computer-programs), Equating Recipes; [González and Wiberg (2017)](https://doi.org/10.1007/978-3-319-51824-4)                                    | Designs, smoothing, linear, equipercentile and IRT methods, kernel, errors                                      |
| `scaling`          | [Kolen and Brennan (2014)](https://doi.org/10.1007/978-1-4939-0317-7), chapters 9 and 10                                                                                                                                                                                                           | Score scales, linking, vertical scaling                                                                         |
| `standard-setting` | [Cizek and Bunch (2007)](https://doi.org/10.4135/9781412985918); [Cizek (2013)](https://doi.org/10.4324/9781410600264); [Livingston and Lewis (1995)](https://doi.org/10.1111/j.1745-3984.1995.tb00462.x); [McKinley and Norcini (2014)](https://doi.org/10.3109/0142159X.2013.853119)             | Methods of setting cut scores, also of OSCEs; the accuracy and consistency of classifications                   |
| `explanatory`      | [De Boeck and Wilson (2004)](https://doi.org/10.1007/978-1-4757-3990-9), chapters 5 to 7; [Fischer and Molenaar (1995)](https://doi.org/10.1007/978-1-4612-4230-7)                                                                                                                                 | Person regression, item predictors; linear logistic models, for dichotomous and polytomous items and for change |
| `change`           | [Cronbach and Furby (1970)](https://doi.org/10.1037/h0029382); [Jacobson and Truax (1991)](https://doi.org/10.1037/0022-006X.59.1.12)                                                                                                                                                              | Measuring change; reliable change                                                                               |
| `mokken`           | [Sijtsma and Molenaar (2002)](https://doi.org/10.4135/9781412984676); [Sijtsma and van der Ark (2017)](https://doi.org/10.1111/bmsp.12078)                                                                                                                                                         | The method and its ten steps                                                                                    |
| `generalizability` | [Brennan (2001)](https://doi.org/10.1007/978-1-4757-3456-0), chapters 1 to 12                                                                                                                                                                                                                      | Balanced and unbalanced univariate designs in chapters 1 to 8; multivariate designs in 9 to 12                  |
| `rater`            | [Eckes (2023)](https://doi.org/10.3726/b20875); Myford and Wolfe (2003, 2004); [Williamson, Xi and Breyer (2012)](https://doi.org/10.1111/j.1745-3992.2011.00223.x)                                                                                                                                | Many-facet models, facet interactions, rater effects; the evaluation of automated scores                        |
| `response-time`    | [van der Linden (2016)](https://doi.org/10.1201/9781315374512), Models; [De Boeck and Jeon (2019)](https://doi.org/10.3389/fpsyg.2019.00102)                                                                                                                                                       | Lognormal and hierarchical models; the overview of the field                                                    |
| `similarity`       | [Cizek and Wollack (2017)](https://doi.org/10.4324/9781315743097), chapters 2 and 3; [Sotaridona (2003)](https://ris.utwente.nl/ws/files/6072637/thesis_Sotaridona.pdf)                                                                                                                            | Answer copying and similarity of pairs of persons, and the pairs to compare                                     |

`data`, `math`, `sim` and `parameterization` have no list of their own: they follow what the other modules need.

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
`@measurekit/psychometrics/sim`, `@measurekit/psychometrics/similarity` and `@measurekit/psychometrics/standard-setting`.
Each public name has one entry, its module's subpath. `.` holds only what the modules share and core owns, so that a user
does not import core ([AGENTS.md](../../AGENTS.md#deviations-from-the-standards)): `createRandom`, `DataError`,
`MeasurekitError` and the types of core that the modules take or return, `Random`, `QMatrix`, `ThresholdItem`,
`NominalParameters`, `ThetaDistribution` and `KappaWeights`, which no module re-exports. `math/` is not exported.

## Dependencies between modules

Modules form layers; a module imports only from lower layers, never from its own or a higher one.

| Layer | Modules                                                                                   |
| ----- | ----------------------------------------------------------------------------------------- |
| 1     | `@measurekit/core`, `checks.ts`                                                           |
| 2     | `math/`, `data/`                                                                          |
| 3     | `ctt/`, `change/`, `irt/`, `dimensionality/`, `generalizability/`, `mokken/`              |
| 4     | `cdm/`, `equating/`, `response-time/`, `explanatory/`, `multidimensional/`, `similarity/` |
| 5     | `sim/`, `dif/`, `standard-setting/`, `scaling/`, `parameterization/`                      |
| 6     | `fit/`, `cat/`                                                                            |
| 7     | `rater/`                                                                                  |

Analysis modules (`ctt/`, `change/`, `irt/`, `dimensionality/`, `generalizability/` and `mokken/`) build on `data/` and
`math/` or on nothing, not on each other. `sim/` builds on `irt/`, `cdm/` and `multidimensional/` to simulate their models, `parameterization/`
on `irt/` and `multidimensional/` to convert their items, `cat/` on `sim/` for the same response simulation, on `cdm/` for
the profile estimates of diagnostic tests and on `multidimensional/` for the items, estimates and information of
multidimensional tests, `dif/` and `standard-setting/` on `equating/` for linking and score distributions, `scaling/` on
`equating/` for the linear transformation of a theta scale, `fit/` on `sim/` for the replications of the critical value of
Q3, and `rater/` on `irt/` for the Rasch model of its many
facets and on `fit/` for the fit of their elements. `explanatory/` builds on `irt/` for the calibrated items it
regresses, `multidimensional/` on `irt/` for its priors, compressed responses and parameter bounds, and `similarity/` on
`irt/` for the probability of each category at a person's theta.
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

### Why the numerics are written here

`math/` holds the numerics that this library alone needs: the eigendecomposition of a symmetric matrix by Householder's
reduction to a tridiagonal matrix and the implicit QL algorithm, as EISPACK's tred2 and imtql2, quadrature, rotation, the
minimum residual fit, Newton's method and the noncentral chi-square. Of them, `ml-matrix` 6.15.1 covers the
eigendecomposition, beside the singular value decomposition. The numerics both libraries share are in
`@measurekit/core`, whose [README](../../packages/core/README.md#why-the-numerics-are-written-here) says why `ml-matrix`
and `@stdlib` are not used for them.

| Eigendecomposition of a sample correlation matrix, fastest sample, 2026-11-05 | 25 items |   50 | 100 | 200 |  1,000 |
| ----------------------------------------------------------------------------- | -------: | ---: | --: | --: | -----: |
| `math/`, Householder and implicit QL (ms)                                     |    0.059 | 0.34 | 2.4 |  21 |  3,200 |
| `ml-matrix`, Householder and QL of JAMA (ms)                                  |    0.072 | 0.45 | 3.3 |  28 |  4,500 |
| Cyclic Jacobi, in the same style (ms)                                         |     0.24 |  1.9 |  17 | 145 | 26,000 |

The method makes the difference, not the library: the cyclic Jacobi method gives small eigenvalues to their relative
accuracy, which no analysis here needs, and takes 4 to 8 times as long as the QL method, whose eigenvalues are exact to
rounding relative to the largest, as LAPACK's are. Written in `math/`, with no dependency, the QL method takes 0.7 to
0.8 times as long as `ml-matrix`'s on the same matrices, alternated in one process, and adds 16 MB to the peak at 1,000
items, the most that the analyses of a matrix of items take. Base R's `eigen`, LAPACK's dsyevr through OpenBLAS, is
faster above about 100 items, 0.24 s at 1,000 as the reference of smoothing in [performance](performance.md), which
takes the eigenvectors too: it reduces the matrix in blocks by native kernels, and finds the eigenvectors of the
tridiagonal matrix by MRRR, in a time that grows with the square of its order, where the rotations of the QL method take
half of its time. A caller that reads the eigenvalues alone, as parallel analysis and the check of smoothing do, skips
the eigenvectors, in a fifth of the time at 1,000 items.
