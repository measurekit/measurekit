---
"@measurekit/core": major
---

The foundation of `@measurekit/psychometrics` and `@measurekit/knowledge-tracing`: `MeasurekitError` and
`DataError`, the checks of arguments, the seeded random number generator `createRandom` and the numerics that
both packages use: the normal distribution function, density and quantile, the tails of the chi-square and Student's t
distributions, Cohen's kappa, BFGS minimization with its
line search, the Fisher-Yates shuffle, and the category probabilities of logistic, graded and partial credit items.

`checkRecord` refuses an own key `__proto__`, `prototype` or `constructor` in a record whose keys are names the code
knows, and `checkRecordById` takes a record whose keys are the user's identifiers, where any of the three is data.
`checkArray` refuses an array with a hole, such as `new Array(3)` or `[1, , 2]`, which `forEach`, `map` and `every`
would skip, so that every check of a list reads each of its entries. `chiSquareUpperTail` is 0 at an infinite
statistic.
`kappaOfPairs` takes the chance agreement in time that grows with the categories that occur, not with their pairs, and
its weights as the option `weights`.
`softmax` turns contiguous logits into probabilities, and `stridedSoftmax` the logits of a column at a stride.
`NominalParameters` and `nominalProbabilities` give the nominal response item, which both packages read.
`hasOrderedThresholds` and `hasValidAsymptotes` test the thresholds of a graded item and the asymptotes of a logistic
item.

`createRandom` has an example, which a test runs against the values its comments state.

`bfgs` and `stepAlong` take a function as its value and, apart, its gradient (`ValueAndGradient`), as R's `optim` takes
`fn` and `gr`: the line search evaluates the value alone at each length it tries and the gradient at the length it
accepts, so that a gradient by finite differences is computed once a step. A length where that gradient is not finite
is refused and the step halved again, as `backtrack` refuses a length that its option `accept` refuses.

`checkOptions` takes the names of a function's options, a `FieldNames` constant that the type checker holds complete,
and refuses an option the function does not take, naming it; `checkFields` does the same for a record of settings inside
the options. `bfgs`, `backtrack` and `kappaOfPairs` refuse an option they do not take.
