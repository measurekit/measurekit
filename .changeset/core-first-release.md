---
"@measurekit/core": major
---

The foundation of `@measurekit/psychometrics` and `@measurekit/knowledge-tracing`: `MeasurekitError` and
`DataError`, the checks of arguments, the seeded random number generator `createRandom` and the numerics that
both packages use: the tails of the chi-square and Student's t distributions, Cohen's kappa, BFGS minimization with its
line search, the Fisher-Yates shuffle, and the category probabilities of logistic, graded and partial credit items.

`checkArray` refuses an array with a hole, such as `new Array(3)` or `[1, , 2]`, which `forEach`, `map` and `every`
would skip, so that every check of a list reads each of its entries. `chiSquareUpperTail` is 0 at an infinite
statistic.
`kappaOfPairs` takes the chance agreement in time that grows with the categories that occur rather than with their
pairs: 20,000 categories took seconds. It takes its weights as the option `weights`.
`softmax` turns contiguous logits into probabilities, and `stridedSoftmax` the logits of a column at a stride.
