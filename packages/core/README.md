# @measurekit/core

The shared foundation of the measurekit packages: most users install [`@measurekit/psychometrics`](https://github.com/measurekit/measurekit/tree/main/packages/psychometrics#readme)
or [`@measurekit/knowledge-tracing`](https://github.com/measurekit/measurekit/tree/main/packages/knowledge-tracing#readme),
which depend on it and re-export `createRandom`, `Random`, `DataError` and `MeasurekitError`.

It holds the errors the packages throw, the checks of their arguments, a seeded random number generator and the numerics
that both packages use. It has no dependencies and runs in any JavaScript runtime.

```bash
npm install @measurekit/core
```

## API

| Group    | Exports                                                                                                                                                                                                                                                                                                             |
| -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Errors   | `MeasurekitError`, the base class; `DataError`, for invalid input                                                                                                                                                                                                                                                   |
| Random   | `createRandom(seed)`, a seeded xoshiro128\*\* generator, and its interface `Random`                                                                                                                                                                                                                                 |
| Checks   | `describe`; `isNonEmptyString`, `isPositive`, `isRecord`, `isUnitInterval`; `checkRecord`, `checkArray`, `checkOptions`, `checkMethod`, `checkRandom`, `checkBoolean`, `checkChoice`, `checkFinite`, `checkPositive`, `checkNonNegative`, `checkPositiveInteger`, `checkUnitInterval`, `checkSeed`, `checkStopping` |
| Numerics | `normalQuantile`, `logistic`, `logLogistic`, `normalizeLogs`, `cholesky`, `dot`, `maxAbs`, `mean`                                                                                                                                                                                                                   |

A check throws `DataError` with a sentence that names the value and what was expected, such as
`A seed must be a safe integer, found 1.5.`

```ts
import { createRandom, DataError } from "@measurekit/core";

const random = createRandom(42);
random.uniform(); // a draw from [0, 1), the same for every run with seed 42
try {
  createRandom(1.5);
} catch (error) {
  error instanceof DataError; // true
}
```

## License

[Apache-2.0](LICENSE). Copyright Fariz Wakan; see [NOTICE](NOTICE).
