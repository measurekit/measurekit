# @measurekit/core

The shared foundation of the measurekit packages: most users install [`@measurekit/psychometrics`](https://github.com/measurekit/measurekit/tree/main/packages/psychometrics#readme)
or [`@measurekit/knowledge-tracing`](https://github.com/measurekit/measurekit/tree/main/packages/knowledge-tracing#readme),
which depend on it and re-export `createRandom`, `Random`, `DataError` and `MeasurekitError`.

It holds the errors the packages throw, the checks of their arguments, a seeded random number generator and the numerics
that both packages use. It has no dependencies and runs in any JavaScript runtime.

## License

[Apache-2.0](LICENSE). Copyright Fariz Wakan; see [NOTICE](NOTICE).
