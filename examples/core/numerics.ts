// What the two libraries share, used on its own: a seeded random number generator, the checks of arguments and their
// errors, minimization with BFGS, the tails of the chi-square and Student's t distributions, and Cohen's kappa for two
// raters. Every function validates its input and throws a DataError, which a MeasurekitError check also catches.
// Run after `pnpm build`: node examples/core/numerics.ts
import {
  bfgs,
  chiSquareUpperTail,
  createRandom,
  DataError,
  type GradientFunction,
  kappaOfPairs,
  MeasurekitError,
  normalQuantile,
  studentLowerTail,
} from "@measurekit/core";

// The same seed gives the same numbers on every machine.
const random = createRandom(2026);
console.log(
  "Three draws:",
  [random.uniform(), random.normal(), random.uniform()].map((value) => value.toFixed(4)),
);

// Minimize (x - 1)² + 10 (y + 2)²: the function writes its gradient into the array it is given.
const bowl: GradientFunction = ([x = 0, y = 0], gradient) => {
  gradient[0] = 2 * (x - 1);
  gradient[1] = 20 * (y + 2);
  return (x - 1) ** 2 + 10 * (y + 2) ** 2;
};
const fit = bfgs(bowl, [0, 0]);
console.log(
  "Minimum at",
  Array.from(fit.x).map((value) => value.toFixed(6)),
  "after",
  fit.iterations,
  "iterations",
);

// Tails and quantiles.
console.log("P(chi-square(3) > 7.81):", chiSquareUpperTail(7.81, 3).toFixed(4));
console.log("P(t(10) < -2.228):", studentLowerTail(-2.228, 10).toFixed(4));
console.log("The 97.5th percentile of the normal distribution:", normalQuantile(0.975).toFixed(4));

// Two raters grade ten essays on a scale from 0 to 2.
const first = [0, 1, 2, 2, 1, 0, 1, 2, 2, 1];
const second = [0, 1, 2, 1, 1, 0, 2, 2, 2, 1];
console.log("Cohen's kappa:", kappaOfPairs(first, second)?.toFixed(3));
console.log("Quadratic weights:", kappaOfPairs(first, second, "quadratic")?.toFixed(3));

// An invalid argument is a DataError, a full sentence that names the value and says what to do.
try {
  createRandom(1.5);
} catch (error) {
  console.log(error instanceof DataError, error instanceof MeasurekitError, (error as Error).message);
}
