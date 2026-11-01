/** The violations of an order or of monotonicity found over a set of comparisons. */
export interface Violations {
  /** The comparisons in which a violation could occur. */
  activeComparisons: number;
  /** The violations, each larger than the minimum violation. */
  violations: number;
  /** The largest violation. */
  maximumViolation: number;
  /** The sum of the violations. */
  violationSum: number;
  /** The largest test statistic of a violation. */
  maximumZ: number;
  /** The violations whose test is significant. */
  significantViolations: number;
}

/** Returns violations without any comparison. */
export function emptyViolations(): Violations {
  return {
    activeComparisons: 0,
    violations: 0,
    maximumViolation: 0,
    violationSum: 0,
    maximumZ: 0,
    significantViolations: 0,
  };
}

/** Counts a violation of the given size. */
export function addViolation(total: Violations, size: number): void {
  total.violations += 1;
  total.violationSum += size;
  total.maximumViolation = Math.max(total.maximumViolation, size);
}
