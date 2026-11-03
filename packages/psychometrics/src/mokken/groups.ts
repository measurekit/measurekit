/**
 * Returns the smallest number of persons in a group of the rest score checks, by mokken's rule: a third of the persons
 * up to 250, a fifth up to 499, a tenth from 500; at least 50 below 150 persons.
 */
export function defaultGroupSize(persons: number): number {
  if (persons < 150) return 50;
  if (persons <= 250) return Math.floor(persons / 3);
  return persons >= 500 ? Math.floor(persons / 10) : Math.floor(persons / 5);
}

/** The groups of persons that a rest score defines. */
export interface RestScoreGroups {
  /** The number of groups. */
  readonly count: number;
  /** The group of each person, from 0, the lowest rest scores first. */
  readonly members: Int32Array;
}

/**
 * Splits persons into groups of similar rest scores, as mokken does: sorted rest scores are cut after the person at each
 * multiple of the minimum size, moved on to the end of any tie, so that persons with one rest score share a group, and
 * the last group takes what is left, at least the minimum size.
 *
 * @param rest - The rest score of each person.
 * @param minimumSize - The fewest persons in a group, at most the number of persons.
 */
export function restScoreGroups(rest: Float64Array, minimumSize: number): RestScoreGroups {
  const persons = rest.length;
  const sorted = Float64Array.from(rest).sort();
  // The number of persons up to the end of the tie that holds the person at each position, from the last position back.
  const tieEnds = new Int32Array(persons);
  for (let position = persons - 1; position >= 0; position--) {
    const tied = position + 1 < persons && sorted[position + 1] === sorted[position];
    tieEnds[position] = tied ? (tieEnds[position + 1] ?? persons) : position + 1;
  }
  const endOfTie = (position: number): number => tieEnds[position] ?? persons;
  const cuts = [endOfTie(minimumSize - 1)];
  while (persons - (cuts.at(-1) ?? 0) >= minimumSize) cuts.push(endOfTie(minimumSize + (cuts.at(-1) ?? 0) - 1));
  cuts.pop();
  const bounds = Float64Array.from(cuts, (cut) => sorted[cut - 1] ?? 0);
  const members = Int32Array.from(rest, (score) => boundsBelow(bounds, score));
  return { count: bounds.length + 1, members };
}

/** Returns the number of the ascending bounds below a score, by bisection, so that a person costs the log of the groups. */
function boundsBelow(bounds: Float64Array, score: number): number {
  let low = 0;
  let high = bounds.length;
  while (low < high) {
    const middle = (low + high) >> 1;
    if ((bounds[middle] ?? 0) < score) low = middle + 1;
    else high = middle;
  }
  return low;
}
