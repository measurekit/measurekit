/**
 * The result that a sample shows after a statement, `statement; // => value, what it means`, and the comparison with what the
 * statement gives. The value is a literal of JavaScript, read up to a `,`, a `;` or a `:` that ends it:
 *
 * | Shown                   | Matches                                                                                  |
 * | ----------------------- | ---------------------------------------------------------------------------------------- |
 * | `0.68`, `1.1e-15`       | a number that rounds to it at the digits shown; an integer must be exact                 |
 * | `"q2"`, `null`, `true`  | the same value                                                                           |
 * | `[6, -1, 9]`            | an array or a typed array of that length, element by element                             |
 * | `[6, ..., 9]`           | the elements before `...` at the start and those after it at the end, of any length     |
 * | `{ a: 1, b: [2] }`      | an object with exactly those keys; a `...` among the entries allows more keys           |
 */

/** A value that a result states. */
export type Stated =
  | { readonly kind: "number"; readonly value: number; readonly tolerance: number }
  | { readonly kind: "string"; readonly value: string }
  | { readonly kind: "literal"; readonly value: boolean | null | undefined }
  | {
      readonly kind: "array";
      readonly items: readonly Stated[];
      /** Whether a `...` stands among the elements, and the elements after it, which the end of the array holds. */
      readonly open: boolean;
      readonly tail: readonly Stated[];
    }
  | {
      readonly kind: "object";
      readonly entries: readonly (readonly [string, Stated])[];
      readonly open: boolean;
    };

/** What a result states: the value, and the text after it. */
export interface Reading {
  readonly value: Stated;
  readonly rest: string;
}

/** What may follow a value: the end of the result, or a mark that starts the explanation. */
const EXPLANATION = /^\s*(?:$|[,;:])/;
const NUMBER = /^-?\d+(?:\.(\d+))?(?:[eE]([-+]?\d+))?/;
const STRING = /^"((?:[^"\\]|\\.)*)"/;
const KEY = /^(?:([A-Za-z_$][\w$]*)|"((?:[^"\\]|\\.)*)")\s*:\s*/;
/**
 * The relative tolerance of every number: a double keeps about 16 digits, and the last of them may differ between platforms,
 * so a number shown with all its digits is compared to 1e-12 of it.
 */
const FULL_PRECISION = 1e-12;

/** The tolerance of a number as it is written: half a unit of its last digit, and an integer exactly. */
function toleranceOf(written: RegExpExecArray): number {
  const [, decimals, exponent] = written;
  const value = Math.abs(Number(written[0]));
  const power = Number(exponent ?? 0) - (decimals?.length ?? 0);
  // A division by an exact power of ten rounds once, as `10 ** -4` does not on every version of V8.
  const half = power < 0 ? 0.5 / 10 ** -power : 0.5 * 10 ** power;
  const shown = decimals === undefined && exponent === undefined ? 0 : half;
  return Math.max(shown, FULL_PRECISION * value);
}

/** Reads the value of a result, the text after `// =>`. Throws when it does not begin with a value, or runs on without a mark. */
export function readStated(result: string): Reading {
  const text = result.trim();
  const reader = new Reader(text);
  const value = reader.value();
  const rest = reader.rest();
  if (!EXPLANATION.test(rest)) {
    throw new Error(
      `The result "${result}" has a value that "${rest.trim()}" follows; end the value with a ",", ";" or ":".`,
    );
  }
  return { value, rest: rest.replace(/^\s*[,;:]\s*/, "") };
}

class Reader {
  private position = 0;
  private readonly text: string;

  constructor(text: string) {
    this.text = text;
  }

  rest(): string {
    return this.text.slice(this.position);
  }

  private skip(): void {
    while (this.text[this.position] === " ") this.position++;
  }

  private take(pattern: RegExp): RegExpExecArray | null {
    const match = pattern.exec(this.text.slice(this.position));
    if (match !== null) this.position += match[0].length;
    return match;
  }

  private fail(expected: string): never {
    throw new Error(
      `The result "${this.text}" does not state a value: expected ${expected} at "${this.rest().slice(0, 20)}".`,
    );
  }

  value(): Stated {
    this.skip();
    const number = this.take(NUMBER);
    if (number !== null) return { kind: "number", value: Number(number[0]), tolerance: toleranceOf(number) };
    const string = this.take(STRING);
    if (string !== null) return { kind: "string", value: JSON.parse(string[0]) as string };
    const word = this.take(/^(?:null|true|false|undefined)(?![\w$])/);
    if (word !== null)
      return { kind: "literal", value: { null: null, true: true, false: false, undefined: undefined }[word[0]] };
    if (this.take(/^\[/) !== null) return this.array();
    if (this.take(/^\{/) !== null) return this.object();
    return this.fail("a number, a string, null, true, false, undefined, an array or an object");
  }

  private array(): Stated {
    const items: Stated[] = [];
    const tail: Stated[] = [];
    let open = false;
    for (this.skip(); this.take(/^\]/) === null; this.skip()) {
      if (this.take(/^\.\.\./) !== null) {
        if (open) this.fail("one ... at most");
        open = true;
      } else (open ? tail : items).push(this.value());
      this.skip();
      if (this.take(/^,/) === null && this.text[this.position] !== "]") this.fail('"," or "]"');
    }
    return { kind: "array", items, open, tail };
  }

  private object(): Stated {
    const entries: [string, Stated][] = [];
    let open = false;
    for (this.skip(); this.take(/^\}/) === null; this.skip()) {
      if (this.take(/^\.\.\./) !== null) {
        open = true;
      } else {
        const key = this.take(KEY);
        if (key === null) this.fail("a key");
        entries.push([key[1] ?? (JSON.parse(`"${key[2] ?? ""}"`) as string), this.value()]);
      }
      this.skip();
      if (this.take(/^,/) === null && this.text[this.position] !== "}") this.fail('"," or "}"');
    }
    return { kind: "object", entries, open };
  }
}

/** Writes a stated value as the result gave it, for a message. */
function show(stated: Stated): string {
  switch (stated.kind) {
    case "number":
    case "string":
    case "literal":
      return JSON.stringify(stated.value) ?? "undefined";
    case "array":
      return `[${[...stated.items.map(show), ...(stated.open ? ["..."] : []), ...stated.tail.map(show)].join(", ")}]`;
    case "object":
      return `{ ${[...stated.entries.map(([key, value]) => `${key}: ${show(value)}`), ...(stated.open ? ["..."] : [])].join(", ")} }`;
  }
}

/** Writes what the code gave, shortened, for a message. */
function describeActual(actual: unknown): string {
  const text =
    JSON.stringify(actual, (_, value: unknown) =>
      typeof value === "number" && !Number.isFinite(value) ? String(value) : value,
    ) ?? String(actual);
  return text.length > 200 ? `${text.slice(0, 200)}...` : text;
}

/** The array that the code gave, or the elements of a typed array; `undefined` for anything else. */
function listOf(actual: unknown): readonly unknown[] | undefined {
  if (ArrayBuffer.isView(actual)) return Array.from(actual as unknown as ArrayLike<unknown>);
  return Array.isArray(actual) ? actual : undefined;
}

/** Returns what differs between an array that the code gave and one stated, with the path to it. */
function arrayDifferences(
  actual: unknown,
  stated: Extract<Stated, { kind: "array" }>,
  path: string,
): string | undefined {
  const list = listOf(actual);
  if (list === undefined) return `${path} is ${describeActual(actual)}, not an array`;
  const shown = stated.items.length + stated.tail.length;
  if (stated.open ? list.length < shown : list.length !== shown) {
    return `${path} has ${list.length} elements, not ${stated.open ? "at least " : ""}${shown}`;
  }
  const placed = [
    ...stated.items.map((item, index) => [item, index] as const),
    ...stated.tail.map((item, index) => [item, list.length - stated.tail.length + index] as const),
  ];
  for (const [item, index] of placed) {
    const difference = differences(list[index], item, `${path}[${index}]`);
    if (difference !== undefined) return difference;
  }
  return undefined;
}

/** Returns what differs between an object that the code gave and one stated, with the path to it. */
function objectDifferences(
  actual: unknown,
  stated: Extract<Stated, { kind: "object" }>,
  path: string,
): string | undefined {
  if (typeof actual !== "object" || actual === null || Array.isArray(actual))
    return `${path} is ${describeActual(actual)}, not an object`;
  const record = actual as Record<string, unknown>;
  for (const [key, value] of stated.entries) {
    if (!(key in record)) return `${path} has no key "${key}"`;
    const difference = differences(record[key], value, `${path}.${key}`);
    if (difference !== undefined) return difference;
  }
  const extra = Object.keys(record).filter((key) => !stated.entries.some(([named]) => named === key));
  return stated.open || extra.length === 0
    ? undefined
    : `${path} also has the keys ${extra.join(", ")}; add "..." to the result to leave them out`;
}

/** Returns what differs between a stated value and an actual one, with the path to it, or `undefined` when they agree. */
export function differences(actual: unknown, stated: Stated, path = "the value"): string | undefined {
  switch (stated.kind) {
    case "number":
      if (typeof actual !== "number") return `${path} is ${describeActual(actual)}, not the number ${show(stated)}`;
      return Math.abs(actual - stated.value) <= stated.tolerance
        ? undefined
        : `${path} is ${actual}, not ${show(stated)}`;
    case "string":
    case "literal":
      return Object.is(actual, stated.value) ? undefined : `${path} is ${describeActual(actual)}, not ${show(stated)}`;
    case "array":
      return arrayDifferences(actual, stated, path);
    case "object":
      return objectDifferences(actual, stated, path);
  }
}

/**
 * Compares what a statement gives with the result that its sample shows, and returns it for the code that follows. Throws an
 * error that names the line and the first difference.
 */
export function stated<T>(actual: T, result: string, where: string): T {
  const difference = differences(actual, readStated(result).value);
  if (difference !== undefined)
    throw new Error(`${where}: the result "${result}" is not what the code gives: ${difference}.`);
  return actual;
}
