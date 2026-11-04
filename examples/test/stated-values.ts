/**
 * The values that a comment states after an expression, `expression; // value, what it means`, and the comparison with
 * what the expression gives. The value is a literal of JavaScript, read up to a `,`, a `;` or a `:` that ends it:
 *
 * | Stated                  | Matches                                                                         |
 * | ----------------------- | ------------------------------------------------------------------------------- |
 * | `0.68`                  | a number that rounds to it at the decimals shown; an integer must be exact      |
 * | `"q2"`, `null`, `true`  | the same value                                                                  |
 * | `[6, -1, 9]`            | an array of that length, element by element; a final `...` allows more         |
 * | `{ a: 1, b: [2] }`      | an object with exactly those keys; a `...` among the entries allows more keys   |
 *
 * A comment that does not begin like a value, such as `// the id of the first item`, states none.
 */

/** A value that a comment states. */
export type Stated =
  | { readonly kind: "number"; readonly value: number; readonly decimals: number }
  | { readonly kind: "string"; readonly value: string }
  | { readonly kind: "literal"; readonly value: boolean | null | undefined }
  | { readonly kind: "array"; readonly items: readonly Stated[]; readonly open: boolean }
  | {
      readonly kind: "object";
      readonly entries: readonly (readonly [string, Stated])[];
      readonly open: boolean;
    };

/** What a comment states: the value, and the text after it. */
export interface Reading {
  readonly value: Stated;
  readonly rest: string;
}

/** Where a comment begins like a value: a digit, a sign, a quote, a bracket or a word of JavaScript's values. */
const VALUE_START = /^(?:[-\d"[{]|(?:null|true|false|undefined)(?![\w$]))/;
/** What may follow a value: the end of the comment, or a mark that starts the explanation. */
const EXPLANATION = /^\s*(?:$|[,;:])/;
const NUMBER = /^-?\d+(?:\.(\d+))?(?:[eE][-+]?\d+)?/;
const STRING = /^"((?:[^"\\]|\\.)*)"/;
const KEY = /^(?:([A-Za-z_$][\w$]*)|"((?:[^"\\]|\\.)*)")\s*:\s*/;
/** The relative tolerance for a number that shows more decimals than a double keeps across platforms. */
const FULL_PRECISION = 1e-12;
/** The decimals from which a stated number is compared at full precision. */
const FULL_PRECISION_DECIMALS = 12;

/** Reads the value that opens a comment, `undefined` when the comment states none. Throws when it starts like one and is not. */
export function readStated(comment: string): Reading | undefined {
  const text = comment.trim();
  if (!VALUE_START.test(text)) return undefined;
  const reader = new Reader(text);
  const value = reader.value();
  const rest = reader.rest();
  if (!EXPLANATION.test(rest)) {
    throw new Error(
      `The comment "${comment}" starts with a value that "${rest.trim()}" follows; end the value with a ",", ";" or ":".`,
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
      `The comment "${this.text}" does not state a value: expected ${expected} at "${this.rest().slice(0, 20)}".`,
    );
  }

  value(): Stated {
    this.skip();
    const number = this.take(NUMBER);
    if (number !== null)
      return {
        kind: "number",
        value: Number(number[0]),
        decimals: /[eE]/.test(number[0]) ? FULL_PRECISION_DECIMALS : (number[1]?.length ?? 0),
      };
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
    let open = false;
    for (this.skip(); this.take(/^\]/) === null; this.skip()) {
      if (this.take(/^\.\.\./) !== null) open = true;
      else items.push(this.value());
      this.skip();
      if (this.take(/^,/) === null && this.text[this.position] !== "]") this.fail('"," or "]"');
    }
    return { kind: "array", items, open };
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

/** Writes a stated value as the comment gave it, for a message. */
function show(stated: Stated): string {
  switch (stated.kind) {
    case "number":
    case "string":
    case "literal":
      return JSON.stringify(stated.value) ?? "undefined";
    case "array":
      return `[${[...stated.items.map(show), ...(stated.open ? ["..."] : [])].join(", ")}]`;
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

/** Returns what differs between a stated value and an actual one, with the path to it, or `undefined` when they agree. */
export function differences(actual: unknown, stated: Stated, path = "the value"): string | undefined {
  switch (stated.kind) {
    case "number": {
      if (typeof actual !== "number") return `${path} is ${describeActual(actual)}, not the number ${show(stated)}`;
      const tolerance =
        stated.decimals === 0
          ? 1e-9
          : stated.decimals >= FULL_PRECISION_DECIMALS
            ? FULL_PRECISION * Math.abs(stated.value)
            : 0.5 * 10 ** -stated.decimals;
      return Math.abs(actual - stated.value) <= tolerance ? undefined : `${path} is ${actual}, not ${show(stated)}`;
    }
    case "string":
    case "literal":
      return Object.is(actual, stated.value) ? undefined : `${path} is ${describeActual(actual)}, not ${show(stated)}`;
    case "array": {
      const list = ArrayBuffer.isView(actual) ? Array.from(actual as unknown as ArrayLike<unknown>) : actual;
      if (!Array.isArray(list)) return `${path} is ${describeActual(actual)}, not an array`;
      if (stated.open ? list.length < stated.items.length : list.length !== stated.items.length) {
        return `${path} has ${list.length} elements, not ${stated.open ? "at least " : ""}${stated.items.length}`;
      }
      for (const [index, item] of stated.items.entries()) {
        const difference = differences(list[index], item, `${path}[${index}]`);
        if (difference !== undefined) return difference;
      }
      return undefined;
    }
    case "object": {
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
        : `${path} also has the keys ${extra.join(", ")}; add "..." to the comment to leave them out`;
    }
  }
}

/**
 * Compares the value of an expression with the value its comment states, and returns the value for the code that
 * follows. A comment that states none passes. Throws an error that names the line and the first difference.
 */
export function stated<T>(actual: T, comment: string, where: string): T {
  const reading = readStated(comment);
  const difference = reading === undefined ? undefined : differences(actual, reading.value);
  if (difference !== undefined)
    throw new Error(`${where}: the comment "${comment}" states a value that the code does not give: ${difference}.`);
  return actual;
}
