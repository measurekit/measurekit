import { describe, expect, it } from "vitest";
import { differences, readStated, stated } from "./stated-values.js";

/** The first difference between what the code gives and what the comment states, `undefined` when they agree. */
function compare(actual: unknown, comment: string): string | undefined {
  const reading = readStated(comment);
  if (reading === undefined) throw new Error(`"${comment}" states no value`);
  return differences(actual, reading.value);
}

describe("reading a comment", () => {
  it("reads a number, a string, null, true, false and undefined, and the explanation that follows", () => {
    expect(readStated("0.68, the probability")?.rest).toBe("the probability");
    expect(readStated('"q3"; null when the test has ended')?.rest).toBe("null when the test has ended");
    expect(readStated("1000: each draw holds two thetas")?.rest).toBe("each draw holds two thetas");
    expect(readStated("null")?.rest).toBe("");
    expect(readStated("true")?.value).toEqual({ kind: "literal", value: true });
    expect(readStated("undefined")?.value).toEqual({ kind: "literal", value: undefined });
    expect(readStated("-1.5e-3")?.value).toEqual({ kind: "number", value: -0.0015, decimals: 12 });
  });

  it("reads arrays and objects, nested, with unquoted or quoted keys and a trailing comma", () => {
    expect(readStated('{ group: 0.5, "persons": [0.5, null,], }')?.value).toEqual({
      kind: "object",
      open: false,
      entries: [
        ["group", { kind: "number", value: 0.5, decimals: 1 }],
        [
          "persons",
          {
            kind: "array",
            open: false,
            items: [
              { kind: "number", value: 0.5, decimals: 1 },
              { kind: "literal", value: null },
            ],
          },
        ],
      ],
    });
  });

  it("says that a comment states no value when it does not begin like one", () => {
    expect(readStated("the id of the first item")).toBeUndefined();
    expect(readStated("nullable when the test has ended")).toBeUndefined();
    expect(readStated("")).toBeUndefined();
  });

  it("refuses a comment that begins like a value and is not one, or runs on without a mark", () => {
    expect(() => readStated("0.68 the probability")).toThrow('starts with a value that "the probability" follows');
    expect(() => readStated("[1, 2")).toThrow("does not state a value");
    expect(() => readStated("{ a 1 }")).toThrow("expected a key");
    expect(() => readStated("[1 2]")).toThrow('expected "," or "]"');
  });
});

describe("comparing a value with its comment", () => {
  it("accepts a number that rounds to the stated decimals and refuses one that does not", () => {
    expect(compare(0.6803, "0.68")).toBeUndefined();
    expect(compare(0.6851, "0.68")).toBe("the value is 0.6851, not 0.68");
    expect(compare(-1.5594, "-1.56")).toBeUndefined();
    expect(compare(543.93, "543.9")).toBeUndefined();
  });

  it("holds an integer to its exact value and a number with all its digits to 1e-12 of it", () => {
    expect(compare(500, "500")).toBeUndefined();
    expect(compare(500.3, "500")).toBe("the value is 500.3, not 500");
    expect(compare(0.7451612903225806, "0.7451612903225805")).toBeUndefined();
    expect(compare(0.74516129033, "0.7451612903225805")).toBe("the value is 0.74516129033, not 0.7451612903225805");
  });

  it("compares strings, null and booleans for identity", () => {
    expect(compare("q2", '"q2"')).toBeUndefined();
    expect(compare("q3", '"q2"')).toBe('the value is "q3", not "q2"');
    expect(compare(null, "null")).toBeUndefined();
    expect(compare(0, "null")).toBe("the value is 0, not null");
    expect(compare(false, "true")).toBe("the value is false, not true");
    expect(compare("1", "1")).toBe('the value is "1", not the number 1');
  });

  it("compares arrays element by element, and typed arrays as arrays", () => {
    expect(compare([6, -1, 9], "[6, -1, 9]")).toBeUndefined();
    expect(compare(new Float64Array([1, 2]), "[1, 2]")).toBeUndefined();
    expect(compare([6, -1], "[6, -1, 9]")).toBe("the value has 2 elements, not 3");
    expect(compare([6, 2, 9], "[6, -1, 9]")).toBe("the value[1] is 2, not -1");
    expect(compare("x", "[1]")).toBe('the value is "x", not an array');
  });

  it("lets a final ... leave out the rest of an array and a ... among the entries leave out the rest of an object", () => {
    expect(compare([1, 2, 3], "[1, ...]")).toBeUndefined();
    expect(compare([1], "[1, 2, ...]")).toBe("the value has 1 elements, not at least 2");
    expect(compare({ a: 1, b: 2 }, "{ a: 1, ... }")).toBeUndefined();
    expect(compare([{ name: "x", estimate: 0.67, se: 1 }], '[{ name: "x", estimate: 0.67, ... }]')).toBeUndefined();
  });

  it("requires an object to have exactly the stated keys without ...", () => {
    expect(compare({ a: 1, b: 2 }, "{ a: 1 }")).toBe(
      'the value also has the keys b; add "..." to the comment to leave them out',
    );
    expect(compare({ a: 1 }, "{ a: 1, b: 2 }")).toBe('the value has no key "b"');
    expect(compare({ a: { b: 3 } }, "{ a: { b: 2 } }")).toBe("the value.a.b is 3, not 2");
    expect(compare([1], "{ a: 1 }")).toBe("the value is [1], not an object");
  });

  it("names the place and the comment in the error that the code of an example throws", () => {
    expect(stated(0.6803, "0.68, the probability", "elo.ts:12")).toBe(0.6803);
    expect(stated(7, "the id of the first item", "elo.ts:13")).toBe(7);
    expect(() => stated(0.5, "0.68, the probability", "elo.ts:12")).toThrow(
      'elo.ts:12: the comment "0.68, the probability" states a value that the code does not give: the value is 0.5, not 0.68.',
    );
  });
});
