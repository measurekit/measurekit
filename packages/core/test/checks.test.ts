import { describe, expect, it } from "vitest";
import {
  checkArray,
  checkBoolean,
  checkChoice,
  checkFinite,
  checkIntegerFrom,
  checkMethod,
  checkNonNegative,
  checkOptions,
  checkPositive,
  checkPositiveInteger,
  checkRandom,
  checkRecord,
  checkSeed,
  checkStopping,
  checkUnitInterval,
  createRandom,
  DataError,
  describe as describeValue,
} from "../src/index.js";

/** Parses JSON so that a test passes a value of another type as JavaScript or JSON would, without a cast. */
const json = (text: string): unknown => JSON.parse(text);

describe("describe", () => {
  it("quotes strings and names arrays, functions and objects", () => {
    expect(describeValue("a")).toBe('"a"');
    expect(describeValue([1])).toBe("an array");
    expect(describeValue(() => 1)).toBe("a function");
    expect(describeValue({})).toBe("an object");
    expect(describeValue(null)).toBe("null");
    expect(describeValue(undefined)).toBe("undefined");
    expect(describeValue(1.5)).toBe("1.5");
  });
});

describe("checks of one value", () => {
  it("accept a valid value and return nothing", () => {
    expect(() => {
      checkRecord({}, "A record");
      checkArray([], "An array");
      checkOptions({});
      checkMethod({ stops: () => true }, "stops", "A rule");
      checkRandom(createRandom(1));
      checkBoolean(false, "A flag");
      checkChoice("a", "A choice", ["a", "b"]);
      checkFinite(-1, "A number");
      checkPositive(0.5, "A number");
      checkNonNegative(0, "A number");
      checkPositiveInteger(1, "A number");
      checkIntegerFrom(2, "A number", 2);
      checkUnitInterval(1, "A number");
      checkSeed(-3);
      checkStopping(0, 1);
    }).not.toThrow();
  });

  it("throw DataError with a sentence that names the label and the value found", () => {
    expect(() => checkRecord(json("[]"), "The states must be an object")).toThrow(
      "The states must be an object, found an array.",
    );
    expect(() => checkArray(json("null"), "The items must be an array")).toThrow(
      "The items must be an array, found null.",
    );
    expect(() => checkOptions(json("null"))).toThrow("The options must be an object, found null.");
    expect(() => checkMethod({}, "stops", "The option stopping must be an object with a stops method")).toThrow(
      "The option stopping must be an object with a stops method, found an object.",
    );
    expect(() => checkRandom({ uniform: () => 0.5 })).toThrow(
      "A random number generator must be one from createRandom, found an object.",
    );
    expect(() => checkBoolean(json("1"), "The option purify")).toThrow("The option purify must be a boolean, found 1.");
    expect(() => checkChoice("c", "The option routing", ["a", "b"])).toThrow(
      'The option routing must be "a" or "b", found "c".',
    );
    expect(() => checkFinite(json('"5"'), "Theta")).toThrow('Theta must be a finite number, found "5".');
    expect(() => checkPositive(0, "The option sd")).toThrow("The option sd must be a positive number, found 0.");
    expect(() => checkNonNegative(-1, "A rating")).toThrow("A rating must be a non-negative number, found -1.");
    expect(() => checkPositiveInteger(1.5, "The option maxIterations")).toThrow(
      "The option maxIterations must be a positive integer, found 1.5.",
    );
    expect(() => checkIntegerFrom(1, "The option points", 2)).toThrow(
      "The option points must be an integer from 2, found 1.",
    );
    expect(() => checkIntegerFrom(json('"3"'), "The option points", 2)).toThrow(
      'The option points must be an integer from 2, found "3".',
    );
    expect(() => checkIntegerFrom(2.5, "The option points", 2)).toThrow(
      "The option points must be an integer from 2, found 2.5.",
    );
    expect(() => checkUnitInterval(2, "A probability")).toThrow("A probability must be a number from 0 to 1, found 2.");
    expect(() => checkSeed(1.5)).toThrow("A seed must be a safe integer, found 1.5.");
  });

  it("refuse an array with a hole, which forEach, map and every would skip unseen", () => {
    const sparse: unknown[] = [1];
    sparse.length = 3;
    expect(() => checkArray(sparse, "The items must be an array")).toThrow(
      "The items must be an array, found an array with a hole at index 1.",
    );
    // biome-ignore lint/suspicious/noSparseArray: the hole is what the check refuses.
    expect(() => checkArray([1, , 3], "The items must be an array")).toThrow("found an array with a hole at index 1.");
    expect(() => checkArray([undefined, null], "The items must be an array")).not.toThrow();
  });

  it("throw DataError for every other type of input", () => {
    for (const value of [json("null"), json('"x"'), json("[]"), json("{}"), Number.NaN, Number.POSITIVE_INFINITY]) {
      expect(() => checkFinite(value, "A number")).toThrow(DataError);
    }
  });

  it("check the stopping options together", () => {
    expect(() => checkStopping(-1, 10)).toThrow("The option tolerance must be a non-negative number, found -1.");
    expect(() => checkStopping(1e-6, 0)).toThrow("The option maxIterations must be a positive integer, found 0.");
  });
});
