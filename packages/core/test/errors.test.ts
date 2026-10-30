import { describe, expect, it } from "vitest";
import { DataError, MeasurekitError } from "../src/index.js";

describe("errors", () => {
  it("makes DataError a MeasurekitError and an Error, named for its class", () => {
    const error = new DataError("A value is wrong.");
    expect(error).toBeInstanceOf(DataError);
    expect(error).toBeInstanceOf(MeasurekitError);
    expect(error).toBeInstanceOf(Error);
    expect(error).toHaveProperty("name", "DataError");
    expect(error.message).toBe("A value is wrong.");
  });

  it("names the base class for itself", () => {
    expect(new MeasurekitError("Failed.")).toHaveProperty("name", "MeasurekitError");
  });
});
