import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { bundle, entrySize, entrySources, moduleOf } from "../entry-sizes.ts";
import { ENTRY_BUDGETS } from "../size-budgets.ts";

const PACKAGES = resolve(import.meta.dirname, "../../packages");
const ENTRIES = entrySources();
/** Bundling the root entry points takes about a second each, longer on a loaded machine. */
const BUNDLE_TIMEOUT = 30_000;

describe("moduleOf", () => {
  it("names a file of core by its package, and a file of a library by its folder or its file", () => {
    expect(moduleOf(resolve(PACKAGES, "core/src/bfgs.ts"))).toBe("core");
    expect(moduleOf(resolve(PACKAGES, "psychometrics/src/irt/calibration.ts"))).toBe("psychometrics/irt");
    expect(moduleOf(resolve(PACKAGES, "psychometrics/src/checks.ts"))).toBe("psychometrics/checks");
    expect(moduleOf(resolve(PACKAGES, "knowledge-tracing/src/model.ts"))).toBe("knowledge-tracing/model");
  });
});

describe("bundle", () => {
  const Core = JSON.stringify(ENTRIES.get("@measurekit/core"));
  const Ctt = JSON.stringify(ENTRIES.get("@measurekit/psychometrics/ctt"));

  it(
    "counts a module that a static import loads, and leaves out one that a dynamic import() loads later in a chunk of its own",
    async () => {
      const statically = await bundle(`export * from ${Core}; export * from ${Ctt};`);
      const dynamically = await bundle(`export * from ${Core}; export const load = () => import(${Ctt});`);
      expect(statically.modules).toContain("psychometrics/ctt");
      expect(dynamically.modules).toEqual(["core"]);
      expect(dynamically.gzip).toBeLessThan(statically.gzip);
    },
    BUNDLE_TIMEOUT,
  );
});

describe("every entry point alone", () => {
  it("has a budget for each entry point of the packages, and none for a name that is not one", () => {
    expect(Object.keys(ENTRY_BUDGETS).sort()).toEqual([...ENTRIES.keys()].sort());
  });

  it.each([...ENTRIES])(
    "keeps %s within its budget, holding only the modules it lists",
    async (specifier, source) => {
      const { gzip, holds } = await entrySize(source);
      const budget = ENTRY_BUDGETS[specifier] ?? expect.unreachable(`No budget for ${specifier}.`);
      expect(gzip, `The gzipped bytes of ${specifier}`).toBeLessThan(budget.gzip);
      expect(holds, `The modules other than its own in ${specifier}`).toEqual(budget.holds);
    },
    BUNDLE_TIMEOUT,
  );
});

describe("a named import from the root of a library", () => {
  // sideEffects: false lets a bundler drop what a program does not use, so that the root costs what the subpath does.
  it.each([
    ["@measurekit/psychometrics", "cronbachAlpha", "@measurekit/psychometrics/ctt"],
    ["@measurekit/psychometrics", "calibrate", "@measurekit/psychometrics/irt"],
    ["@measurekit/knowledge-tracing", "bkt", "@measurekit/knowledge-tracing/bkt"],
  ])(
    "holds from %s only the modules that %s needs, those of %s",
    async (root, name, subpath) => {
      const { modules } = await bundle(`export { ${name} } from ${JSON.stringify(ENTRIES.get(root))};`);
      const allowed = [...(ENTRY_BUDGETS[subpath]?.holds ?? []), subpath.replace("@measurekit/", "")];
      expect(modules.filter((module) => !allowed.includes(module))).toEqual([]);
    },
    BUNDLE_TIMEOUT,
  );
});
