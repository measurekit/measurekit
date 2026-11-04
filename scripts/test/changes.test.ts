import { execFileSync, spawnSync } from "node:child_process";
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { parse } from "yaml";

/** The program of the step of the `changes` job that decides what a push reaches, which the step runs with `node -`. */
function changesProgram(): string {
  const workflow = parse(readFileSync(join(import.meta.dirname, "../../.github/workflows/ci.yml"), "utf8")) as {
    readonly jobs: { readonly changes: { readonly steps: readonly { readonly id?: string; readonly run?: string }[] } };
  };
  const run = workflow.jobs.changes.steps.find((step) => step.id === "filter")?.run ?? "";
  return /node - <<'EOF'\n([\s\S]*?)\nEOF/u.exec(run)?.[1] ?? expect.unreachable("The step runs no program.");
}

/** The members of the workspace, as their manifests say: what they use, and whether they have benchmarks. */
const MANIFESTS: Readonly<Record<string, object>> = {
  "packages/core": { name: "@measurekit/core", scripts: { bench: "x" } },
  "packages/psychometrics": {
    name: "@measurekit/psychometrics",
    dependencies: { "@measurekit/core": "workspace:^" },
    scripts: { bench: "x" },
  },
  "packages/knowledge-tracing": {
    name: "@measurekit/knowledge-tracing",
    dependencies: { "@measurekit/core": "workspace:^" },
    scripts: { bench: "x" },
  },
  examples: {
    name: "measurekit-examples",
    dependencies: {
      "@measurekit/core": "workspace:*",
      "@measurekit/knowledge-tracing": "workspace:*",
      "@measurekit/psychometrics": "workspace:*",
    },
  },
  compat: {
    name: "@measurekit/compat",
    dependencies: { "@measurekit/knowledge-tracing": "workspace:*", "@measurekit/psychometrics": "workspace:*" },
  },
};

const EVERY_PROJECT = ["core", "examples", "knowledge-tracing", "psychometrics", "scripts"];

/** What a change to a file reaches: the Vitest projects, whether the benchmarks run, whether the runtimes do. */
type Reach = readonly [projects: readonly string[], bench: boolean, runtimes: boolean];

const CASES: readonly (readonly [file: string, reach: Reach])[] = [
  ["packages/core/src/errors.ts", [EVERY_PROJECT, true, true]],
  ["packages/psychometrics/src/irt/index.ts", [["examples", "psychometrics", "scripts"], true, true]],
  ["packages/knowledge-tracing/README.md", [["examples", "knowledge-tracing", "scripts"], true, true]],
  ["docs/psychometrics/irt.md", [["examples"], false, false]],
  ["README.md", [["examples"], false, false]],
  ["AGENTS.md", [["scripts"], false, false]],
  [".changeset/pending.md", [["scripts"], false, false]],
  [".github/workflows/rules.yml", [["scripts"], false, false]],
  ["scripts/check-rules.ts", [["scripts"], false, false]],
  ["scripts/guide-samples.ts", [["examples", "scripts"], false, false]],
  ["scripts/stated-values.ts", [["examples", "scripts"], false, false]],
  ["scripts/test/versions.test.ts", [["scripts"], false, false]],
  ["examples/psychometrics/item-analysis.ts", [["examples"], false, false]],
  ["validation/benchmarks/input.json", [[], true, false]],
  ["compat/check.ts", [[], false, true]],
  ["compat/package.json", [["scripts"], false, true]],
  ["docs/core/performance.md", [[], false, false]],
  ["docs/psychometrics/performance.md", [[], false, false]],
  ["CONTRIBUTING.md", [[], false, false]],
  ["biome.json", [[], false, false]],
  ["validation/psychometrics/fixtures.R", [[], false, false]],
  [".github/workflows/ci.yml", [EVERY_PROJECT, true, true]],
  ["pnpm-lock.yaml", [EVERY_PROJECT, true, true]],
  ["package.json", [EVERY_PROJECT, true, true]],
  ["tsconfig.base.json", [EVERY_PROJECT, true, true]],
  ["vitest.config.ts", [EVERY_PROJECT, true, true]],
  ["scripts/bench-harness.ts", [EVERY_PROJECT, true, true]],
  ["a-file-no-rule-names.txt", [EVERY_PROJECT, true, true]],
];

describe("the changes job of ci.yml", () => {
  const program = changesProgram();
  let repository = "";
  let base = "";

  const git = (...args: string[]): string =>
    execFileSync("git", ["-c", "user.name=test", "-c", "user.email=test@example.com", ...args], {
      cwd: repository,
      encoding: "utf8",
    }).trim();

  /** Runs the program on a change since `since`, as the job does, and reads what it prints. */
  function reach(since: string): Reach {
    const run = spawnSync(process.execPath, ["-"], {
      cwd: repository,
      input: program,
      encoding: "utf8",
      // biome-ignore lint/style/useNamingConvention: these are the names of the variables.
      env: { ...process.env, BASE: since, GITHUB_OUTPUT: "" },
    });
    expect(run.status, run.stderr).toBe(0);
    const outputs = new Map(
      run.stdout
        .trim()
        .split("\n")
        .map((line) => line.split("=", 2) as [string, string]),
    );
    const projects: string[] = JSON.parse(outputs.get("projects") ?? "null");
    return [projects.sort(), outputs.get("bench") === "true", outputs.get("runtimes") === "true"];
  }

  beforeAll(() => {
    repository = mkdtempSync(join(tmpdir(), "measurekit-changes-"));
    git("init", "--quiet");
    for (const [directory, manifest] of Object.entries(MANIFESTS)) {
      mkdirSync(join(repository, directory), { recursive: true });
      writeFileSync(join(repository, directory, "package.json"), JSON.stringify(manifest));
    }
    git("add", "--all");
    git("commit", "--quiet", "--message", "base");
    base = git("rev-parse", "HEAD");
  });

  afterAll(() => rmSync(repository, { recursive: true, force: true }));

  it.each(CASES)("reaches the jobs that read %s and no other", (file, [projects, bench, runtimes]) => {
    mkdirSync(dirname(join(repository, file)), { recursive: true });
    writeFileSync(join(repository, file), file.endsWith(".json") ? "{}" : "changed\n");
    git("add", "--all");
    git("commit", "--quiet", "--message", file);
    try {
      expect(reach(base)).toStrictEqual([[...projects].sort(), bench, runtimes]);
    } finally {
      git("reset", "--quiet", "--hard", base);
    }
  });

  it("reaches the union of what two files reach, runs every job when nothing can be compared, and none when nothing changed", () => {
    expect(reach(base)).toStrictEqual([[], false, false]);
    expect(reach("0".repeat(40))).toStrictEqual([EVERY_PROJECT, true, true]);
    expect(reach("")).toStrictEqual([EVERY_PROJECT, true, true]);
    for (const file of ["docs/psychometrics/irt.md", "packages/knowledge-tracing/src/index.ts"]) {
      mkdirSync(dirname(join(repository, file)), { recursive: true });
      writeFileSync(join(repository, file), "changed\n");
    }
    git("add", "--all");
    git("commit", "--quiet", "--message", "two files");
    expect(reach(base)).toStrictEqual([["examples", "knowledge-tracing", "scripts"], true, true]);
  });
});
