import { deepStrictEqual } from "node:assert/strict";
import { type ChildProcess, spawn } from "node:child_process";
import { createRequire } from "node:module";
import { connect, createServer } from "node:net";
import { join } from "node:path";
import { setTimeout as sleep } from "node:timers/promises";
import { build, type Metafile } from "esbuild";
import type { Results } from "./scenarios.js";

/** The root of the repository: the metafile of a bundle names its inputs by their path from it. */
const ROOT = join(import.meta.dirname, "..");
const DIST = join(import.meta.dirname, ".dist");
/** The workerd program: the package of that name exports the path of the binary for the platform. */
const WORKERD = (createRequire(import.meta.url)("workerd") as { readonly default: string }).default;
/** How long workerd has to listen, and the time between two tries. */
const LISTEN_LIMIT_MS = 15_000;
const LISTEN_TRY_MS = 50;

/** A file of a package that is not in its `dist`, such as a source, and a file that is. */
const NOT_SHIPPED = /^packages\/[^/]+\/(?!dist\/)/;
const SHIPPED = /^packages\/[^/]+\/dist\//;

/**
 * Throws when the bundles hold a file of a package that the package does not ship, or hold none that it ships, so that the
 * pattern above could not tell the one from the other.
 */
function assertShipped({ inputs }: Metafile): void {
  const files = Object.keys(inputs);
  const unshipped = files.filter((file) => NOT_SHIPPED.test(file));
  if (unshipped.length > 0) {
    throw new Error(
      `The bundles hold ${unshipped.length} files that the packages do not ship, such as ${unshipped[0]}.`,
    );
  }
  if (!files.some((file) => SHIPPED.test(file))) {
    throw new Error("The bundles hold no file of the dist of a package, so this check cannot tell what they hold.");
  }
}

/**
 * Bundles the scenarios into one file for each way a runtime runs them: `run.js` for Node.js, Deno and Bun, which print the
 * results, and `worker.js` for workerd, which answers a request with them. The packages are bundled as they ship, from their
 * `dist/` through their `exports`, and no tsconfig is read, so that none of the `paths` that point the type checker at the
 * sources applies. The platform is the browser's, so that a module of Node.js in a package fails the bundle: a Worker has none.
 */
async function bundle(): Promise<void> {
  const { metafile } = await build({
    entryPoints: { run: join(import.meta.dirname, "run.ts"), worker: join(import.meta.dirname, "worker.ts") },
    absWorkingDir: ROOT,
    outdir: DIST,
    bundle: true,
    format: "esm",
    target: "es2022",
    platform: "browser",
    tsconfigRaw: {},
    metafile: true,
    logLevel: "warning",
  });
  assertShipped(metafile);
}

/** Runs a command and returns what it printed. */
function execute(command: string, args: readonly string[]): Promise<string> {
  return new Promise((done, fail) => {
    const child = spawn(command, args, { stdio: ["ignore", "pipe", "pipe"] });
    let out = "";
    let err = "";
    child.stdout.on("data", (chunk: Buffer) => {
      out += chunk;
    });
    child.stderr.on("data", (chunk: Buffer) => {
      err += chunk;
    });
    child.on("error", (error) => fail(new Error(`${command} could not start: ${error.message}`)));
    child.on("close", (code) => {
      if (code === 0) done(out);
      else fail(new Error(`${command} exited with ${code}: ${err.trim()}`));
    });
  });
}

/** A runtime: how to ask its version, and how to run the bundle in it. */
interface Runtime {
  readonly version: () => Promise<string>;
  readonly run: () => Promise<Results>;
}

const script = join(DIST, "run.js");
const fromCommand = async (command: string, args: readonly string[]): Promise<Results> =>
  JSON.parse((await execute(command, args)).trim().split("\n").at(-1) ?? "");

/** Finds a port that nothing listens on. */
function freePort(): Promise<number> {
  return new Promise((done, fail) => {
    const probe = createServer();
    probe.on("error", fail);
    probe.listen(0, "127.0.0.1", () => {
      const address = probe.address();
      probe.close(() =>
        address !== null && typeof address === "object" ? done(address.port) : fail(new Error("No port.")),
      );
    });
  });
}

/** Whether something accepts connections at the port. */
function listening(port: number): Promise<boolean> {
  return new Promise((done) => {
    const socket = connect(port, "127.0.0.1");
    socket.on("connect", () => socket.end(() => done(true)));
    socket.on("error", () => done(false));
  });
}

/** Waits until the child listens at the port, or it exits, or the limit passes. */
async function untilListening(child: ChildProcess, port: number, output: () => string): Promise<void> {
  let exited = false;
  child.once("exit", () => {
    exited = true;
  });
  for (let waited = 0; waited < LISTEN_LIMIT_MS; waited += LISTEN_TRY_MS) {
    if (exited) throw new Error(`workerd exited before it listened: ${output().trim()}`);
    if (await listening(port)) return;
    await sleep(LISTEN_TRY_MS);
  }
  throw new Error(`workerd did not listen within ${LISTEN_LIMIT_MS} ms: ${output().trim()}`);
}

/** Runs the worker in workerd and asks it once for the results. */
async function runOnWorkerd(): Promise<Results> {
  const port = await freePort();
  const child = spawn(
    WORKERD,
    ["serve", `--socket-addr=http=127.0.0.1:${port}`, join(import.meta.dirname, "worker.capnp")],
    { stdio: ["ignore", "ignore", "pipe"] },
  );
  let log = "";
  child.stderr.on("data", (chunk: Buffer) => {
    log += chunk;
  });
  try {
    await untilListening(child, port, () => log);
    const response = await fetch(`http://127.0.0.1:${port}/`);
    if (!response.ok)
      throw new Error(`The worker answered ${response.status}: ${(await response.text()).slice(0, 400)} ${log.trim()}`);
    return (await response.json()) as Results;
  } finally {
    child.kill();
  }
}

const RUNTIMES: Readonly<Record<string, Runtime>> = {
  node: {
    version: async () => `node ${(await execute("node", ["--version"])).trim()}`,
    run: () => fromCommand("node", [script]),
  },
  deno: {
    version: async () => (await execute("deno", ["--version"])).split("\n")[0]?.trim() ?? "deno",
    // No permission is granted: the packages use no platform API, and a script that reached for one would be refused.
    run: () => fromCommand("deno", ["run", script]),
  },
  bun: {
    version: async () => `bun ${(await execute("bun", ["--version"])).trim()}`,
    run: () => fromCommand("bun", ["run", script]),
  },
  workerd: {
    version: async () => (await execute(WORKERD, ["--version"])).trim(),
    run: runOnWorkerd,
  },
};

/** Runs the bundle on the runtimes, and checks the results against what the packages promise and against Node.js. */
async function main(names: readonly string[]): Promise<number> {
  const selected = names.length === 0 ? Object.keys(RUNTIMES) : names;
  const unknown = selected.find((name) => RUNTIMES[name] === undefined);
  if (unknown !== undefined) {
    throw new Error(`Unknown runtime ${unknown}; the runtimes are ${Object.keys(RUNTIMES).join(", ")}.`);
  }
  await bundle();
  const reference = await RUNTIMES.node?.run();
  deepStrictEqual(
    [reference?.calibration.converged, reference?.tracing.converged],
    [true, true],
    "Node.js does not give the results that the packages promise: both fits converge.",
  );
  let failures = 0;
  for (const name of selected) {
    const runtime = RUNTIMES[name];
    if (runtime === undefined) continue;
    try {
      const version = await runtime.version();
      deepStrictEqual(await runtime.run(), reference);
      console.log(`pass  ${name.padEnd(8)} ${version}`);
    } catch (error) {
      failures += 1;
      console.error(`FAIL  ${name.padEnd(8)} ${error instanceof Error ? error.message : String(error)}`);
    }
  }
  console.log(failures === 0 ? `The packages run the same on ${selected.join(", ")}.` : `${failures} runtimes differ.`);
  return failures === 0 ? 0 : 1;
}

process.exitCode = await main(process.argv.slice(2));
