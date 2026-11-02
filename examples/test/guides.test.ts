import { mkdirSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { afterAll, afterEach, describe, it, vi } from "vitest";

/** The root of the repository. */
const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..", "..");

/** The folder that holds the samples while they run, which git ignores. */
const SAMPLES = join(ROOT, "examples", ".samples");

/** A sample: `continues` runs it after the sample before it, and `no-run` shows code that needs a worker or a package of the reader's. */
const FENCE = /^```ts( continues| no-run)?\n([\s\S]*?)^```$/gm;

/** One import of named bindings, the only form the samples use. */
const IMPORT = /^import\s+(?:type\s+)?\{([^}]*)\}\s+from\s+"([^"]+)";?\s*$/gm;

/** The documents with samples: the guides of each library and the READMEs. */
const DOCUMENTS = [
  ...["psychometrics", "knowledge-tracing"].flatMap((name) =>
    readdirSync(join(ROOT, "docs", name))
      .filter((file) => file.endsWith(".md"))
      .map((file) => `docs/${name}/${file}`),
  ),
  "README.md",
  "packages/core/README.md",
  "packages/psychometrics/README.md",
  "packages/knowledge-tracing/README.md",
];

/** A run of samples of a document that share their variables, from the line of the first. */
interface Chain {
  readonly document: string;
  readonly line: number;
  readonly samples: string[];
}

function chainsOf(document: string): Chain[] {
  const text = readFileSync(join(ROOT, document), "utf8");
  const chains: Chain[] = [];
  for (const { 0: whole, 1: mark, 2: code = "", index } of text.matchAll(FENCE)) {
    if (mark === " no-run" || whole === undefined) continue;
    if (mark === " continues") chains.at(-1)?.samples.push(code);
    else chains.push({ document, line: text.slice(0, index).split("\n").length, samples: [code] });
  }
  return chains;
}

/** Returns the samples as one module: their imports merged, and each body in a block inside the block of the one before. */
function moduleOf({ samples }: Chain): string {
  const modules = new Map<string, Set<string>>();
  const bodies = samples.map((code) =>
    code
      .replace(IMPORT, (statement: string, names: string, from: string) => {
        const bindings = modules.get(from) ?? new Set<string>();
        for (const name of names.split(",").map((part) => part.trim())) {
          if (name !== "" && !name.startsWith("type ") && !statement.startsWith("import type")) bindings.add(name);
        }
        modules.set(from, bindings);
        return "";
      })
      .replace(/^export /gm, ""),
  );
  const imports = [...modules].map(([from, bindings]) => `import { ${[...bindings].join(", ")} } from "${from}";`);
  return [...imports, ...bodies.map((body) => `{\n${body}`), "}".repeat(bodies.length), ""].join("\n");
}

const chains = DOCUMENTS.flatMap(chainsOf);

describe("the samples of the guides and READMEs", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  afterAll(() => {
    rmSync(SAMPLES, { recursive: true, force: true });
  });

  it.each(chains)("runs the sample of $document from line $line to its end", async (chain) => {
    vi.spyOn(console, "log").mockImplementation(() => {});
    mkdirSync(SAMPLES, { recursive: true });
    const file = join(SAMPLES, `${chain.document.replaceAll(/[/.]/g, "-")}-${chain.line}.ts`);
    writeFileSync(file, moduleOf(chain));
    await import(pathToFileURL(file).href);
  });
});
