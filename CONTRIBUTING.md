# Contributing

Contributions in English or Bahasa Indonesia are welcome. Everyone taking part is expected to follow the
[Code of Conduct](CODE_OF_CONDUCT.md).

```mermaid
flowchart LR
  subgraph repo["The repository"]
    packages["packages/core, psychometrics,<br/>knowledge-tracing<br/>src, test"]
    docs["docs/psychometrics, docs/knowledge-tracing<br/>guides that run in tests"]
    validation["validation<br/>R and Python references<br/>that write the fixtures"]
    examples["examples<br/>run by a test"]
    tooling["scripts, compat"]
  end
  change["Your change"] --> reference["A new statistic: its reference<br/>and fixture in validation/"]
  change --> local["Local gates<br/>lint, check:standards, check:duplication, typecheck, test, knip"]
  reference --> local
  local --> hooks["Pre-commit: Biome formats<br/>Pre-push: typecheck, test, knip"]
  hooks --> pr["Pull request<br/>with a changeset"]
  pr --> ci["CI: check, then the jobs that read<br/>the packages the change reaches"]
  ci --> review["Review"]
  review --> merge["Merge"]
  repo -.- change
```

## Setup

Node.js 22.18+ and pnpm 12 are required. One command sets the repository up, and in a git clone it also installs the
Lefthook git hooks:

```bash
pnpm install
```

[AGENTS.md](AGENTS.md) lists the commands and the conventions for layout, naming, code, docs, evidence and tests;
[psychometrics](docs/psychometrics/architecture.md) and [knowledge tracing](docs/knowledge-tracing/architecture.md)
state the design principles of each package.

## Validation

A new or changed statistic needs a reference and a fixture: extend the script for its module in
`validation/<package>/` and regenerate the fixtures as [validation/README.md](validation/README.md) describes. Name
the reference function or the formula in the pull request, and add the reference and its tolerance to the package's
`docs/<package>/validation.md`. Commit the regenerated fixtures; CI runs only the TypeScript tests.

## Before opening a pull request

Run the commands of [AGENTS.md](AGENTS.md#commands); `pnpm check:rules packages/<package>` limits the mutation rules to the
package you changed. A bug fix comes with a test that fails without it. A change to a package goes into the pending
release's changeset for that package in `.changeset/`, or a new one made with `pnpm changeset` when none is pending
([AGENTS.md](AGENTS.md#releases)).

The pre-commit hook formats staged code, JSON, Markdown and YAML with Biome; the pre-push hook runs the type check,
tests and Knip. CI runs only the jobs that a change reaches: it tests each package, the scripts that check the repository and the
examples on Node.js 22 and 24, with the coverage floors of core and knowledge tracing, and runs the packages on Deno, Bun and
workerd. The mutation rules and the coverage floor of psychometrics, which take
long, run each week and before a release, so that per-push CI stays short.

## License of contributions

measurekit is licensed under the [Apache License 2.0](LICENSE). As its section 5 states, a contribution you submit for
inclusion is licensed under the same terms, unless you state otherwise. There is no contributor license agreement;
you keep the copyright in your contributions.
