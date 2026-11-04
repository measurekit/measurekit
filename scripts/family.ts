/**
 * The identity of the family: its site, its sentence, the identity of each package and the marks of others that its texts
 * name. `check:standards`, the TypeDoc plugin of `docs/typedoc-seo.ts`, `scripts/llms.ts` and `scripts/check-site.ts` read
 * this file and nothing else of the family, so another family copies them unchanged and writes its own copy of this file.
 */

/**
 * The base of every URL of the docs site, with its trailing slash: the `hostedBaseUrl` of TypeDoc, the `homepage` of each
 * package, the links of the READMEs, the sitemap, `robots.txt` and the `CNAME` of the site; the owner's domain (2026-11-04).
 * Changing it is this one line, and `check:standards` then lists each homepage, README link and citation that names the old one.
 */
export const SITE = "https://measurekit.wakan.dev/";

/** The repository, which a link to a file that the docs site does not publish points into, on its default branch. */
export const REPOSITORY = "https://github.com/measurekit/measurekit";

/** The name of the family, its sentence and the `author` of every package. */
export const FAMILY = {
  name: "measurekit",
  sentence:
    "Psychometrics and knowledge tracing for JavaScript and TypeScript, validated against R, pyBKT and RSMTool: IRT, CTT, adaptive testing, cognitive diagnosis, BKT.",
  author: "Fariz Wakan (https://github.com/luthfanwakan)",
} as const;

/** The identity of one package: what npm, the README, the docs site and `llms.txt` say of it. */
export interface PackageIdentity {
  /** The name on npm, which is also the folder below `packages/` and `docs/` after the scope. */
  readonly name: string;
  /** The standard and its version that the package implements, which the title of each of its pages carries. */
  readonly label?: string;
  /** The identity sentence: the `description` of `package.json` and the first sentence of the README. */
  readonly description: string;
  /** The `keywords` of `package.json`, as people type them. */
  readonly keywords: readonly string[];
}

/**
 * Every package of the family, in the order of the README's table. No package implements a standard of another body, so none
 * has a label: the title of a page names the subject and the package.
 */
export const PACKAGES: readonly PackageIdentity[] = [
  {
    name: "@measurekit/psychometrics",
    description:
      "Psychometrics for JavaScript and TypeScript, validated against R and RSMTool: item response theory including multidimensional models, classical test theory, adaptive testing, cognitive diagnosis, differential item functioning, equating, generalizability theory, standard setting, rater agreement, the evaluation of automated scores and answer similarity.",
    keywords: [
      "psychometrics",
      "classical-test-theory",
      "item-response-theory",
      "irt",
      "multidimensional-irt",
      "computerized-adaptive-testing",
      "multistage-testing",
      "testlets",
      "item-analysis",
      "reliability",
      "cronbach-alpha",
      "subscores",
      "assessment",
      "item-fit",
      "differential-item-functioning",
      "mantel-haenszel",
      "polytomous-items",
      "test-equating",
      "explanatory-irt",
      "generalizability-theory",
      "mokken-scale-analysis",
      "variance-components",
      "score-scaling",
      "scale-scores",
      "percentile-ranks",
      "simulation",
      "standard-setting",
      "rater-agreement",
      "many-facet-rasch",
      "rater-severity",
      "cohen-kappa",
      "percent-agreement",
      "intraclass-correlation",
      "response-times",
      "speed-accuracy",
      "rapid-guessing",
      "answer-copying",
      "answer-similarity",
      "test-security",
      "cognitive-diagnosis",
      "gdina",
      "acdm",
      "rrum",
      "dimensionality",
      "parallel-analysis",
      "polychoric-correlation",
      "tetrachoric-correlation",
      "automated-scoring",
      "rasch-model",
      "graded-response-model",
      "partial-credit-model",
      "test-theory",
      "statistics",
      "typescript",
    ],
  },
  {
    name: "@measurekit/knowledge-tracing",
    description:
      "Knowledge tracing for JavaScript and TypeScript, validated against pyBKT and R: Bayesian knowledge tracing, AFM and PFA, Elo, online IRT and knowledge space theory, with evaluation, mastery criteria and practice recommendations.",
    keywords: [
      "knowledge-tracing",
      "bayesian-knowledge-tracing",
      "bkt",
      "additive-factors-model",
      "performance-factors-analysis",
      "logistic-knowledge-tracing",
      "learner-model",
      "student-model",
      "elo",
      "computerized-adaptive-testing",
      "irt",
      "item-response-theory",
      "adaptive-learning",
      "adaptive-practice",
      "mastery-learning",
      "educational-data-mining",
      "learning-analytics",
      "learning-curve",
      "cross-validation",
      "q-matrix",
      "knowledge-space-theory",
      "learning-space",
      "adaptive-assessment",
      "student-modeling",
      "typescript",
    ],
  },
  {
    name: "@measurekit/core",
    description:
      "The shared foundation of the measurekit packages: errors, argument checks, a seeded random number generator and numerics. Install @measurekit/psychometrics or @measurekit/knowledge-tracing.",
    keywords: ["measurekit", "psychometrics", "knowledge-tracing", "seeded-random"],
  },
];

/** A body whose standards the family implements: how a text names it, and the lines that it asks for. */
export interface Holder {
  /** The name in the line that the family is not affiliated with it. */
  readonly name: string;
  /** Words that name the body or a standard of it that holds no mark of its own, so that a text that has one carries the line. */
  readonly named: RegExp;
  /** The line that names the body's own mark, where it asks for one. */
  readonly trademark?: string;
  /** The line of each of its marks, with `{mark}` where the mark goes. */
  readonly markLine?: string;
}

/**
 * The bodies, in the order of their lines. ETS publishes the DIF categories that `@measurekit/psychometrics/dif` gives and the
 * acceptance thresholds of automated scores that `rater` applies; its guidelines for third parties ask for the registered
 * mark and the line that a work is not endorsed or approved by it. Products of others that the texts name, such as R packages,
 * pyBKT and RSMTool, are named plainly, with no line.
 */
export const HOLDERS: Readonly<Record<string, Holder>> = {
  ets: {
    name: "Educational Testing Service (ETS)",
    named: /\bEducational Testing Service\b/,
    markLine: "{mark} is a registered trademark of Educational Testing Service (ETS).",
  },
};

/** A mark of another, as its holder lists it, and how a text names it. */
export interface Mark {
  /** The mark with its symbol, as the holder writes it. */
  readonly mark: string;
  /** The key of its holder in `HOLDERS`. */
  readonly holder: string;
  /** Words that name it. */
  readonly named: RegExp;
}

/** The marks that the texts of the family name, in the order of their lines. */
export const MARKS: readonly Mark[] = [{ mark: "ETS®", holder: "ets", named: /\bETS\b/ }];
