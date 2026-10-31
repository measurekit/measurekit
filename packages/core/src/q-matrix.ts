/**
 * The attributes or skills each item measures, by item id, such as `{ q1: ["fractions"], q2: ["fractions", "borrow"] }`.
 * Both libraries take this shape, so a Q-matrix moves between them unchanged; whether a row may be empty is each
 * library's rule.
 */
// biome-ignore lint/style/useNamingConvention: Q-matrix is the name of this matrix in the field.
export type QMatrix = Readonly<Record<string, readonly string[]>>;
