/**
 * A matrix in compressed sparse row form, as the design matrices of learner models are: mostly indicators. The
 * nonzero entries of row i are at positions `rowStarts[i]` to `rowStarts[i + 1] - 1` of `columns` and `values`.
 */
export interface SparseMatrix {
  /** The number of columns. */
  readonly columnCount: number;
  /** Where each row starts in `columns` and `values`, with the number of entries last: one more than the rows. */
  readonly rowStarts: Int32Array;
  /** The column of each entry. */
  readonly columns: Int32Array;
  /** The value of each entry. */
  readonly values: Float64Array;
}
