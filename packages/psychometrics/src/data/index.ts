/**
 * Response matrices, response records in long format for sparse data, faceted scores, and scoring of multiple-choice
 * responses against a key.
 *
 * @module @measurekit/psychometrics/data
 */
export type { FacetedScore } from "./faceted-scores.js";
export {
  buildResponseMatrix,
  buildResponseMatrixFromColumns,
  type ResponseMatrix,
  type ResponseMatrixOptions,
} from "./response-matrix.js";
export {
  buildResponseRecords,
  type ResponseRecord,
  type ResponseRecords,
  type ResponseRecordsOptions,
} from "./response-records.js";
export { type ScoredResponses, type ScoringOptions, scoreResponses } from "./scoring.js";
