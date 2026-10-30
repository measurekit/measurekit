/**
 * Response matrices, response records in long format for sparse data, and scoring of multiple-choice responses
 * against a key.
 *
 * @module @measurekit/psychometrics/data
 */
export {
  type ResponseMatrix,
  type ResponseMatrixOptions,
  responseMatrix,
  responseMatrixFromColumns,
} from "./response-matrix.js";
export {
  type ResponseRecord,
  type ResponseRecords,
  type ResponseRecordsOptions,
  responseRecords,
} from "./response-records.js";
export { type ScoredResponses, type ScoringOptions, scoreResponses } from "./scoring.js";
