// Helpers that the modules and tests of this package share. They are not exported from the package.

export { checkDichotomous } from "./dichotomous.js";
export { checkFacetedScores, type FacetedScore } from "./faceted-scores.js";
export { indexIds } from "./ids.js";
export { checkMatrixItems, MAX_MATRIX_ITEMS } from "./item-matrices.js";
export { isResponse, MAX_RESPONSE } from "./response.js";
export { checkResponseMatrix, type ResponseMatrix, responseMatrix } from "./response-matrix.js";
export { checkResponseRecords, type ResponseRecord, type ResponseRecords } from "./response-records.js";
export { type ScoredResponses, scoreResponses } from "./scoring.js";
