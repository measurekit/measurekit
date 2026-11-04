import { describe, expect, it } from "vitest";
import { checkFacetedScores, checkResponseRecords } from "../src/data/internal.js";
import {
  buildResponseMatrix,
  buildResponseMatrixFromColumns,
  buildResponseRecords,
  DataError,
  type FacetedScore,
  itemStatistics,
  type ResponseMatrix,
  scoreResponses,
} from "../src/index.js";

describe("checkFacetedScores", () => {
  const scores: FacetedScore[] = [
    { levels: { person: "p1", item: "essay", rater: "r2" }, score: 3 },
    { levels: { person: "p2", item: "essay", rater: "r1", occasion: "a" }, score: 0.5 },
  ];

  it("accepts scores with an element for each facet, and ignores other facets and the kind of score", () => {
    expect(() => checkFacetedScores(scores, ["person", "item", "rater"])).not.toThrow();
    expect(() => checkFacetedScores(scores, [])).not.toThrow();
    expect(() => checkFacetedScores([], ["person"])).not.toThrow();
  });

  it("names the score and the facet of an element that is missing or not a string", () => {
    expect(() => checkFacetedScores(scores, ["person", "occasion"])).toThrow(
      'Score 1 needs a string element for the facet "occasion", found undefined.',
    );
    expect(() => checkFacetedScores(JSON.parse('[{ "levels": { "person": 1 }, "score": 1 }]'), ["person"])).toThrow(
      'Score 1 needs a string element for the facet "person", found 1.',
    );
    expect(() => checkFacetedScores(scores, ["constructor"])).toThrow(DataError);
  });

  it.each([
    ["a missing score", '[{ "levels": { "person": "p1" } }]', "Score 1 needs a finite score, found undefined."],
    ["a score that is a string", '[{ "levels": { "person": "p1" }, "score": "3" }]', 'found "3"'],
    ["a score that is null", '[{ "levels": { "person": "p1" }, "score": null }]', "found null"],
    [
      "levels that are null",
      '[{ "levels": null, "score": 1 }]',
      "Score 1 needs levels, an object of elements by facet",
    ],
    ["levels that are an array", '[{ "levels": ["p1"], "score": 1 }]', "Score 1 needs levels"],
    ["a score that is a number", "[3]", "Score 1 must be an object { levels, score }, found 3."],
    ["scores that are an object", "{}", "The scores must be an array of objects { levels, score }, found an object."],
  ])("rejects %s from JSON", (_, json, message) => {
    expect(() => checkFacetedScores(JSON.parse(json), ["person"])).toThrow(message);
  });

  it("rejects a score that is not finite", () => {
    expect(() => checkFacetedScores([{ levels: { person: "p1" }, score: Number.NaN }], ["person"])).toThrow(
      "Score 1 needs a finite score, found NaN.",
    );
    expect(() =>
      checkFacetedScores([{ levels: { person: "p1" }, score: Number.POSITIVE_INFINITY }], ["person"]),
    ).toThrow("found Infinity");
  });
});

describe("buildResponseMatrix", () => {
  it("builds a matrix with default ids", () => {
    expect(
      buildResponseMatrix([
        [1, 0, null],
        [0, 2, 1],
      ]),
    ).toEqual({
      personIds: ["person1", "person2"],
      itemIds: ["item1", "item2", "item3"],
      responses: [
        [1, 0, null],
        [0, 2, 1],
      ],
    });
  });

  it("keeps the given ids and copies the rows", () => {
    const rows = [[1], [0]];
    const matrix = buildResponseMatrix(rows, { personIds: ["ana", "budi"], itemIds: ["q1"] });
    rows[0]?.splice(0, 1, 0);
    expect(matrix).toEqual({ personIds: ["ana", "budi"], itemIds: ["q1"], responses: [[1], [0]] });
  });

  it("builds the same matrix from columns", () => {
    const fromColumns = buildResponseMatrixFromColumns(
      [
        [1, 0],
        [0, 2],
        [null, 1],
      ],
      { itemIds: ["a", "b", "c"] },
    );
    expect(fromColumns).toEqual(
      buildResponseMatrix(
        [
          [1, 0, null],
          [0, 2, 1],
        ],
        { itemIds: ["a", "b", "c"] },
      ),
    );
  });

  it.each([
    ["no persons", [], {}, "at least one person and one item"],
    ["no items", [[]], {}, "at least one person and one item"],
    ["rows of unequal length", [[1, 0], [1]], {}, 'Person "person2" has 1 responses, but there are 2 item ids.'],
    ["a negative response", [[1, -1]], {}, "found -1"],
    ["a fractional response", [[0.5]], {}, "found 0.5"],
    [
      "a response above 2³¹ - 1, which 32-bit arrays cannot hold",
      [[1, 2 ** 32 + 1]],
      {},
      'The response of person "person1" to item "item2" must be an integer from 0 to 2147483647 or null, found 4294967297.',
    ],
    ["too few person ids", [[1], [0]], { personIds: ["a"] }, "Found 2 rows of responses and 1 person ids."],
    ["repeated item ids", [[1, 0]], { itemIds: ["q", "q"] }, 'The item id "q" appears more than once.'],
  ])("rejects %s", (_, rows, options, message) => {
    expect(() => buildResponseMatrix(rows, options)).toThrow(DataError);
    expect(() => buildResponseMatrix(rows, options)).toThrow(message);
  });

  it("rejects NaN, infinities and values of other types", () => {
    expect(() => buildResponseMatrix([[1, Number.NaN]])).toThrow("found NaN");
    for (const json of ["[[1, 1e999]]", '[[1, "1"]]', "[[1, true]]"]) {
      expect(() => buildResponseMatrix(JSON.parse(json))).toThrow("must be an integer from 0 to 2147483647 or null");
    }
  });

  it("rejects holes in sparse arrays", () => {
    const row: (number | null)[] = [1];
    row.length = 2;
    expect(() => buildResponseMatrix([row])).toThrow("found undefined");
    expect(() => buildResponseMatrixFromColumns([[1, 0], row])).toThrow(
      "Column 2 must be an array of responses, found an array with a hole at index 1.",
    );
  });

  it("rejects a hole among the rows or the ids, which forEach would skip", () => {
    const rows: (number | null)[][] = [[1, 0]];
    rows[2] = [0, 1];
    expect(() => buildResponseMatrix(rows)).toThrow(
      "The rows of a response matrix must be an array, found an array with a hole at index 1.",
    );
    const matrix = { personIds: ["a", "b", "c"], itemIds: ["q1", "q2"], responses: rows };
    expect(() => itemStatistics(matrix)).toThrow('The responses of person "b" must be an array.');
    const personIds: string[] = ["a"];
    personIds.length = 2;
    expect(() => buildResponseMatrix([[1], [0]], { personIds })).toThrow(
      "Every person id must be a string, found undefined.",
    );
    const times: (number | null)[][] = [[1]];
    times.length = 2;
    expect(() => buildResponseMatrix([[1], [0]], { times })).toThrow(
      'Person "person2" needs one time per item, 1 times.',
    );
  });

  it("checks that a first row or column is an array before its length sizes the ids or the rows", () => {
    // A length of 2³² would make an array that JavaScript cannot hold, and a smaller one an array of that many ids.
    const object = JSON.parse('[{ "length": 4294967296 }]');
    expect(() => buildResponseMatrix(object)).toThrow('The responses of person "person1" must be an array.');
    expect(() => buildResponseMatrixFromColumns(object)).toThrow(
      "Column 1 must be an array of responses, found an object.",
    );
  });

  it("rejects matrices from JSON that are another value, lack an array or hold ids of another type", () => {
    const parse = (json: string): ResponseMatrix => JSON.parse(json);
    for (const json of [
      "null",
      '"q1"',
      '{"personIds": ["a"], "itemIds": ["q1"]}',
      '{"itemIds": ["q1"], "responses": [[1]]}',
    ]) {
      expect(() => itemStatistics(parse(json))).toThrow(
        "A response matrix needs the arrays personIds, itemIds and responses.",
      );
    }
    expect(() => itemStatistics(parse('{"personIds": ["a"], "itemIds": ["q1"], "responses": [1]}'))).toThrow(
      'The responses of person "a" must be an array.',
    );
    expect(() => itemStatistics(parse('{"personIds": ["a"], "itemIds": [1], "responses": [[1]]}'))).toThrow(
      "Every item id must be a string, found 1.",
    );
  });

  it("rejects columns of unequal length", () => {
    expect(() => buildResponseMatrixFromColumns([[1, 0], [1]])).toThrow("Column 2 has 1 responses, column 1 has 2.");
  });

  it("rejects rows, columns and options of other types from JavaScript", () => {
    expect(() => buildResponseMatrix(JSON.parse("null"))).toThrow(
      "The rows of a response matrix must be an array, found null.",
    );
    expect(() => buildResponseMatrix([[1]], JSON.parse("null"))).toThrow("The options must be an object, found null.");
    expect(() => buildResponseMatrix([[1]], JSON.parse('{"verbose":1}'))).toThrow('There is no option "verbose"');
    expect(() => buildResponseMatrix([[1]], { times: JSON.parse("3") })).toThrow(
      "A response matrix needs one row of times per person, 1 rows.",
    );
    expect(() => buildResponseMatrixFromColumns(JSON.parse("[null]"))).toThrow(
      "Column 1 must be an array of responses, found null.",
    );
  });
});

describe("buildResponseRecords", () => {
  it("takes the persons and items in the order they first appear, and copies the records", () => {
    const input = [
      { person: "ana", item: "q2", response: 1 },
      { person: "budi", item: "q1", response: 0 },
      { person: "ana", item: "q1", response: 2 },
    ];
    const data = buildResponseRecords(input);
    expect(data).toEqual({ personIds: ["ana", "budi"], itemIds: ["q2", "q1"], records: input });
    expect(data.records[0]).not.toBe(input[0]);
    const listed = buildResponseRecords(input, { personIds: ["budi", "ana", "citra"], itemIds: ["q1", "q2", "q3"] });
    expect(listed.personIds).toEqual(["budi", "ana", "citra"]);
  });

  it("rejects records outside the ids, invalid responses, a repeated record and repeated ids", () => {
    const record = { person: "ana", item: "q1", response: 1 };
    expect(() => buildResponseRecords([record], { itemIds: ["q2"] })).toThrow(
      'Record 1 needs a person and an item among the ids, found person "ana" and item "q1".',
    );
    expect(() => buildResponseRecords([{ ...record, response: -1 }])).toThrow(
      'The response of person "ana" to item "q1" must be an integer from 0 to 2147483647, found -1.',
    );
    expect(() => buildResponseRecords([{ ...record, response: 2 ** 31 }])).toThrow(
      'The response of person "ana" to item "q1" must be an integer from 0 to 2147483647, found 2147483648.',
    );
    expect(() => buildResponseRecords([record, { ...record, response: 0 }])).toThrow(
      'Person "ana" has two records of item "q1"; a person answers an item once.',
    );
    expect(() => buildResponseRecords([record], { personIds: ["ana", "ana"] })).toThrow(
      'The person id "ana" appears more than once.',
    );
    expect(() => buildResponseRecords(JSON.parse('[{ "person": 1, "item": "q1", "response": 1 }]'))).toThrow(DataError);
    expect(() => buildResponseRecords(JSON.parse("null"))).toThrow("Response records must be an array, found null.");
    expect(() => buildResponseRecords(JSON.parse("[null]"))).toThrow(
      "Record 1 must be an object { person, item, response }, found null.",
    );
  });

  it("rejects a hole among the records or the ids of records from JavaScript", () => {
    const records = [{ person: "ana", item: "q1", response: 1 }];
    records.length = 2;
    expect(() => buildResponseRecords(records)).toThrow(
      "Response records must be an array, found an array with a hole at index 1.",
    );
    const data = { personIds: ["ana"], itemIds: ["q1"], records };
    expect(() => checkResponseRecords(data)).toThrow(
      "Record 2 must be an object { person, item, response }, found undefined.",
    );
    const itemIds = ["q1"];
    itemIds.length = 2;
    expect(() => buildResponseRecords([{ person: "ana", item: "q1", response: 1 }], { itemIds })).toThrow(
      "Every item id must be a string, found undefined.",
    );
  });
});

describe("scoreResponses", () => {
  const choices = [
    ["A", "C", null],
    ["B", "C", "D"],
  ];

  it("scores the key 1, other options 0 and missing choices null, and keeps the choices", () => {
    expect(scoreResponses(choices, ["A", "C", "B"], { itemIds: ["q1", "q2", "q3"] })).toEqual({
      personIds: ["person1", "person2"],
      itemIds: ["q1", "q2", "q3"],
      responses: [
        [1, 1, null],
        [0, 1, 0],
      ],
      choices,
      key: ["A", "C", "B"],
      optionLabels: ["A", "B", "C", "D"],
    });
  });

  it("keeps the given option labels, including options nobody chose", () => {
    const scored = scoreResponses(choices, ["A", "C", "B"], { optionLabels: ["D", "C", "B", "A", "E"] });
    expect(scored.optionLabels).toEqual(["D", "C", "B", "A", "E"]);
  });

  it("accepts more labels than it scans before it asks the set, in the choices and in the given labels", () => {
    const labels = Array.from("ABCDEFGHIJKL");
    const shifted = [...labels.slice(6), ...labels.slice(0, 6)];
    const rows = [labels, [...labels].reverse(), shifted];
    const key = [...labels].reverse();
    const scored = scoreResponses(rows, key);
    expect(scored.responses).toEqual([
      labels.map((label, item) => (label === key[item] ? 1 : 0)),
      labels.map(() => 1),
      shifted.map((label, item) => (label === key[item] ? 1 : 0)),
    ]);
    expect(scored.optionLabels).toEqual(labels);
    expect(scoreResponses(rows, key, { optionLabels: labels }).responses).toEqual(scored.responses);
    expect(() => scoreResponses([[...labels.slice(0, 11), "M"]], key, { optionLabels: labels })).toThrow(
      'row 1, item 12, found "M"',
    );
  });

  it("checks the ids that are given and the response times against the scores", () => {
    expect(() => scoreResponses(choices, ["A", "C", "B"], { personIds: ["p", "p"] })).toThrow('"p" appears more than');
    expect(() => scoreResponses(choices, ["A", "C", "B"], { itemIds: ["q1", "q2"] })).toThrow(
      "but there are 2 item ids",
    );
    expect(() =>
      scoreResponses(choices, ["A", "C", "B"], {
        times: [
          [1, 1, 1],
          [1, 1, 1],
        ],
      }),
    ).toThrow("needs a response");
  });

  it.each([
    ["a choice outside the labels", choices, ["A", "C", "B"], ["A", "B", "C"], 'row 2, item 3, found "D"'],
    ["a key outside the labels", choices, ["A", "C", "E"], ["A", "B", "C", "D"], 'the key of item 3, found "E"'],
    ["an empty choice", [["", "A"]], ["A", "A"], undefined, 'Expected an option label in row 1, item 1, found "".'],
    ["repeated labels", choices, ["A", "C", "B"], ["A", "A", "B", "C", "D"], "Option labels must be unique."],
    ["a row of the wrong length", [["A"]], ["A", "B"], undefined, "Row 1 has 1 choices, but the key has 2 items."],
    [
      "a row that is not an array",
      JSON.parse('["AB"]'),
      ["A", "B"],
      undefined,
      "Row 1 of the choices must be an array",
    ],
    ["a key label that is not a string", [["A"]], JSON.parse("[1]"), undefined, "the key of item 1, found 1."],
    ["option labels that are not an array", choices, ["A", "C", "B"], JSON.parse('"ABCD"'), "must be an array"],
    [
      "an empty option label",
      choices,
      ["A", "C", "B"],
      ["A", "B", "C", "D", ""],
      'Expected an option label in option label 5, found "".',
    ],
    [
      "an option label that is not a string",
      choices,
      ["A", "C", "B"],
      JSON.parse('["A", "B", "C", "D", 5]'),
      "Expected an option label in option label 5, found 5.",
    ],
  ])("rejects %s", (_, rows, key, optionLabels, message) => {
    const options = optionLabels ? { optionLabels } : {};
    expect(() => scoreResponses(rows, key, options)).toThrow(DataError);
    expect(() => scoreResponses(rows, key, options)).toThrow(message);
  });
});
