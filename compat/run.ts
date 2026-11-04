import { runAll } from "./scenarios.js";

// Node.js, Deno and Bun read the results as the last line of what the bundle prints, as JSON.
console.log(JSON.stringify(runAll()));
