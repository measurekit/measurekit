import { runAll } from "./scenarios.js";

/** The worker answers a request with the results as JSON. */
// biome-ignore lint/style/noDefaultExport: a Worker module exports its handlers as the default export.
export default {
  fetch(): Response {
    return Response.json(runAll());
  },
};
