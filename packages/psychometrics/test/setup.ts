import fc from "fast-check";

// Every property runs from one seed, so that a run on CI draws the inputs that a run on a laptop draws.
fc.configureGlobal({ seed: 20_261_102 });
