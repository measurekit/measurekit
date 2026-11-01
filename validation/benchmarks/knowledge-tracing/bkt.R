# Bayesian knowledge tracing: the fit of the skills of a log by EM, and the replay of the log through the fitted model,
# against pyBKT, which benchmarks.py times on the same log. pyBKT fits every skill, five starts each by default, with
# the processes of its parallel option off; measurekit fits every skill with its default of five starts.

source("validation/benchmarks/simulation.R")

set.seed(20261029)
learners <- 200
skill_count <- 5
attempts <- 20
log <- simulate_bkt(learners, skill_count, attempts)

bench_fixture("knowledge-tracing", "bkt", list(
  source = "A simulation, set.seed(20261029): Bayesian knowledge tracing with a prior of 0.3, learning 0.15, slip 0.1 and guess 0.2.",
  learner = I(log$learner), skill = I(log$skill), item = I(log$item), time = I(log$time), outcome = I(log$outcome)
))

size <- sprintf("%s attempts of %d learners at %d skills", count(nrow(log)), learners, skill_count)
bench_task("bkt.fit", "Fit of the skills of a log by EM from five starts", size, "pyBKT 1.4.3 `Model.fit`")
bench_task("bkt.fitLessons", "Fit of the skills of a log by EM from five starts, a fifth of its attempts without an outcome",
           size, "pyBKT 1.4.3 `Model.fit`, correct = -1")
bench_task("bkt.fitPairs", "Fit of the skills of a log by EM from five starts, learn and forget rates by pair of items",
           size, "pyBKT 1.4.3 `Model.fit(multipair = True)`")
bench_task("bkt.replay", "Replay of the log through the fitted model, a prediction and an update for each attempt", size,
           "pyBKT 1.4.3 `Model.predict`")
