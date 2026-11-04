# Bayesian knowledge tracing: the fit of the skills of a log by EM, and the replay of the log through the fitted model,
# against pyBKT, which benchmarks.py times on the same logs. pyBKT fits every skill, five starts each by default, with
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

# A log in which each learner practices 20 of 400 skills twice, so that a skill has few learners and a learner few
# attempts at it, as in the logs of tutors with many knowledge components; each skill follows the rates of simulate_bkt.
set.seed(20261104)
sparse_learners <- 500
sparse_skill_count <- 400
skills_per_learner <- 20
sparse_attempts <- 2
chains <- do.call(rbind, lapply(seq_len(sparse_learners), function(learner) {
  data.frame(learner = learner, skill = sort(sample(sparse_skill_count, skills_per_learner)))
}))
rows <- nrow(chains) * sparse_attempts
outcome <- integer(rows)
for (chain in seq_len(nrow(chains))) {
  current <- runif(1) < 0.3
  for (a in seq_len(sparse_attempts)) {
    outcome[(chain - 1) * sparse_attempts + a] <- as.integer(runif(1) < ifelse(current, 0.9, 0.2))
    if (!current) current <- runif(1) < 0.15
  }
}
chain <- rep(seq_len(nrow(chains)), each = sparse_attempts)
skill <- paste0("s", chains$skill[chain])
sparse <- data.frame(learner = paste0("l", chains$learner[chain]), skill = skill,
                     item = paste0(skill, "q", (seq_len(rows) - 1) %% 4 + 1),
                     time = (seq_len(rows) - 1) %% (skills_per_learner * sparse_attempts) + 1, outcome = outcome)

bench_fixture("knowledge-tracing", "bkt-skills", list(
  source = paste(
    "A simulation, set.seed(20261104): 500 learners each practice 20 of 400 skills twice, by Bayesian knowledge tracing",
    "with a prior of 0.3, learning 0.15, slip 0.1 and guess 0.2."
  ),
  learner = I(sparse$learner), skill = I(sparse$skill), item = I(sparse$item), time = I(sparse$time),
  outcome = I(sparse$outcome)
))

size <- sprintf("%s attempts of %d learners at %d skills", count(nrow(log)), learners, skill_count)
bench_task("bkt.fit", "Fit of the skills of a log by EM from five starts", size, "pyBKT 1.4.3 `Model.fit`")
bench_task("bkt.fitSkills", "Fit of the skills of a log by EM from five starts, each learner at 20 of many skills",
           sprintf("%s attempts of %s learners at %d skills", count(nrow(sparse)), count(sparse_learners),
                   sparse_skill_count),
           "pyBKT 1.4.3 `Model.fit`",
           note = paste(
             "pyBKT stops a start when the log-likelihood changes by at most 0.005, or after 100 iterations, and",
             "measurekit at 1e-6 or 500, so measurekit runs more iterations: a median of 38 for a skill's best start,",
             "and 4 of the 400 skills reach 500"
           ))
bench_task("bkt.fitLessons", "Fit of the skills of a log by EM from five starts, a fifth of its attempts without an outcome",
           size, "pyBKT 1.4.3 `Model.fit`, correct = -1")
bench_task("bkt.fitPairs", "Fit of the skills of a log by EM from five starts, learn and forget rates by pair of items",
           size, "pyBKT 1.4.3 `Model.fit(multipair = True)`")
bench_task("bkt.replay", "Replay of the log through the fitted model, a prediction and an update for each attempt", size,
           "pyBKT 1.4.3 `Model.predict`")
