# The additive factors model of the log of the bkt benchmark: the fit of a learner proficiency, a difficulty for each skill
# and a learning rate for each skill by maximum likelihood, against base R's glm of the same design. The log is that of
# the bkt script.

source("validation/benchmarks/simulation.R")

set.seed(20261029)
log <- simulate_bkt(200, 5, 20)
log$opportunity <- ave(seq_len(nrow(log)), log$learner, log$skill, FUN = seq_along) - 1
log$learner <- factor(log$learner)
log$skill <- factor(log$skill)

bench_task("logistic.afm", "Fit of the additive factors model by maximum likelihood",
           sprintf("%s attempts of 200 learners at 5 skills", count(nrow(log))), "base R `glm`, binomial family",
           function() glm(outcome ~ 0 + learner + skill + skill:opportunity, family = binomial, data = log),
           note = "the proficiency of a learner is a fixed effect, as measurekit's default is none")
