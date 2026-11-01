# Generalizability of a fully crossed design of persons, items and raters: the variance components of the six sources by
# REML with lme4's lmer, the reference that the gtheory package, which wraps it, was removed from CRAN for. measurekit
# estimates the components from the mean squares of the analysis of variance, which equal REML's when none is negative.
# The same design with a quarter of its scores missing: Henderson's Method 1, written out in base R, as GeneralizIT and
# urGENOVA compute it. GeneralizIT is a Python package that benchmarks.py does not time.

suppressPackageStartupMessages(library(lme4))
source("validation/psychometrics/generalizability-simulation.R")
source("validation/psychometrics/generalizability-method-one.R")

set.seed(20261029)
persons <- 200
item_count <- 10
rater_count <- 4

scores <- expand.grid(person = seq_len(persons), item = seq_len(item_count), rater = seq_len(rater_count))
effect <- function(factor, spread) rnorm(max(factor), 0, spread)[factor]
p <- rnorm(persons, 0, 1)[scores$person]
scores$score <- 5 + p + effect(scores$item, 0.5) + effect(scores$rater, 0.4) +
  rnorm(persons * item_count, 0, 0.6)[(scores$person - 1) * item_count + scores$item] +
  rnorm(persons * rater_count, 0, 0.3)[(scores$person - 1) * rater_count + scores$rater] +
  rnorm(nrow(scores), 0, 0.8)

bench_fixture("psychometrics", "generalizability", list(
  source = "A simulation, set.seed(20261029): persons x items x raters with the six variance components.",
  scores = lapply(seq_len(nrow(scores)), function(i) {
    list(levels = list(person = paste0("p", scores$person[i]), item = paste0("i", scores$item[i]),
                       rater = paste0("r", scores$rater[i])), score = scores$score[i])
  })
))

for (name in c("person", "item", "rater")) scores[[name]] <- factor(scores[[name]])
bench_task("generalizability.generalizabilityStudy", "Variance components of a fully crossed design of three facets",
           sprintf("%d persons, %d items, %d raters: %s scores", persons, item_count, rater_count, count(nrow(scores))),
           reference("lme4", "lmer, six random effects, REML"),
           function() lmer(score ~ 1 + (1 | person) + (1 | item) + (1 | rater) + (1 | person:item) + (1 | person:rater) +
                             (1 | item:rater), data = scores, REML = TRUE))

# --- Unbalanced ------------------------------------------------------------------------------------------------------

set.seed(20261029)
grid <- grid_of(rater = paste0("r", seq_len(rater_count)), item = paste0("i", seq_len(item_count)),
                person = paste0("p", seq_len(persons)))
unbalanced <- cbind(grid, score = simulate_scores(grid, list(
  list(columns = "person", sd = 1), list(columns = "rater", sd = 0.4), list(columns = "item", sd = 0.5),
  list(columns = c("person", "item"), sd = 0.6), list(columns = c("person", "rater"), sd = 0.3)), error_sd = 0.8))
unbalanced <- unbalanced[runif(nrow(unbalanced)) < 0.75, ]

bench_fixture("psychometrics", "generalizability-unbalanced", list(
  source = "A simulation, set.seed(20261029): persons x items x raters with a quarter of the scores missing at random.",
  scores = lapply(seq_len(nrow(unbalanced)), function(i) {
    list(levels = list(person = as.character(unbalanced$person[i]), item = as.character(unbalanced$item[i]),
                       rater = as.character(unbalanced$rater[i])), score = unbalanced$score[i])
  })
))

bench_task("generalizability.generalizabilityStudyUnbalanced",
           "Variance components of a design of three facets with a quarter of the scores missing, by Henderson's Method 1",
           sprintf("%d persons, %d items, %d raters: %s scores", persons, item_count, rater_count, count(nrow(unbalanced))),
           reference("base", "method_one, Henderson's Method 1"),
           function() method_one(unbalanced, c("person", "item", "rater"), list()))
