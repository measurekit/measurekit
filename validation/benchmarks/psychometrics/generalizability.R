# Generalizability of a fully crossed design of persons, items and raters: the variance components of the six sources by
# REML with lme4's lmer, the reference that the gtheory package, which wraps it, was removed from CRAN for. measurekit
# estimates the components from the mean squares of the analysis of variance, which equal REML's when none is negative.

suppressPackageStartupMessages(library(lme4))

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
