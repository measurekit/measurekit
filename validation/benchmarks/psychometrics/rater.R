# Rater agreement and rater effects: Cohen's kappa by irr's kappa2, Fleiss' kappa by irr's kappam.fleiss, exact and adjacent
# agreement by irr's agree, the standardized mean difference of two raters by effsize's cohen.d, the intraclass correlations
# by psych's ICC, and the many-facet Rasch model by TAM's tam.mml.mfr with the rating scale shared by items.

suppressPackageStartupMessages({
  library(effsize)
  library(irr)
  library(psych)
  library(TAM)
})

set.seed(20261029)

# --- Data ----------------------------------------------------------------------------------------------------------

# Ratings of `n` subjects by `raters` raters on 5 categories that agree about the subject's level up to a rater's noise.
ratings <- function(n, raters) {
  level <- rnorm(n)
  sapply(seq_len(raters), function(r) as.integer(cut(level + rnorm(n, 0, 0.6), c(-Inf, -1, -0.3, 0.3, 1, Inf))) - 1L)
}

pair <- ratings(2000, 2)
panel <- ratings(500, 5)
continuous <- sapply(1:4, function(r) rnorm(200, 0, 1) + rnorm(200))
subjects <- rnorm(200)
continuous <- sapply(1:4, function(r) subjects + rnorm(200, 0, 0.7) + 0.1 * r)

persons <- 200
item_count <- 10
rater_count <- 4
ability <- rnorm(persons)
difficulty <- round(rnorm(item_count, 0, 0.7), 3)
severity <- round(rnorm(rater_count, 0, 0.4), 3)
scores <- expand.grid(person = seq_len(persons), item = seq_len(item_count), rater = seq_len(rater_count))
eta <- ability[scores$person] - difficulty[scores$item] - severity[scores$rater]
thresholds <- c(-1, 0, 1)
uniform <- runif(nrow(scores))
scores$score <- as.integer(uniform < plogis(eta - thresholds[1])) + as.integer(uniform < plogis(eta - thresholds[2])) +
  as.integer(uniform < plogis(eta - thresholds[3]))

smd_subjects <- 2000
smd_level <- rnorm(smd_subjects, 3, 1)
smd <- cbind(smd_level + rnorm(smd_subjects, 0, 0.5), smd_level + 0.4 + rnorm(smd_subjects, 0, 0.7))

bench_fixture("psychometrics", "rater", list(
  source = "Simulations, set.seed(20261029): ratings of two raters and of five, continuous ratings of four, and scores of persons by items by raters.",
  pair = unname(pair),
  panel = unname(panel),
  continuous = unname(continuous),
  smd = unname(smd),
  scores = lapply(seq_len(nrow(scores)), function(i) {
    list(levels = list(person = paste0("p", scores$person[i]), item = paste0("i", scores$item[i]),
                       rater = paste0("r", scores$rater[i])), score = scores$score[i])
  })
))

# --- References -----------------------------------------------------------------------------------------------------

wide <- reshape(scores, idvar = c("person", "rater"), timevar = "item", direction = "wide")
facets <- data.frame(rater = factor(wide$rater))
responses <- wide[, grep("^score\\.", names(wide))]
nodes <- seq(-6, 6, length.out = 61)

bench_task("rater.cohenKappa", "Cohen's kappa of two raters", "2,000 subjects, 5 categories", reference("irr", "kappa2"),
           function() irr::kappa2(pair))
bench_task("rater.fleissKappa", "Fleiss' kappa of five raters", "500 subjects, 5 raters, 5 categories",
           reference("irr", "kappam.fleiss"), function() irr::kappam.fleiss(panel))
bench_task("rater.exactAgreement", "The share of subjects on whom five raters agree exactly", "500 subjects, 5 raters, 5 categories",
           reference("irr", "agree(tolerance = 0)"), function() irr::agree(panel, 0))
bench_task("rater.adjacentAgreement", "The share of subjects whose ratings are within one category", "500 subjects, 5 raters, 5 categories",
           reference("irr", "agree(tolerance = 1)"), function() irr::agree(panel, 1))
bench_task("rater.meanDifference", "The standardized mean difference of two raters", sprintf("%s subjects, 2 raters", count(smd_subjects)),
           reference("effsize", "cohen.d"), function() effsize::cohen.d(smd[, 1], smd[, 2]))
bench_task("rater.intraclassCorrelation", "The six intraclass correlations of four raters", "200 subjects, 4 raters",
           reference("psych", "ICC"), function() suppressWarnings(psych::ICC(continuous, lmer = FALSE)))
bench_task("rater.manyFacetRasch", "The many-facet Rasch model of items and raters, with standard errors",
           sprintf("%d persons, %d items, %d raters: %s scores", persons, item_count, rater_count, count(nrow(scores))),
           reference("TAM", "tam.mml.mfr(formulaA = ~ item + rater + step)"),
           function() invisible(capture.output(tam.mml.mfr(resp = responses, facets = facets, pid = wide$person,
                                                           formulaA = ~ item + rater + step, constraint = "items",
                                                           control = list(nodes = nodes, conv = 1e-4, convD = 1e-4),
                                                           verbose = FALSE))),
           note = paste("TAM takes its standard errors from the diagonal of the second derivatives of its last M-step,",
                        "with no E-step more, and measurekit from the Hessian of the marginal log-likelihood by central",
                        "differences of its exact gradient, two E-steps for each parameter, which take about half its",
                        "time: its estimates alone take about 0.6 times as long as TAM's"))

# --- Interactions ---------------------------------------------------------------------------------------------------
# Every person scored by every rater on every item, with a bias of 0.8 logits in two cells of items by raters.

set.seed(20261029)
bias_persons <- 150
bias_items <- 8
bias_raters <- 3
bias_ability <- rnorm(bias_persons)
bias_difficulty <- round(rnorm(bias_items, 0, 0.7), 3)
bias_severity <- c(-0.3, 0.1, 0.2)
bias_scores <- expand.grid(person = seq_len(bias_persons), item = seq_len(bias_items), rater = seq_len(bias_raters))
planted <- matrix(0, bias_items, bias_raters)
planted[1, 1] <- 0.8
planted[1, 2] <- -0.8
eta <- bias_ability[bias_scores$person] - bias_difficulty[bias_scores$item] - bias_severity[bias_scores$rater] -
  planted[cbind(bias_scores$item, bias_scores$rater)]
uniform <- runif(nrow(bias_scores))
bias_scores$score <- as.integer(uniform < plogis(eta - thresholds[1])) + as.integer(uniform < plogis(eta - thresholds[2])) +
  as.integer(uniform < plogis(eta - thresholds[3]))

bench_fixture("psychometrics", "rater-interactions", list(
  source = "A simulation, set.seed(20261029): every person scored by every rater on every item, with a bias of 0.8 logits in two cells of items by raters.",
  scores = lapply(seq_len(nrow(bias_scores)), function(i) {
    list(levels = list(person = paste0("p", bias_scores$person[i]), item = paste0("i", bias_scores$item[i]),
                       rater = paste0("r", bias_scores$rater[i])), score = bias_scores$score[i])
  })
))

bias_wide <- reshape(bias_scores, idvar = c("person", "rater"), timevar = "item", direction = "wide")
bias_facets <- data.frame(rater = factor(bias_wide$rater))
bias_responses <- bias_wide[, grep("^score\\.", names(bias_wide))]
colnames(bias_responses) <- paste0("i", seq_len(bias_items))
bench_task("rater.manyFacetRaschInteractions", "The many-facet Rasch model of items and raters with their interaction, with standard errors",
           sprintf("%d persons, %d items, %d raters: %s scores", bias_persons, bias_items, bias_raters, count(nrow(bias_scores))),
           reference("TAM", "tam.mml.mfr(formulaA = ~ item + rater + item:rater + step)"),
           function() invisible(capture.output(tam.mml.mfr(resp = bias_responses, facets = bias_facets, pid = bias_wide$person,
                                                           formulaA = ~ item + rater + item:rater + step, constraint = "items",
                                                           control = list(nodes = nodes, conv = 1e-4, convD = 1e-4),
                                                           verbose = FALSE))))
