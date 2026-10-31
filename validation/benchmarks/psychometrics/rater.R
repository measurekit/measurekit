# Rater agreement and rater effects: Cohen's kappa by irr's kappa2, Fleiss' kappa by irr's kappam.fleiss, the intraclass
# correlations by psych's ICC, and the many-facet Rasch model by TAM's tam.mml.mfr with the rating scale shared by items.

suppressPackageStartupMessages({
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

bench_fixture("psychometrics", "rater", list(
  source = "Simulations, set.seed(20261029): ratings of two raters and of five, continuous ratings of four, and scores of persons by items by raters.",
  pair = unname(pair),
  panel = unname(panel),
  continuous = unname(continuous),
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
           function() kappa2(pair))
bench_task("rater.fleissKappa", "Fleiss' kappa of five raters", "500 subjects, 5 raters, 5 categories",
           reference("irr", "kappam.fleiss"), function() kappam.fleiss(panel))
bench_task("rater.intraclassCorrelation", "The six intraclass correlations of four raters", "200 subjects, 4 raters",
           reference("psych", "ICC"), function() suppressWarnings(ICC(continuous, lmer = FALSE)))
bench_task("rater.manyFacetRasch", "The many-facet Rasch model of items and raters, with standard errors",
           sprintf("%d persons, %d items, %d raters: %s scores", persons, item_count, rater_count, count(nrow(scores))),
           reference("TAM", "tam.mml.mfr(formulaA = ~ item + rater + step)"),
           function() invisible(capture.output(tam.mml.mfr(resp = responses, facets = facets, pid = wide$person,
                                                           formulaA = ~ item + rater + step, constraint = "items",
                                                           control = list(nodes = nodes, conv = 1e-4, convD = 1e-4),
                                                           verbose = FALSE))))
