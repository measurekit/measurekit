# Item-level model comparison fixtures. Reference: GDINA's modelcomp(method = "Wald") with Wald.args = list(SE.type = 2),
# on G-DINA fits run to convergence: the Wald statistic, degrees of freedom and p-value of the DINA, DINO, A-CDM, LLM and
# R-RUM of each item with two or more attributes, the model it selects by the largest p-value above 0.05, and the
# covariance of each item's success probabilities that the statistics use (extract(catprob.cov, SE.type = 2)). Data:
# GDINA's sim10GDINA (1000 persons, 10 items, 3 attributes), and 20 seeded simulations of 600 persons on 3 attributes
# and 10 items, three of one attribute and seven of two or three, whose true model is the DINA, DINO, A-CDM, LLM or R-RUM.

source("validation/psychometrics/common.R")
suppressPackageStartupMessages(library(GDINA))

SEED <- 20261029
CONTROL <- list(conv.crit = 1e-9, maxitr = 20000)
MODELS <- c("DINA", "DINO", "ACDM", "LLM", "RRUM")
ALPHA <- 0.05
SETS <- 20
PERSONS <- 600

# The success probabilities of the latent groups of an item with `K` attributes, in GDINA's order of groups, when the
# item follows `model`: a guessing probability of 0.15 and a success probability of 0.85 for the DINA and DINO, and for
# the others an equal effect of each attribute from 0.1 to 0.9 on the identity scale, from -2 to 2 on the logit scale and
# a penalty of 0.4 for each attribute not mastered on the log scale.
success_probabilities <- function(model, K) {
  patterns <- attributepattern(K)
  mastered <- rowSums(patterns)
  switch(model,
    DINA = ifelse(mastered == K, 0.85, 0.15),
    DINO = ifelse(mastered == 0, 0.15, 0.85),
    ACDM = 0.1 + 0.8 * mastered / K,
    LLM = plogis(-2 + 4 * mastered / K),
    RRUM = 0.9 * 0.4^(K - mastered)
  )
}

# Persons' responses to the items of `Q`, whose item j follows `models[j]`, from a saturated distribution over the profiles.
simulate_responses <- function(Q, models, persons, weights) {
  K <- ncol(Q)
  patterns <- attributepattern(K)
  alpha <- patterns[sample.int(nrow(patterns), persons, replace = TRUE, prob = weights), , drop = FALSE]
  sapply(seq_len(nrow(Q)), function(j) {
    required <- which(Q[j, ] == 1)
    groups <- attributepattern(length(required))
    probabilities <- success_probabilities(models[j], length(required))
    group <- match(apply(alpha[, required, drop = FALSE], 1, paste, collapse = ""), apply(groups, 1, paste, collapse = ""))
    rbinom(persons, 1, probabilities[group])
  })
}

# The case for one set of responses: the G-DINA fit, its covariance by item, and what modelcomp reports.
comparison_case <- function(name, data, Q, truth = NULL) {
  Q <- psychometrics_order(Q)
  colnames(data) <- paste0("item", seq_len(ncol(data)))
  fit <- GDINA(data, Q, model = "GDINA", control = CONTROL, verbose = 0)
  stopifnot(extract(fit, "convergence"))
  compared <- which(rowSums(Q) > 1)
  result <- modelcomp(fit, method = "Wald", models = MODELS, Wald.args = list(SE.type = 2),
                      decision.args = list(rule = "largestp", alpha.level = ALPHA, adjusted = FALSE))
  # GDINA breaks a tie of the largest p-value at random, so a tie would have no reference.
  stopifnot(!any(apply(result$pvalues, 1, function(p) sum(p == max(p)) > 1 && max(p) > ALPHA)))
  cov <- extract(fit, "catprob.cov", SE.type = 2)
  covariance <- lapply(compared, function(j) {
    index <- cov$index$loc[cov$index$item == j]
    lapply(seq_along(index), function(a) cov$cov[index[a], index])
  })
  list(
    name = name,
    itemIds = colnames(data),
    responses = unname(as.matrix(data)),
    qmatrix = qmatrix(Q, colnames(data)),
    probabilities = unname(lapply(extract(fit, "catprob.parm"), unname)),
    compared = colnames(data)[compared],
    covariance = covariance,
    statistics = unname(lapply(seq_along(compared), function(i) unname(result$stats[i, ]))),
    degreesOfFreedom = unname(lapply(seq_along(compared), function(i) unname(result$df[i, ]))),
    pValues = unname(lapply(seq_along(compared), function(i) unname(result$pvalues[i, ]))),
    selected = tolower(result$selected.model$models[compared]),
    truth = if (is.null(truth)) NULL else tolower(truth[compared])
  )
}

data(sim10GDINA, package = "GDINA")
real_q <- as.matrix(sim10GDINA$simQ)
colnames(real_q) <- paste0("A", seq_len(ncol(real_q)))
real <- comparison_case("sim10GDINA", sim10GDINA$simdat, real_q)

# Simulations: three single-attribute items, then seven items of two or three attributes, one true model after another.
set.seed(SEED)
simulation_q <- rbind(diag(3), matrix(c(1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1), ncol = 3, byrow = TRUE))
colnames(simulation_q) <- c("A1", "A2", "A3")
weights <- runif(8, 0.5, 1)
sets <- lapply(seq_len(SETS), function(s) {
  models <- c("DINA", "DINA", "DINA", rep_len(MODELS[c((s - 1) %% 5 + 1, s %% 5 + 1, (s + 1) %% 5 + 1, (s + 2) %% 5 + 1,
                                                       (s + 3) %% 5 + 1, (s - 1) %% 5 + 1, s %% 5 + 1)], 7))
  data <- simulate_responses(simulation_q, models, PERSONS, weights)
  comparison_case(sprintf("simulation %d", s), data, simulation_q, models)
})

# The recovery: how often the Wald test rejects the true model of an item at 0.05, and how often it selects it.
outcomes <- do.call(rbind, lapply(sets, function(set) {
  data.frame(rejected = mapply(function(p, truth) p[match(toupper(truth), MODELS)] <= ALPHA, set$pValues, set$truth),
             selected = set$selected == set$truth)
}))
message(sprintf("True model rejected %.1f%%, selected %.1f%%, of %d items", 100 * mean(outcomes$rejected),
                100 * mean(outcomes$selected), nrow(outcomes)))

write_fixture("cdm-item-comparison", list(
  source = "GDINA's modelcomp(method = 'Wald', Wald.args = list(SE.type = 2)) on G-DINA fits of sim10GDINA and of 20 simulations, set.seed(20261029): 600 persons, 3 attributes, 10 items, each item following the DINA, DINO, A-CDM, LLM or R-RUM.",
  models = tolower(MODELS),
  level = ALPHA,
  cases = c(list(real), sets),
  recovery = list(items = nrow(outcomes), rejected = mean(outcomes$rejected), selected = mean(outcomes$selected))
))
