# Numerics fixtures: the normal quantile and the logistic function from base R; the forward-backward quantities of
# hidden Markov sequences by enumerating every path of hidden states, an algorithm independent of measurekit's;
# and logistic regressions by glm.

source("validation/knowledge-tracing/common.R")

p <- c(1e-300, 1e-20, 1e-10, 1e-5, 0.001, 0.01, 0.025, 0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.975, 0.999,
       1 - 1e-10)
x <- c(-30, -10, -5, -2.5, -1, -0.3, 0, 0.3, 1, 2.5, 5, 10, 30)
write_fixture("math-functions", list(
  source = "Base R qnorm and plogis.",
  p = p,
  normalQuantile = qnorm(p),
  x = x,
  logistic = plogis(x)
))

# The joint probability of a path of hidden states with the observations up to step `through`: the emissions of the
# steps before `through`, and that of `through` itself only when `emit`.
path_weight <- function(path, through, emit, initial, transitions, emissions) {
  w <- initial[path[1]]
  for (t in seq_len(through)) {
    if (t > 1) w <- w * transitions[[t - 1]][path[t - 1], path[t]]
    if (t < through || emit) w <- w * emissions[t, path[t]]
  }
  w
}

# A sequence with its posterior quantities by enumeration. `transitions` is a list of T - 1 K x K matrices, `emissions`
# a T x K matrix; matrices are flattened by rows, as measurekit stores them.
enumerate <- function(initial, transitions, emissions) {
  steps <- nrow(emissions)
  k <- length(initial)
  paths_to <- function(t) as.matrix(expand.grid(rep(list(seq_len(k)), t)))
  paths <- paths_to(steps)
  weights <- apply(paths, 1, path_weight, steps, TRUE, initial, transitions, emissions)
  total <- sum(weights)
  posteriors <- sapply(seq_len(steps), function(t) sapply(seq_len(k), function(i) sum(weights[paths[, t] == i])))
  joint <- lapply(seq_len(steps - 1), function(t) {
    outer(seq_len(k), seq_len(k), Vectorize(function(i, j) sum(weights[paths[, t] == i & paths[, t + 1] == j])))
  })
  predictions <- sapply(seq_len(steps), function(t) {
    prefixes <- paths_to(t)
    w <- apply(prefixes, 1, path_weight, t, FALSE, initial, transitions, emissions)
    sapply(seq_len(k), function(i) sum(w[prefixes[, t] == i])) / sum(w)
  })
  by_rows <- function(matrices) I(as.numeric(unlist(lapply(matrices, function(m) as.vector(t(m))))))
  list(
    stateCount = k,
    initial = initial,
    transitions = by_rows(transitions),
    emissions = as.vector(t(emissions)),
    logLikelihood = log(total),
    posteriors = as.vector(posteriors / total),
    transitionPosteriors = by_rows(lapply(joint, function(m) m / total)),
    predictions = as.vector(predictions)
  )
}

row_stochastic <- function(k) {
  m <- matrix(runif(k * k, 0.05, 1), k, k)
  m / rowSums(m)
}

set.seed(20261029)
# A knowledge tracing sequence: states unlearned and learned, a learn rate and a forget rate per step, observations
# scored with a guess and a slip rate.
bkt_steps <- 7
learn <- runif(bkt_steps - 1, 0.05, 0.4)
forget <- runif(bkt_steps - 1, 0, 0.1)
bkt_transitions <- lapply(seq_len(bkt_steps - 1), function(t) matrix(c(1 - learn[t], forget[t], learn[t], 1 - forget[t]), 2))
observed <- c(0, 1, 0, 1, 1, 0, 1)
guess <- 0.2
slip <- 0.1
bkt_emissions <- cbind(ifelse(observed == 1, guess, 1 - guess), ifelse(observed == 1, 1 - slip, slip))
bkt <- enumerate(c(0.7, 0.3), bkt_transitions, bkt_emissions)

three_transitions <- lapply(1:5, function(t) row_stochastic(3))
three_emissions <- matrix(runif(18, 0.05, 1), 6, 3)
three_initial <- runif(3)
three_initial <- three_initial / sum(three_initial)
three <- enumerate(three_initial, three_transitions, three_emissions)

single <- enumerate(c(0.4, 0.6), list(), matrix(c(0.3, 0.9), 1, 2))

write_fixture("math-forward-backward", list(
  source = "Enumeration of every path of hidden states: P(observations, path) summed and conditioned in base R.",
  sequences = list(bkt, three, single)
))

# Logistic regressions: indicators of four skills, the number of earlier attempts, and two skill-specific slopes, as in
# the additive factors model; the second with outcomes in [0, 1].
n <- 800
skill <- sample(1:4, n, replace = TRUE)
attempts <- sample(0:9, n, replace = TRUE)
design <- cbind(
  intercept = 1,
  skill2 = as.numeric(skill == 2),
  skill3 = as.numeric(skill == 3),
  skill4 = as.numeric(skill == 4),
  attempts = attempts,
  attempts2 = attempts * (skill == 2),
  attempts3 = attempts * (skill == 3)
)
beta <- c(-0.4, 0.5, -0.3, 0.8, 0.15, 0.1, -0.05)
binary <- rbinom(n, 1, plogis(design %*% beta))
partial <- round(plogis(design %*% beta + rnorm(n, 0, 0.5)), 2)

regression <- function(y) {
  fit <- suppressWarnings(glm(y ~ design - 1, family = binomial, control = glm.control(epsilon = 1e-14, maxit = 100)))
  stopifnot(fit$converged)
  list(
    design = unname(design),
    outcomes = as.vector(y),
    coefficients = unname(coef(fit)),
    # The Bernoulli log-likelihood, which logLik reports only for 0/1 outcomes.
    logLikelihood = sum(y * log(fitted(fit)) + (1 - y) * log(1 - fitted(fit)))
  )
}
binary_fit <- regression(binary)
stopifnot(isTRUE(all.equal(binary_fit$logLikelihood, as.numeric(logLik(glm(binary ~ design - 1, family = binomial))),
                           tolerance = 1e-8)))
write_fixture("math-logistic-regression", list(
  source = "glm(family = binomial), epsilon = 1e-14; the second regression has outcomes in [0, 1].",
  regressions = list(binary_fit, regression(partial))
))
