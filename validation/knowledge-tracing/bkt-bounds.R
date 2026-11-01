# Bounds on the rates of Bayesian knowledge tracing: the maximum of the forward likelihood within the bounds, found by
# optim's L-BFGS-B from a start inside them. pyBKT has no bounds, so base R is the reference. The log-likelihood is the
# probability of each learner's outcomes with their P(L) summed out by the forward algorithm.

source("validation/knowledge-tracing/common.R")

set.seed(20261102)
TRUTH <- c(prior = 0.3, learn = 0.2, forget = 0.04, guess = 0.35, slip = 0.12)
learners <- 300
attempts <- 10
events <- do.call(rbind, lapply(seq_len(learners), function(l) {
  known <- runif(1) < TRUTH[["prior"]]
  outcome <- integer(attempts)
  for (t in seq_len(attempts)) {
    outcome[t] <- as.integer(runif(1) < if (known) 1 - TRUTH[["slip"]] else TRUTH[["guess"]])
    known <- if (known) runif(1) >= TRUTH[["forget"]] else runif(1) < TRUTH[["learn"]]
  }
  data.frame(learner = sprintf("learner%03d", l), time = seq_len(attempts) * 1000 + l, outcome = outcome)
}))
events <- events[order(events$time), ]
outcomes <- split(events$outcome, events$learner)

# The log of the probability of a learner's outcomes: the forward algorithm, P(L) at the next attempt after each.
log_likelihood <- function(par, forgetting) {
  prior <- par[1]; learn <- par[2]; forget <- if (forgetting) par[3] else 0; guess <- par[4]; slip <- par[5]
  total <- 0
  for (y in outcomes) {
    p <- prior
    for (t in seq_along(y)) {
      known <- p * (if (y[t] == 1) 1 - slip else slip)
      unknown <- (1 - p) * (if (y[t] == 1) guess else 1 - guess)
      total <- total + log(known + unknown)
      p <- (unknown / (known + unknown)) * learn + (known / (known + unknown)) * (1 - forget)
    }
  }
  total
}

START <- c(prior = 0.5, learn = 0.05, forget = 0.01, guess = 0.15, slip = 0.05)
CASES <- list(
  list(name = "guess and slip capped", forgetting = FALSE, bounds = list(guess = c(0, 0.2), slip = c(0, 0.1))),
  list(name = "learn capped", forgetting = FALSE, bounds = list(learn = c(0, 0.1))),
  list(name = "forget capped", forgetting = TRUE, bounds = list(forget = c(0, 0.02))),
  list(name = "prior floored, guess capped, with forgetting", forgetting = TRUE,
       bounds = list(prior = c(0.4, 1), guess = c(0, 0.3)))
)
rates <- c("prior", "learn", "forget", "guess", "slip")
fits <- lapply(CASES, function(case) {
  lower <- rep(0, 5); upper <- rep(1, 5); names(lower) <- names(upper) <- rates
  for (rate in names(case$bounds)) { lower[rate] <- case$bounds[[rate]][1]; upper[rate] <- case$bounds[[rate]][2] }
  # measurekit keeps a guess and a slip 2^-52 off 0 and 1, which changes nothing a tolerance can see.
  lower[c("guess", "slip")] <- pmax(lower[c("guess", "slip")], 1e-9)
  upper[c("guess", "slip")] <- pmin(upper[c("guess", "slip")], 1 - 1e-9)
  free <- if (case$forgetting) rates else setdiff(rates, "forget")
  objective <- function(x) {
    par <- START; par[free] <- x
    -log_likelihood(par, case$forgetting)
  }
  fit <- optim(START[free], objective, method = "L-BFGS-B", lower = lower[free], upper = upper[free],
               control = list(maxit = 5000, factr = 100, pgtol = 1e-9))
  stopifnot(fit$convergence == 0)
  par <- START; par[free] <- fit$par
  if (!case$forgetting) par[["forget"]] <- 0
  list(name = case$name, forgetting = case$forgetting, bounds = case$bounds,
       parameters = as.list(par), logLikelihood = -fit$value)
})

write_fixture("bkt-bounds", list(
  source = "set.seed(20261102): 300 learners, 10 attempts at one skill, BKT with guess 0.35; L-BFGS-B on the forward likelihood.",
  tolerance = 1e-10,
  start = as.list(START),
  truth = as.list(TRUTH),
  events = lapply(seq_len(nrow(events)), function(i) list(
    learner = events$learner[i], item = "q", skills = I("s"), time = events$time[i], outcome = events$outcome[i])),
  fits = fits
))
