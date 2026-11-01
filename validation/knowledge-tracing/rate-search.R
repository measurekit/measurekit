# The rates of logistic features estimated by maximum likelihood: a seeded log of four skills whose outcomes follow the
# decayed proportion of successes (decay 0.6) and the recency of the last attempt (exponent 0.4). The features are written
# out in base R for any decay and exponent, glm fits the intercepts and coefficients of each skill at a tight tolerance,
# and optim maximizes the log-likelihood over the two rates.

source("validation/knowledge-tracing/common.R")

set.seed(20261102)
SKILLS <- paste0("s", 1:4)
TRUE_RATES <- c(decay = 0.6, exponent = 0.4)
BOUNDS <- list(lower = c(0.05, 0.01), upper = c(1, 3), start = c(0.5, 0.5))
learner_count <- 60
per_learner <- 60

# The features of an event before it, from the learner's earlier attempts at the skill: the decayed proportion of successes
# with a prior of one success and one failure decayed over every attempt, and the recency of the last attempt.
features <- function(learner, skill, time, outcome, decay, exponent) {
  n <- length(learner)
  state <- new.env()
  proportion <- numeric(n)
  recency <- numeric(n)
  for (i in seq_len(n)) {
    key <- paste(learner[i], skill[i], sep = "\r")
    h <- state[[key]] %||% list(attempts = 0, successes = 0, failures = 0, last = NA)
    prior <- decay^h$attempts
    proportion[i] <- (prior + h$successes) / (prior + h$successes + prior + h$failures)
    elapsed <- if (is.na(h$last)) 0 else time[i] - h$last
    recency[i] <- if (elapsed == 0) 0 else elapsed^-exponent
    y <- outcome[i]
    state[[key]] <- list(attempts = h$attempts + 1, successes = decay * h$successes + y, failures = decay * h$failures + 1 - y,
                         last = time[i])
  }
  list(proportion = proportion, recency = recency)
}

log <- do.call(rbind, lapply(seq_len(learner_count), function(l) {
  data.frame(learner = sprintf("learner%02d", l), skill = sample(SKILLS, per_learner, replace = TRUE),
             time = cumsum(rexp(per_learner, 1 / 30)) + l)
}))
log <- log[order(log$time), ]
true_theta <- rnorm(learner_count, 0, 0.5)
true_beta <- setNames(rnorm(4, -0.2, 0.3), SKILLS)

# The outcomes are drawn event by event, each from the features its learner's earlier outcomes make.
log$outcome <- NA
state <- new.env()
for (i in seq_len(nrow(log))) {
  key <- paste(log$learner[i], log$skill[i], sep = "\r")
  h <- state[[key]] %||% list(attempts = 0, successes = 0, failures = 0, last = NA)
  prior <- TRUE_RATES[["decay"]]^h$attempts
  proportion <- (prior + h$successes) / (prior + h$successes + prior + h$failures)
  elapsed <- if (is.na(h$last)) 0 else log$time[i] - h$last
  recency <- if (elapsed == 0) 0 else elapsed^-TRUE_RATES[["exponent"]]
  theta <- true_theta[as.integer(sub("learner", "", log$learner[i]))]
  y <- rbinom(1, 1, plogis(theta + true_beta[[log$skill[i]]] + 2 * (proportion - 0.5) + 1.2 * recency))
  log$outcome[i] <- y
  state[[key]] <- list(attempts = h$attempts + 1, successes = TRUE_RATES[["decay"]] * h$successes + y,
                       failures = TRUE_RATES[["decay"]] * h$failures + 1 - y, last = log$time[i])
}
row.names(log) <- NULL

# The log-likelihood of the glm with an intercept and a slope of each feature for each skill, at the rates.
skill <- factor(log$skill, levels = SKILLS)
profile <- function(rates) {
  f <- features(log$learner, log$skill, log$time, log$outcome, rates[1], rates[2])
  design <- cbind(model.matrix(~ 0 + skill), model.matrix(~ 0 + skill) * f$proportion, model.matrix(~ 0 + skill) * f$recency)
  fit <- suppressWarnings(glm(log$outcome ~ 0 + design, family = binomial,
                              control = glm.control(epsilon = 1e-14, maxit = 100)))
  stopifnot(fit$converged)
  as.numeric(logLik(fit))
}
negative <- function(rates) -profile(rates)

# L-BFGS-B within the bounds with central differences, then Newton steps on the differenced gradient to a gradient near 0.
steps <- c(1e-5, 1e-5)
gradient <- function(rates) {
  sapply(1:2, function(j) {
    up <- rates
    down <- rates
    up[j] <- up[j] + steps[j]
    down[j] <- down[j] - steps[j]
    (negative(up) - negative(down)) / (2 * steps[j])
  })
}
found <- optim(BOUNDS$start, negative, gradient, method = "L-BFGS-B", lower = BOUNDS$lower, upper = BOUNDS$upper,
               control = list(factr = 1, pgtol = 0, maxit = 500))
rates <- found$par
for (iteration in 1:10) {
  g <- gradient(rates)
  if (max(abs(g)) < 1e-6) break
  rates <- rates - solve(optimHess(rates, negative, gradient), g)
}
stopifnot(max(abs(gradient(rates))) < 1e-5, all(rates > BOUNDS$lower), all(rates < BOUNDS$upper))

# A second search from another start, and a grid, reach the same maximum.
other <- optim(c(0.8, 0.8), negative, gradient, method = "L-BFGS-B", lower = BOUNDS$lower, upper = BOUNDS$upper,
               control = list(factr = 1, pgtol = 0, maxit = 500))
grid <- expand.grid(decay = seq(0.1, 0.95, by = 0.05), exponent = seq(0.1, 2, by = 0.1))
grid$loglik <- apply(grid, 1, function(r) profile(r))
stopifnot(abs(other$par[1] - rates[1]) < 1e-3, abs(other$par[2] - rates[2]) < 1e-3, max(grid$loglik) <= profile(rates) + 1e-9)

# No skill is separated at the maximum, so that glm's estimates are finite.
fit_at <- function(rates) {
  f <- features(log$learner, log$skill, log$time, log$outcome, rates[1], rates[2])
  design <- cbind(model.matrix(~ 0 + skill), model.matrix(~ 0 + skill) * f$proportion, model.matrix(~ 0 + skill) * f$recency)
  glm(log$outcome ~ 0 + design, family = binomial, control = glm.control(epsilon = 1e-14, maxit = 100))
}
stopifnot(!any(fitted(fit_at(rates)) < 1e-10 | fitted(fit_at(rates)) > 1 - 1e-10))

write_fixture("logistic-rates", list(
  source = "set.seed(20261102): 60 learners, 60 events each at 4 skills; decayedProportion and recency; glm and optim.",
  trueRates = unname(TRUE_RATES),
  bounds = lapply(BOUNDS, unname),
  events = events_of(log$learner, sprintf("q%d", seq_len(nrow(log))), as.list(log$skill), log$time, log$outcome),
  rates = rates,
  logLikelihood = profile(rates),
  gradient = gradient(rates),
  startLogLikelihood = profile(BOUNDS$start)
))
