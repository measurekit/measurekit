# Change fixtures. Gains are base R: the differences, the residuals of lm(post ~ pre) and Hake's normalized gain from its
# definition. The reliability of a difference is Lord's formula written out in base R. Reliable change from test scores
# is clinicalsignificance 3.0.0, cs_distribution(rci_method = "JT"), with a cross-check in base R; reliable change from
# ability estimates is the same index with the standard errors of the estimates, written out in base R.

source("validation/psychometrics/common.R")
suppressPackageStartupMessages(library(clinicalsignificance))

set.seed(20261029)

LEVELS <- c(0.8, 0.9, 0.95, 0.99)

# Two occasions of a test: persons with a true score, an error on each occasion and a mean gain, rounded to `digits`
# and kept within 0 to `maximum`.
occasions <- function(n, maximum, center, spread, gain, error, digits) {
  true_score <- rnorm(n, center, spread)
  bound <- function(x) round(pmin(maximum, pmax(0, x)), digits)
  list(
    maximum = maximum,
    pre = bound(true_score + rnorm(n, 0, error)),
    post = bound(true_score + gain + rnorm(n, 0, error * 1.2))
  )
}

raw <- occasions(60, 50, 32, 9, 6, 4, 0)
scale <- occasions(45, 800, 500, 100, 40, 40, 1)
stopifnot(any(raw$pre == raw$maximum), any(raw$post == raw$maximum))

gains <- function(x) {
  with(x, list(
    maximum = maximum,
    pre = pre,
    post = post,
    simple = post - pre,
    residual = unname(residuals(lm(post ~ pre))),
    normalized = list(
      group = (mean(post) - mean(pre)) / (maximum - mean(pre)),
      persons = ifelse(pre == maximum, NA, (post - pre) / (maximum - pre))
    )
  ))
}

# Lord (1956): the reliability of the difference of two scores with standard deviations s, reliabilities r and
# correlation rho.
difference_reliability <- function(sd_pre, reliability_pre, sd_post, reliability_post, correlation) {
  (sd_pre^2 * reliability_pre + sd_post^2 * reliability_post - 2 * correlation * sd_pre * sd_post) /
    (sd_pre^2 + sd_post^2 - 2 * correlation * sd_pre * sd_post)
}

difference_case <- function(sd_pre, reliability_pre, sd_post, reliability_post, correlation) {
  list(
    pre = list(sd = sd_pre, reliability = reliability_pre),
    post = list(sd = sd_post, reliability = reliability_post),
    correlation = correlation,
    reliability = difference_reliability(sd_pre, reliability_pre, sd_post, reliability_post, correlation)
  )
}

differences <- list(
  difference_case(sd(raw$pre), 0.86, sd(raw$post), 0.88, cor(raw$pre, raw$post)),
  difference_case(sd(scale$pre), 0.91, sd(scale$post), 0.9, cor(scale$pre, scale$post)),
  difference_case(10, 0.9, 12, 0.85, 0.5),
  difference_case(10, 0.9, 10, 0.9, 0),
  difference_case(10, 0.9, 12, 0.85, -0.3),
  difference_case(10, 0.9, 12, 0.85, 0.9),
  difference_case(15, 0.95, 15, 0.95, 0.7)
)
stopifnot(differences[[6]]$reliability < 0)

write_fixture("change-gain", list(
  cases = list(gains(raw), gains(scale)),
  differenceReliability = differences
))

# Jacobson and Truax (1991) from clinicalsignificance: the index and the classification of each person at each level.
reliable_scores <- function(x, reliability) {
  n <- length(x$pre)
  long <- data.frame(id = seq_len(n), time = rep(c(1, 2), each = n), outcome = c(x$pre, x$post))
  sd_pre <- sd(x$pre)
  base_index <- (x$post - x$pre) / sqrt(2 * (sd_pre * sqrt(1 - reliability))^2)
  levels <- lapply(LEVELS, function(level) {
    fit <- cs_distribution(long, id = id, time = time, outcome = outcome, better_is = "higher",
                           reliability = reliability, rci_method = "JT", significance_level = 1 - level)
    result <- fit$rci_results$data
    stopifnot(identical(as.integer(result$id), seq_len(n)), max(abs(result$rci - base_index)) < 1e-12)
    stopifnot(all(abs(abs(result$rci) - fit$critical_value) > 1e-6))
    list(
      level = level,
      criticalValue = fit$critical_value,
      classification = ifelse(result$improved, "improved", ifelse(result$deteriorated, "declined", "unchanged"))
    )
  })
  list(pre = x$pre, post = x$post, sd = sd_pre, reliability = reliability, index = base_index, levels = levels)
}

# Reliable change from ability estimates: (theta2 - theta1) / sqrt(se1^2 + se2^2).
n_theta <- 40
before <- list(theta = rnorm(n_theta), standardError = runif(n_theta, 0.2, 0.5))
after <- list(theta = before$theta + 0.3 + rnorm(n_theta, 0, 0.4), standardError = runif(n_theta, 0.2, 0.5))
theta_index <- (after$theta - before$theta) / sqrt(before$standardError^2 + after$standardError^2)
theta_levels <- lapply(LEVELS, function(level) {
  critical <- qnorm(1 - (1 - level) / 2)
  stopifnot(all(abs(abs(theta_index) - critical) > 1e-6))
  list(
    level = level,
    criticalValue = critical,
    classification = ifelse(theta_index > critical, "improved", ifelse(theta_index < -critical, "declined", "unchanged"))
  )
})

write_fixture("change-reliable", list(
  scores = list(reliable_scores(raw, 0.88), reliable_scores(scale, 0.91)),
  theta = list(before = before, after = after, index = theta_index, levels = theta_levels)
))
