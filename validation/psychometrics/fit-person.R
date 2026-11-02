# Person fit (mokken-person-scalability.json, fit-corrected-person-fit.json): PerFit 1.4.7's Gnormed, U3, Ht and lzstar,
# with base R write-ups where PerFit is not the statistic. Expected values only, by case name: the responses and items
# are those of mokken.json, fit-residuals.json and cat-3pl.json, which the tests read; the thetas, which PerFit takes as
# `Ability`, are mirt's estimates and are written beside the expected values.
#
# PerFit is not the reference where:
# - Gnormed orders items of equal p-values by their columns; the write-up averages over their orders, as guttmanErrors
#   of measurekit and mokken's check.errors do, a half for each pair of tied items answered 0 and 1, and is checked
#   against PerFit where no p-values tie. Gnormed gives 0 to a person of an extreme score, where the write-up has none.
# - U3 with missing responses takes the sorted log-odds at the positions of the person's answered columns, so its value
#   changes with the order of the items: the script shows it on SAT12, and the write-up takes the person's answered
#   items. Both agree on the persons who answered every item.
# - Every index is rounded to 4 decimals: the write-ups, unrounded, are checked against PerFit rounded so.
# - lzstar with missing responses sums the variance and the correction over every item while
#   W_n skips the missing ones: the write-up sums each over the answered items, and is checked against PerFit, unrounded
#   to 4 decimals, on the persons who answered every item.

source("validation/psychometrics/common.R")
suppressPackageStartupMessages({
  library(mirt)
  library(PerFit)
})

TOLERANCE <- 1e-10
# PerFit's indices are rounded to 4 decimals; the write-ups rounded so equal them but for the rounding of a sum.
ROUNDED <- 1e-12
mokken_cases <- fromJSON("packages/psychometrics/test/fixtures/mokken.json", simplifyVector = TRUE)$cases
residual_cases <- fromJSON("packages/psychometrics/test/fixtures/fit-residuals.json", simplifyVector = FALSE)$cases
cat_items <- fromJSON("packages/psychometrics/test/fixtures/cat-3pl.json", simplifyVector = FALSE)$items

as_matrix <- function(rows) {
  do.call(rbind, lapply(rows, function(row) vapply(row, function(x) if (is.null(x)) NA_real_ else x, 0)))
}
extreme <- function(X) {
  answered <- rowSums(!is.na(X))
  correct <- rowSums(X, na.rm = TRUE)
  answered == 0 | correct == 0 | correct == answered
}

# --- G normed, U3 and Ht ------------------------------------------------------------------------------------------------

# Guttman errors over the answered pairs, a pair of tied p-values a half, over their most for the person's score.
normed_g <- function(X) {
  p <- colMeans(X, na.rm = TRUE)
  out <- vapply(seq_len(nrow(X)), function(n) {
    x <- X[n, ]
    passed <- which(x == 1)
    failed <- which(x == 0)
    errors <- sum(outer(p[passed], p[failed], "<")) + sum(outer(p[passed], p[failed], "==")) / 2
    errors / (length(passed) * length(failed))
  }, 0)
  out[extreme(X)] <- NA
  out
}

# U3 of van der Flier over the person's answered items, the log-odds of an item answered by all or none as 0.
u3 <- function(X) {
  p <- colMeans(X, na.rm = TRUE)
  odds <- log(p / (1 - p))
  odds[is.infinite(odds)] <- 0
  out <- vapply(seq_len(nrow(X)), function(n) {
    answered <- which(!is.na(X[n, ]))
    correct <- sum(X[n, answered])
    sorted <- sort(odds[answered], decreasing = TRUE)
    first <- sum(sorted[seq_len(correct)])
    last <- sum(rev(sorted)[seq_len(correct)])
    if (first == last) NA else (first - sum(X[n, answered] * odds[answered])) / (first - last)
  }, 0)
  out[extreme(X)] <- NA
  out
}

# Ht: the covariance of the person's responses with the other persons' total of each answered item, over its largest
# value for the persons' proportions correct, Σ_m min(p_n, p_m) - p_n p_m over the persons of no extreme score.
ht <- function(X) {
  proportions <- rowMeans(X, na.rm = TRUE)
  totals <- colSums(X, na.rm = TRUE)
  scored <- which(!extreme(X))
  out <- vapply(seq_len(nrow(X)), function(n) {
    answered <- which(!is.na(X[n, ]))
    x <- X[n, answered]
    rest <- totals[answered] - x
    numerator <- mean(x * rest) - mean(x) * mean(rest)
    others <- proportions[setdiff(scored, n)]
    denominator <- sum(pmin(others, proportions[n]) - others * proportions[n])
    numerator / denominator
  }, 0)
  out[extreme(X)] <- NA
  out
}

scalability_case <- function(name, X) {
  g <- normed_g(X)
  u <- u3(X)
  h <- ht(X)
  kept <- !extreme(X)
  p <- colMeans(X, na.rm = TRUE)
  perfit_g <- Gnormed(X)$PFscores$PFscores
  # PerFit rounds every index to 4 decimals.
  if (!any(duplicated(p))) stopifnot(max(abs(perfit_g[kept] - round(g[kept], 4))) < ROUNDED)
  stopifnot(all(perfit_g[!kept & rowSums(!is.na(X)) > 0] == 0))
  complete <- kept & rowSums(is.na(X)) == 0
  stopifnot(max(abs(U3(X)$PFscores$PFscores[complete] - round(u[complete], 4))) < ROUNDED)
  stopifnot(max(abs(Ht(X)$PFscores$PFscores[kept] - round(h[kept], 4))) < ROUNDED)
  list(name = name, ties = any(duplicated(p)), normedGuttmanErrors = g, u3 = u, ht = h)
}

transitive <- mokken_cases$responses[[which(mokken_cases$name == "transitive reasoning")]]
tied <- mokken_cases$responses[[which(mokken_cases$name == "tied item steps")]]
sat12 <- residual_cases[[which(vapply(residual_cases, `[[`, "", "name") == "SAT12 2PL")]]
sat12_responses <- as_matrix(sat12$responses)

# PerFit's U3 with missing responses depends on the order of the items; the write-up does not.
reversed <- rev(seq_len(ncol(sat12_responses)))
sparse <- which(rowSums(is.na(sat12_responses)) > 0 & !extreme(sat12_responses))
perfit_forward <- U3(sat12_responses)$PFscores$PFscores[sparse]
perfit_backward <- U3(sat12_responses[, reversed])$PFscores$PFscores[sparse]
stopifnot(max(abs(perfit_forward - perfit_backward)) > 0.01)
stopifnot(max(abs(u3(sat12_responses)[sparse] - u3(sat12_responses[, reversed])[sparse])) < TOLERANCE)

write_fixture("mokken-person-scalability", list(
  source = paste(
    "PerFit 1.4.7's Gnormed, U3 and Ht, NA.method = \"Pairwise\", checked against the base R write-ups, which are the",
    "values: G with tied p-values averaged over their orders, U3 over each person's answered items."
  ),
  cases = list(
    scalability_case("transitive reasoning", transitive),
    scalability_case("tied item steps", tied),
    scalability_case("SAT12 with missing responses", sat12_responses)
  )
))

# --- lz* -----------------------------------------------------------------------------------------------------------------

# Snijders' lz* of dichotomous logistic items, P = c + (d - c) / (1 + exp(-a (θ - b))), over each person's answered items.
lz_star <- function(X, items, theta, estimator, mean = 0, sd = 1) {
  out <- vapply(seq_len(nrow(X)), function(n) {
    answered <- which(!is.na(X[n, ]))
    x <- X[n, answered]
    a <- items$a[answered]
    b <- items$b[answered]
    c <- items$c[answered]
    d <- items$d[answered]
    t <- theta[n]
    logistic <- 1 / (1 + exp(-a * (t - b)))
    P <- c + (d - c) * logistic
    Q <- 1 - P
    first <- (d - c) * a * logistic * (1 - logistic)
    second <- first * a * (1 - 2 * logistic)
    w <- log(P / Q)
    r <- first / (P * Q)
    r0 <- switch(estimator,
      ML = 0,
      BM = (mean - t) / sd^2,
      WL = sum(first * second / (P * Q)) / (2 * sum(first^2 / (P * Q)))
    )
    cn <- sum(first * w) / sum(first * r)
    tilde <- w - cn * r
    (sum((x - P) * w) + cn * r0) / sqrt(sum(tilde^2 * P * Q))
  }, 0)
  out[extreme(X)] <- NA
  out
}

# mirt's ML, WLE or MAP thetas of the items; an infinite ML theta of an extreme score is written as ±6, the bound of
# measurekit's estimators, as the test passes finite thetas.
thetas_of <- function(X, items, method, mean = 0, sd = 1) {
  colnames(X) <- paste0("item", seq_len(ncol(X)))
  model <- set_items(as.data.frame(X), "3PL", data.frame(
    a1 = items$a, d = -items$a * items$b, g = items$c, u = items$d
  ))
  estimates <- fscores(model, method = method, mean = mean, cov = matrix(sd^2), max_theta = 30)[, "F1"]
  pmax(pmin(estimates, 6), -6)
}

lz_case <- function(name, X, items, estimator, mean = 0, sd = 1) {
  method <- switch(estimator, ML = "ML", WL = "WLE", BM = "MAP")
  theta <- thetas_of(X, items, method, mean, sd)
  values <- lz_star(X, items, theta, estimator, mean, sd)
  complete <- rowSums(is.na(X)) == 0 & !extreme(X)
  perfit <- lzstar(X, IP = cbind(items$a, items$b, items$c), Ability = theta, Ability.PModel = estimator,
                   mu = mean, sigma = sd)$PFscores$PFscores
  if (all(items$d == 1)) stopifnot(max(abs(perfit[complete] - round(values[complete], 4))) < ROUNDED)
  label <- switch(estimator, ML = "maximum-likelihood", WL = "weighted-likelihood", BM = "posterior-mode")
  list(name = name, estimator = label, prior = list(mean = mean, sd = sd), thetas = theta, standardized = values)
}

sat12_items <- list(
  a = vapply(sat12$items, function(item) item$parameters$a, 0),
  b = vapply(sat12$items, function(item) item$parameters$b, 0),
  c = vapply(sat12$items, function(item) item$parameters$c, 0),
  d = vapply(sat12$items, function(item) item$parameters$d, 0)
)
set.seed(20261103)
three_items <- list(
  a = vapply(cat_items[1:25], `[[`, 0, "a"), b = vapply(cat_items[1:25], `[[`, 0, "b"),
  c = vapply(cat_items[1:25], `[[`, 0, "c"), d = vapply(cat_items[1:25], `[[`, 0, "d")
)
truth <- rnorm(400)
three_pl <- t(vapply(truth, function(t) {
  p <- three_items$c + (three_items$d - three_items$c) / (1 + exp(-three_items$a * (t - three_items$b)))
  as.numeric(runif(25) < p)
}, numeric(25)))
three_pl[sample(length(three_pl), 200)] <- NA

write_fixture("fit-corrected-person-fit", list(
  source = paste(
    "PerFit 1.4.7's lzstar at mirt 1.48's ML, WLE and MAP thetas, on SAT12 of fit-residuals.json and on the first 25",
    "items of cat-3pl.json for 400 persons with 200 responses missing (set.seed(20261103)); the base R write-up over",
    "each person's answered items, unrounded, which equals PerFit to its 4 decimals on the persons who answered every item."
  ),
  threePl = list(responses = unname(lapply(seq_len(nrow(three_pl)), function(i) three_pl[i, ]))),
  cases = list(
    lz_case("SAT12, maximum likelihood", sat12_responses, sat12_items, "ML"),
    lz_case("SAT12, weighted likelihood", sat12_responses, sat12_items, "WL"),
    lz_case("SAT12, posterior mode", sat12_responses, sat12_items, "BM", mean = 0.5, sd = 1.5),
    lz_case("3PL, weighted likelihood", three_pl, three_items, "WL")
  )
))
