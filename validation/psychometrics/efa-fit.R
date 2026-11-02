# The extraction by maximum likelihood and the fit of exploratory factors (dimensionality-fit.json). References:
# psych's fa(fm = "ml") and fa(fm = "minres") on the responses, with its STATISTIC, dof, PVAL, RMSEA and its 90%
# interval, TLI, BIC and rms, and fm = "uls" and "pa" beside minres; and the base R solutions of factor-solutions.R, run to a projected gradient below 1e-10,
# with psych's fa.stats of them, which is the reference of the fit at a tight tolerance, as psych's L-BFGS-B stops short
# of the optimum. The persons are the rows of the data, psych's n.obs, and the residuals are weighted by the persons who
# answered each pair, psych's np.obs.
#
# Data: psych::bfi, its first 10 and all 25 items, with missing responses; a seeded simulation of 12 items of 5
# categories on two correlated factors; three bfi items, which one factor fits with no degrees of freedom; and a seeded
# factor of six items, one of loading 0.9995, whose uniqueness stays at its bound of 0.005, scored in hundredths.

source("validation/psychometrics/common.R")
source("validation/psychometrics/factor-solutions.R")
suppressPackageStartupMessages(library(psych))

SEED <- 20261034
# psych's fa computes the RMSEA interval at alpha = 0.1, a 90% interval.
RMSEA_ALPHA <- 0.1

rows_of <- function(X) unname(lapply(seq_len(nrow(X)), function(i) as.numeric(X[i, ])))

# The fields of measurekit's FactorFit from psych's statistics: without degrees of freedom the significance, the RMSEA,
# its interval and the TLI are undefined, and the BIC is the statistic.
fit_of <- function(stats, persons) {
  free <- stats$dof > 0
  list(
    chiSquare = stats$STATISTIC, degreesOfFreedom = stats$dof,
    significance = if (free) stats$PVAL else NA, rmsea = if (free) unname(stats$RMSEA[1]) else NA,
    rmseaInterval = if (free) unname(stats$RMSEA[2:3]) else NA, tli = if (free) stats$TLI else NA,
    bic = stats$STATISTIC - stats$dof * log(persons), srmsr = stats$rms
  )
}

base_of <- function(X, r, m, extraction, rotation) {
  unrotated <- if (extraction == "maximum-likelihood") {
    likelihood_axes(r, m, likelihood_uniquenesses(r, m))
  } else {
    principal_axes(r, m, minres_communalities(r, m))
  }
  oriented <- rotate_solution(unrotated, rotation)
  stats <- suppressWarnings(fa.stats(r, oriented$loadings, phi = oriented$phi, n.obs = nrow(X),
                                     np.obs = pairwiseCount(X), alpha = RMSEA_ALPHA))
  list(loadings = unname(oriented$loadings), phi = unname(oriented$phi),
       communalities = unname(rowSums(unrotated^2)), fit = fit_of(stats, nrow(X)))
}

psych_of <- function(X, m, extraction, rotation) {
  fm <- if (extraction == "maximum-likelihood") "ml" else "minres"
  fitted <- suppressWarnings(suppressMessages(fa(X, nfactors = m, fm = fm, rotate = rotation, warnings = FALSE)))
  phi <- if (is.null(fitted$Phi)) diag(m) else fitted$Phi
  oriented <- orient(unclass(fitted$loadings), phi)
  list(loadings = unname(oriented$loadings), phi = unname(oriented$phi), communalities = unname(fitted$communality),
       uniquenesses = unname(fitted$uniquenesses), fit = fit_of(fitted, nrow(X)))
}

fit_case <- function(name, X, m, extraction, rotation) {
  r <- cor(X, use = "pairwise")
  list(name = name, itemIds = colnames(X), responses = rows_of(X), factors = m, extraction = extraction,
       rotation = rotation, base = base_of(X, r, m, extraction, rotation), psych = psych_of(X, m, extraction, rotation))
}

data(bfi, package = "psych")
bfi_items <- as.matrix(bfi[, 1:25])

set.seed(SEED)
heywood <- local({
  n <- 500
  loadings <- c(0.9995, 0.8, 0.7, 0.6, 0.5, 0.45)
  trait <- rnorm(n)
  # Scores are integers: the normal variables in hundredths, from 1,000.
  X <- sapply(loadings, function(l) round(100 * (l * trait + sqrt(1 - l^2) * rnorm(n))) + 1000)
  colnames(X) <- paste0("h", 1:6)
  X
})

set.seed(SEED + 1)
graded <- local({
  n <- 1000
  traits <- matrix(rnorm(2 * n), n) %*% chol(matrix(c(1, 0.4, 0.4, 1), 2))
  loadings <- rep(c(0.75, 0.6, 0.5), 4)
  X <- sapply(1:12, function(j) {
    latent <- loadings[j] * traits[, if (j <= 6) 1 else 2] + sqrt(1 - loadings[j]^2) * rnorm(n)
    findInterval(latent, c(-1.2, -0.4, 0.4, 1.2))
  })
  colnames(X) <- sprintf("g%02d", 1:12)
  X
})

cases <- list(
  fit_case("bfi, ten items, two factors, maximum likelihood", bfi_items[, 1:10], 2, "maximum-likelihood", "oblimin"),
  fit_case("bfi, ten items, two factors, minres", bfi_items[, 1:10], 2, "minres", "oblimin"),
  fit_case("bfi, five factors, maximum likelihood", bfi_items, 5, "maximum-likelihood", "oblimin"),
  fit_case("graded items, two factors, maximum likelihood, varimax", graded, 2, "maximum-likelihood", "varimax"),
  fit_case("graded items, two factors, minres, unrotated", graded, 2, "minres", "none"),
  fit_case("three items, one factor, no degrees of freedom", bfi_items[, 1:3], 1, "maximum-likelihood", "oblimin"),
  fit_case("a uniqueness at its bound, maximum likelihood", heywood, 1, "maximum-likelihood", "oblimin"),
  fit_case("a uniqueness at its bound, minres", heywood, 1, "minres", "oblimin")
)
# Minres is unweighted least squares, and principal axis factoring iterates to the same solution: psych's "uls" and "pa"
# on the case of minres, which the guide's Differences cite.
other_extraction <- function(fm) {
  fitted <- suppressWarnings(suppressMessages(fa(bfi_items[, 1:10], nfactors = 2, fm = fm, rotate = "oblimin",
                                                 max.iter = 100000, min.err = 1e-12, warnings = FALSE)))
  unname(orient(unclass(fitted$loadings), fitted$Phi)$loadings)
}
cases[[2]]$psych$unweightedLeastSquares <- other_extraction("uls")
cases[[2]]$psych$principalAxis <- other_extraction("pa")

bound <- cases[[7]]$psych$uniquenesses[1]
stopifnot(abs(bound - 0.005) < 1e-5)

write_fixture("dimensionality-fit", list(
  source = paste("psych::fa(fm = \"ml\", \"minres\") and fa.stats of the base R solutions of factor-solutions.R, on",
                 "psych::bfi (10 and 25 items), 12 simulated items of 5 categories on two factors correlated 0.4 for",
                 "1,000 persons (set.seed(20261035)), three bfi items, and six simulated items of one factor, one with",
                 "loading 0.9995, for 500 persons (set.seed(20261034))."),
  cases = cases
))
