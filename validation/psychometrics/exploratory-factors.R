# Exploratory factor analysis: the minimum residual (minres) solution with several factors, its varimax and oblimin
# rotations, the minimum average partial test and the smoothing of correlation matrices, against psych's fa, VSS and
# cor.smooth. The base R solution is an independent reference: L-BFGS-B on the communalities, polished by Newton steps with
# a Hessian by central differences of the gradient to a projected gradient below 1e-10, and GPArotation's oblimin and
# Varimax to a gradient of 1e-10. Data: psych::bfi (25 items with 6 categories and missing responses), mirt::SAT12 scored
# with its key, 8 as missing, a seeded simulation of 12 dichotomous items on two correlated factors, and the 24
# psychological tests of datasets::Harman74.cor.

source("validation/psychometrics/common.R")
suppressPackageStartupMessages({
  library(psych)
  library(GPArotation)
  library(mirt)
  library(mvtnorm)
})

SEED <- 20261029
CORRECTION <- 0.5
# psych's lower bound of a uniqueness, so that a communality is at most 0.995.
UPPER <- 1 - 0.005
# psych's mclapply children draw their own random numbers; one core makes the reference reproducible.
options(mc.cores = 1)

# --- The minimum residual solution in base R ---------------------------------------------------------------------------

# Half the squared eigenvalues that the fit of m factors to the matrix with the communalities h on its diagonal leaves
# out, a negative leading eigenvalue counting in full, and its gradient with respect to h, the diagonal of the residuals.
reduced_fit <- function(h, r, m) {
  s <- r
  diag(s) <- h
  e <- eigen(s, symmetric = TRUE)
  excess <- e$values
  excess[seq_len(m)] <- pmin(excess[seq_len(m)], 0)
  list(value = 0.5 * sum(excess^2), gradient = drop((e$vectors^2) %*% excess))
}

# The coordinates of h that the gradient does not push beyond a bound.
free_communalities <- function(h, g) !((h <= 1e-9 & g > 0) | (h >= UPPER - 1e-9 & g < 0))

minres_communalities <- function(r, m) {
  p <- nrow(r)
  gradient <- function(h) reduced_fit(h, r, m)$gradient
  start <- pmin(pmax(1 - 1 / diag(solve(r)), 0), UPPER)
  h <- optim(start, function(h) reduced_fit(h, r, m)$value, gradient, method = "L-BFGS-B", lower = rep(0, p),
             upper = rep(UPPER, p), control = list(factr = 1, pgtol = 0, maxit = 10000))$par
  for (step in 1:20) {
    g <- gradient(h)
    free <- free_communalities(h, g)
    jacobian <- sapply(which(free), function(j) {
      shift <- replace(numeric(p), j, 1e-6)
      (gradient(h + shift)[free] - gradient(h - shift)[free]) / 2e-6
    })
    h[free] <- h[free] - solve(matrix(jacobian, sum(free)), g[free])
    h <- pmin(pmax(h, 0), UPPER)
  }
  g <- gradient(h)
  stopifnot(max(abs(g[free_communalities(h, g)])) < 1e-10)
  h
}

# The loadings on the principal axes of the reduced matrix, each factor's summing to a number that is not negative.
principal_axes <- function(r, m, h) {
  s <- r
  diag(s) <- h
  e <- eigen(s, symmetric = TRUE)
  loadings <- e$vectors[, seq_len(m), drop = FALSE] %*% diag(sqrt(pmax(e$values[seq_len(m)], 0)), m)
  loadings %*% diag(ifelse(colSums(loadings) < 0, -1, 1), m)
}

# psych's orientation of a rotated solution: each factor's loadings sum to a positive number, and the factors are in
# the order of the variance they account for, the diagonal of Phi L'L.
orient <- function(loadings, phi) {
  m <- ncol(loadings)
  signs <- ifelse(colSums(loadings) < 0, -1, 1)
  loadings <- loadings %*% diag(signs, m)
  phi <- diag(signs, m) %*% phi %*% diag(signs, m)
  order <- order(diag(phi %*% crossprod(loadings)), decreasing = TRUE)
  list(loadings = loadings[, order, drop = FALSE], phi = phi[order, order, drop = FALSE])
}

base_solution <- function(r, m, rotation) {
  h <- minres_communalities(r, m)
  unrotated <- principal_axes(r, m, h)
  rotated <- if (m == 1 || rotation == "none") {
    list(loadings = unrotated, phi = diag(m))
  } else if (rotation == "varimax") {
    v <- Varimax(unrotated, normalize = TRUE, eps = 1e-10, maxit = 100000)
    stopifnot(v$convergence)
    list(loadings = v$loadings, phi = diag(m))
  } else {
    o <- oblimin(unrotated, eps = 1e-10, maxit = 100000)
    stopifnot(o$convergence)
    list(loadings = o$loadings, phi = o$Phi)
  }
  oriented <- orient(unclass(rotated$loadings), rotated$phi)
  list(communalities = unname(rowSums(unrotated^2)), unrotated = unname(unrotated),
       loadings = unname(oriented$loadings), phi = unname(oriented$phi))
}

# psych's fa of a correlation matrix: the loadings, Phi (the identity for an orthogonal rotation) and communalities.
psych_solution <- function(r, m, rotation) {
  fit <- suppressWarnings(suppressMessages(fa(r, nfactors = m, rotate = rotation, fm = "minres", warnings = FALSE)))
  list(loadings = unname(unclass(fit$loadings)), phi = unname(if (is.null(fit$Phi)) diag(m) else fit$Phi),
       communalities = unname(fit$communality))
}

# --- The minimum average partial test ----------------------------------------------------------------------------------

# Velicer's test as psych's VSS computes it: the mean squared partial correlation after the first i principal components,
# written out.
average_partials <- function(r, n) {
  p <- nrow(r)
  e <- eigen(r)
  components <- e$vectors %*% diag(sqrt(e$values))
  sapply(seq_len(n), function(i) {
    residual <- r - components[, seq_len(i), drop = FALSE] %*% t(components[, seq_len(i), drop = FALSE])
    scaled <- diag(1 / sqrt(diag(residual))) %*% residual %*% diag(1 / sqrt(diag(residual)))
    diag(scaled) <- 0
    sum(scaled^2) / (p * (p - 1))
  })
}

# psych's VSS of raw data, whose `map` is computed from the correlations it finds.
psych_map <- function(x, n, correlation) {
  invisible(capture.output(result <- suppressWarnings(suppressMessages(
    VSS(x, n = n, rotate = "none", fm = "minres", n.obs = nrow(x), plot = FALSE, cor = correlation)))))
  unname(result$map)
}

# --- Cases -------------------------------------------------------------------------------------------------------------

# One case: the matrix, the solution in base R and in psych, and the minimum average partial test where the data are given.
factor_case <- function(name, dataset, correlation, r, m, rotation, map = NULL) {
  list(
    name = name, dataset = dataset, correlation = correlation, factors = m, rotation = rotation, matrix = unname(r),
    base = base_solution(r, m, rotation), psych = psych_solution(r, m, rotation), map = map
  )
}

# Seeded data of 12 dichotomous items, 6 on each of 2 factors correlated 0.3, from normal variables with loadings 0.7 and
# 0.6 cut at thresholds from -1 to 1.
set.seed(SEED)
persons <- 1500
scores <- rmvnorm(persons, sigma = matrix(c(1, 0.3, 0.3, 1), 2))
true_loadings <- rep(c(0.7, 0.6), each = 6)
cuts <- rep(c(-1, -0.6, -0.2, 0.2, 0.6, 1), 2)
simulated <- sapply(1:12, function(j) {
  latent <- true_loadings[j] * scores[, if (j <= 6) 1 else 2] + sqrt(1 - true_loadings[j]^2) * rnorm(persons)
  as.numeric(latent > cuts[j])
})
colnames(simulated) <- sprintf("q%02d", 1:12)

data(bfi, package = "psych")
bfi_items <- as.matrix(bfi[, 1:25]) - 1
colnames(bfi_items) <- colnames(bfi)[1:25]
sat12 <- as.matrix(SAT12)
sat12[sat12 == 8] <- NA
sat12 <- key2binary(sat12, SAT12_KEY)
storage.mode(sat12) <- "double"

tetrachoric_of <- function(x) {
  suppressWarnings(suppressMessages(tetrachoric(x, correct = CORRECTION, smooth = FALSE)$rho))
}
polychoric_of <- function(x) {
  suppressWarnings(suppressMessages(polychoric(x, correct = CORRECTION, smooth = FALSE, progress = FALSE)$rho))
}
bfi_pearson <- cor(bfi_items, use = "pairwise.complete.obs")
simulated_rho <- tetrachoric_of(simulated)
sat12_raw <- tetrachoric_of(sat12)
# Not positive definite: the minimum eigenvalue is negative.
stopifnot(min(eigen(sat12_raw)$values) < 0)
sat12_smooth <- suppressWarnings(cor.smooth(sat12_raw))

# The correlations of one factor with loadings 0.99999 and from 0.85 to 0.45: the first item's communality is beyond the
# bound of 0.995, so that its uniqueness stays at its lower bound.
heywood_loadings <- c(0.99999, 0.85, 0.75, 0.65, 0.55, 0.45)
heywood <- outer(heywood_loadings, heywood_loadings)
diag(heywood) <- 1

cases <- list(
  factor_case("two factors, tetrachoric", "simulated", "polychoric", simulated_rho, 2, "oblimin",
              map = list(psych = psych_map(simulated, 6, "tet"), base = average_partials(simulated_rho, 6))),
  factor_case("two factors, tetrachoric, varimax", "simulated", "polychoric", simulated_rho, 2, "varimax"),
  factor_case("two factors, tetrachoric, unrotated", "simulated", "polychoric", simulated_rho, 2, "none"),
  factor_case("bfi, five factors, Pearson", "bfi", "pearson", bfi_pearson, 5, "oblimin",
              map = list(psych = psych_map(bfi_items, 8, "cor"), base = average_partials(bfi_pearson, 8))),
  factor_case("bfi, three factors, Pearson, varimax", "bfi", "pearson", bfi_pearson, 3, "varimax"),
  factor_case("bfi, one factor, Pearson", "bfi", "pearson", bfi_pearson, 1, "oblimin"),
  factor_case("bfi, five factors, polychoric", "bfi", "polychoric", polychoric_of(bfi_items), 5, "oblimin"),
  factor_case("SAT12, two factors, tetrachoric, smoothed", "sat12", "polychoric", sat12_smooth, 2, "oblimin"),
  factor_case("Harman's 24 tests, four factors", "harman74", "matrix", Harman74.cor$cov, 4, "oblimin"),
  factor_case("one factor with a communality beyond its bound", "heywood", "matrix", heywood, 1, "oblimin")
)

write_fixture("dimensionality-factors", list(
  source = paste("Minres, oblimin and varimax in base R and GPArotation, and psych::fa(fm = \"minres\"), on correlation",
                 "matrices of: 1,500 simulated persons answering 12 dichotomous items on 2 factors correlated 0.3 with",
                 "loadings 0.7 and 0.6 (set.seed(20261029)), psych::bfi, mirt::SAT12 scored with its key, and",
                 "datasets::Harman74.cor, and the exact one-factor matrix of six loadings, one of 0.99999. Tetrachoric and",
                 "polychoric correlations by psych with correct = 0.5."),
  truth = list(loadings = c(0.7, 0.6), correlation = 0.3),
  simulated = list(itemIds = colnames(simulated), responses = unname(simulated)),
  cases = cases,
  smoothing = list(
    list(name = "SAT12, tetrachoric", raw = unname(sat12_raw), smoothed = unname(sat12_smooth)),
    list(name = "three items", raw = matrix(c(1, 0.9, 0.9, 0.9, 1, -0.9, 0.9, -0.9, 1), 3),
         smoothed = suppressWarnings(cor.smooth(matrix(c(1, 0.9, 0.9, 0.9, 1, -0.9, 0.9, -0.9, 1), 3))))
  )
))
