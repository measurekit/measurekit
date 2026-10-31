# Dimensionality fixtures: the bivariate normal distribution function, polychoric and tetrachoric correlations, and
# parallel analysis. References: mvtnorm::pmvnorm (Miwa's algorithm) for the distribution function; the likelihood of a
# table written out in base R with pmvnorm cells, maximized by optimize and polished by the root of its score, for the
# correlations, with psych::polychoric and tetrachoric (smooth = FALSE) and polycor::polychor as cross-checks; base R
# eigen and the minimum residual loadings, and psych::fa.parallel, for the eigenvalues. Data: psych::bfi (25 items with
# 6 categories and missing responses), mirt::LSAT7 and mirt::SAT12 scored with its key, 8 as missing.

source("validation/psychometrics/common.R")
suppressPackageStartupMessages({
  library(psych)
  library(polycor)
  library(mvtnorm)
  library(mirt)
})

SEED <- 20261029
CORRECTION <- 0.5
# psych's mclapply children draw their own random numbers; one core makes the reference reproducible.
options(mc.cores = 1)

# --- The bivariate normal distribution function ------------------------------------------------------------------

set.seed(SEED)
cases <- data.frame(h = runif(400, -4, 4), k = runif(400, -4, 4), rho = runif(400, -0.999, 0.999))
# Both sides of every switch of the algorithm (|rho| at 0.3, 0.75 and 0.925), the largest correlations, the tails.
edges <- expand.grid(rho = c(0.2999, 0.3, 0.7499, 0.75, 0.9249, 0.925, 0.99999, -0.2999, -0.3, -0.75, -0.925, -0.99999,
                             1e-9, -1e-9),
                     h = c(-7.5, -1, 0.5), k = c(-7.5, 0, 6))
cases <- rbind(cases, edges[, c("h", "k", "rho")])
cases$p <- mapply(function(h, k, rho) {
  pmvnorm(upper = c(h, k), corr = matrix(c(1, rho, rho, 1), 2), algorithm = Miwa(), abseps = 1e-16)[1]
}, cases$h, cases$k, cases$rho)
write_fixture("dimensionality-bivariate-normal", list(
  source = "mvtnorm::pmvnorm(algorithm = Miwa(), abseps = 1e-16): 400 random points and the edges of the algorithm",
  cases = lapply(seq_len(nrow(cases)), function(i) as.list(cases[i, ]))
))

# --- Polychoric and tetrachoric correlations ---------------------------------------------------------------------

# The thresholds of an item over all its responses: qnorm of the cumulative proportions of its observed categories but
# the last.
thresholds_of <- function(x) {
  counts <- table(x[!is.na(x)])
  qnorm(cumsum(counts) / sum(counts))[-length(counts)]
}

# P(cell) of a table with thresholds a (rows) and b (columns) at rho, by pmvnorm on each rectangle.
cell_probabilities <- function(rho, a, b) {
  rows <- c(-Inf, a, Inf)
  columns <- c(-Inf, b, Inf)
  sigma <- matrix(c(1, rho, rho, 1), 2)
  outer(seq_len(length(rows) - 1), seq_len(length(columns) - 1), Vectorize(function(k, l) {
    pmvnorm(lower = c(rows[k], columns[l]), upper = c(rows[k + 1], columns[l + 1]), corr = sigma,
            algorithm = Miwa(), abseps = 1e-16)[1]
  }))
}

# The score of the log-likelihood: the derivative of Phi2 with respect to rho is the density phi2, at each corner of
# a cell.
score <- function(rho, tab, a, b) {
  rows <- c(-Inf, a, Inf)
  columns <- c(-Inf, b, Inf)
  sigma <- matrix(c(1, rho, rho, 1), 2)
  corner <- function(x, y) if (is.infinite(x) || is.infinite(y)) 0 else dmvnorm(c(x, y), sigma = sigma)
  probabilities <- cell_probabilities(rho, a, b)
  total <- 0
  for (k in seq_len(nrow(tab))) {
    for (l in seq_len(ncol(tab))) {
      if (tab[k, l] == 0) next
      slope <- corner(rows[k + 1], columns[l + 1]) - corner(rows[k], columns[l + 1]) -
        corner(rows[k + 1], columns[l]) + corner(rows[k], columns[l])
      total <- total + tab[k, l] * slope / probabilities[k, l]
    }
  }
  total
}

# The maximum likelihood rho of a table: optimize, then the root of the score near it.
ml_rho <- function(tab, a, b) {
  log_likelihood <- function(rho) {
    p <- cell_probabilities(rho, a, b)
    sum(tab[tab > 0] * log(p[tab > 0]))
  }
  rho <- optimize(log_likelihood, c(-0.99, 0.99), maximum = TRUE, tol = 1e-12)$maximum
  root <- try(uniroot(score, c(rho - 1e-4, rho + 1e-4), tab = tab, a = a, b = b, tol = 1e-15), silent = TRUE)
  if (inherits(root, "try-error")) stop("The score has no root near the optimum.")
  stopifnot(abs(root$root - rho) < 1e-6)
  root$root
}

# The package's counts of a pair of items: the persons who answered both, the empty cells of two dichotomous items given
# `correction`, as psych's tetrachoric.
pair_table <- function(x, y, correction) {
  keep <- !is.na(x) & !is.na(y)
  categories <- function(v) sort(unique(v[!is.na(v)]))
  tab <- table(factor(x[keep], levels = categories(x)), factor(y[keep], levels = categories(y)))
  tab <- matrix(as.numeric(tab), nrow(tab))
  if (all(dim(tab) == 2) && any(tab == 0)) tab[tab == 0] <- correction
  tab
}

# Base R correlations of the items `use` of x, NA elsewhere.
base_correlations <- function(x, use, correction) {
  tau <- lapply(seq_len(ncol(x)), function(i) thresholds_of(x[, i]))
  result <- matrix(NA_real_, ncol(x), ncol(x))
  for (i in use) {
    for (j in use) {
      if (i >= j) next
      tab <- pair_table(x[, i], x[, j], correction)
      result[i, j] <- result[j, i] <- ml_rho(tab, tau[[i]], tau[[j]])
    }
  }
  diag(result)[use] <- 1
  list(rho = result, tau = tau)
}

# One case: the data, the base R correlations of the items `use`, psych's correlations of all items and, for complete
# data, polycor's of the items `use`.
correlation_case <- function(name, x, use, dichotomous, correction = CORRECTION, polycor_pairs = TRUE) {
  base <- base_correlations(x, use, correction)
  reference <- if (dichotomous) {
    suppressWarnings(suppressMessages(tetrachoric(x, correct = correction, smooth = FALSE)$rho))
  } else {
    suppressWarnings(suppressMessages(polychoric(x, correct = correction, smooth = FALSE, progress = FALSE)$rho))
  }
  complete <- complete.cases(x)
  polycor_rho <- if (polycor_pairs) {
    cc <- x[complete, , drop = FALSE]
    m <- matrix(NA_real_, ncol(x), ncol(x))
    for (i in use) for (j in use) if (i < j) m[i, j] <- m[j, i] <- polychor(cc[, i], cc[, j])
    diag(m)[use] <- 1
    m
  }
  list(
    name = name,
    correction = correction,
    itemIds = colnames(x),
    responses = unname(x),
    used = use - 1,
    thresholds = lapply(base$tau, function(t) I(unname(as.numeric(t)))),
    baseR = unname(base$rho),
    psych = unname(reference),
    polycor = if (!is.null(polycor_rho)) unname(polycor_rho),
    completeCases = which(complete) - 1
  )
}

data(bfi, package = "psych")
bfi_items <- as.matrix(bfi[, 1:25]) - 1
colnames(bfi_items) <- colnames(bfi)[1:25]
lsat7 <- as.matrix(expand.table(LSAT7))
sat12 <- as.matrix(SAT12)
sat12[sat12 == 8] <- NA
sat12 <- key2binary(sat12, SAT12_KEY)
storage.mode(lsat7) <- "double"
storage.mode(sat12) <- "double"

# An item pair whose table has an empty cell, to exercise the correction: the first pair of the items that has one.
empty_cell_pair <- function(x) {
  for (i in 1:(ncol(x) - 1)) for (j in (i + 1):ncol(x)) {
    tab <- pair_table(x[, i], x[, j], 0)
    if (all(dim(tab) == 2) && any(tab == 0)) return(c(i, j))
  }
  NULL
}
rare <- empty_cell_pair(sat12)
message("SAT12 pair with an empty cell: ", paste(rare, collapse = ", "))

# polycor's two-step estimate uses each pair's own complete cases for the thresholds, so it is compared on data without
# missing responses: bfi's complete cases.
bfi_complete <- bfi_items[complete.cases(bfi_items), ]
cases_polychoric <- list(
  correlation_case("LSAT7", lsat7, 1:5, dichotomous = TRUE),
  correlation_case("LSAT7 without correction", lsat7, 1:5, dichotomous = TRUE, correction = 0),
  correlation_case("SAT12", sat12, 1:32, dichotomous = TRUE, polycor_pairs = FALSE),
  correlation_case("bfi", bfi_items, 1:10, dichotomous = FALSE, polycor_pairs = FALSE),
  correlation_case("bfi complete cases", bfi_complete[, 1:25], 1:10, dichotomous = FALSE)
)
write_fixture("dimensionality-polychoric", list(
  source = paste("psych::bfi (2800 persons, 25 items recoded 0-5, 508 missing responses), mirt::LSAT7 expanded (1000",
                 "persons, 5 items), mirt::SAT12 scored with its key, 8 as missing (600 persons, 32 items). Base R",
                 "maximum likelihood with pmvnorm cells on the items listed in `used`, psych on all items."),
  cases = cases_polychoric
))

# --- Eigenvalues and parallel analysis ---------------------------------------------------------------------------

# psych's fa keeps every uniqueness at least 0.005, so a loading is at most sqrt(0.995); random data, which one factor
# does not fit, otherwise run off to a Heywood case.
MAX_LOADING <- sqrt(1 - 0.005)

# The one-factor minres loadings of a correlation matrix, which minimize the squared off-diagonal residuals with
# |loading| at most MAX_LOADING: L-BFGS-B from the centroid loadings, then Newton steps on the loadings not at a bound,
# to a projected gradient below 1e-10.
minres_loadings <- function(r) {
  p <- nrow(r)
  objective <- function(l) {
    residual <- r - outer(l, l)
    sum(residual[lower.tri(residual)]^2)
  }
  gradient <- function(l) {
    residual <- r - outer(l, l)
    diag(residual) <- 0
    -2 * drop(residual %*% l)
  }
  hessian <- function(l) {
    h <- -2 * r + 4 * outer(l, l)
    diag(h) <- 2 * (sum(l^2) - l^2)
    h
  }
  projected_gradient <- function(l) max(abs(pmax(pmin(l - gradient(l), MAX_LOADING), -MAX_LOADING) - l))
  l <- optim(rowSums(r) / sqrt(sum(r)), objective, gradient, method = "L-BFGS-B", lower = rep(-MAX_LOADING, p),
             upper = rep(MAX_LOADING, p), control = list(factr = 1, pgtol = 0, maxit = 5000))$par
  for (step in 1:10) {
    g <- gradient(l)
    free <- !(abs(l) >= MAX_LOADING - 1e-9 & g * sign(l) < 0)
    l[free] <- l[free] - solve(hessian(l)[free, free, drop = FALSE], g[free])
    l <- pmax(pmin(l, MAX_LOADING), -MAX_LOADING)
  }
  if (projected_gradient(l) >= 1e-10) stop("minres: projected gradient ", projected_gradient(l))
  l * sign(sum(l))
}

# The eigenvalues of a correlation matrix, and of the matrix with the squared minres loadings on its diagonal.
eigenvalues <- function(r) {
  l <- minres_loadings(r)
  reduced <- r
  diag(reduced) <- l^2
  list(components = eigen(r, symmetric = TRUE, only.values = TRUE)$values,
       factors = eigen(reduced, symmetric = TRUE, only.values = TRUE)$values)
}

# psych's own eigenvalues of the one-factor minres solution of a correlation matrix.
psych_factor_values <- function(r) {
  suppressWarnings(suppressMessages(fa(r, nfactors = 1, rotate = "none", fm = "minres", warnings = FALSE)))$values
}

summarize <- function(values) {
  list(mean = unname(colMeans(values)), sd = unname(apply(values, 2, sd)),
       quantile = unname(apply(values, 2, quantile, probs = 0.95)))
}

# The number of leading `observed` eigenvalues above `quantile`, as psych counts them.
leading_above <- function(observed, quantile) {
  exceeded <- which(!(observed > quantile))[1]
  if (is.na(exceeded)) length(observed) else exceeded - 1
}

# psych::fa.parallel on the data with `iterations` random data sets: the observed eigenvalues, the mean, standard
# deviation and 0.95 quantile of each eigenvalue of the resampled data (psych's `simr`: each item resampled with
# replacement) and of the normal data (`sim`: psych's own suggested counts use these), and the numbers of leading
# eigenvalues above each quantile. `values` holds one row per data set: the components then the factors of the resampled
# data, then of the normal data.
psych_parallel <- function(x, correlation, iterations, seed) {
  set.seed(seed)
  invisible(capture.output(result <- suppressWarnings(suppressMessages(
    fa.parallel(x, n.iter = iterations, fa = "both", cor = correlation, plot = FALSE, quant = 0.95)
  ))))
  p <- ncol(x)
  block <- function(k) summarize(result$values[, (k * p + 1):((k + 1) * p), drop = FALSE])
  resampled_components <- block(0)
  resampled_factors <- block(1)
  normal_components <- block(2)
  normal_factors <- block(3)
  list(iterations = iterations, components = result$pc.values, factors = result$fa.values,
       resampledComponents = resampled_components, resampledFactors = resampled_factors,
       normalComponents = normal_components, normalFactors = normal_factors,
       suggested = list(
         resampledComponents = leading_above(result$pc.values, resampled_components$quantile),
         resampledFactors = leading_above(result$fa.values, resampled_factors$quantile),
         normalComponents = leading_above(result$pc.values, normal_components$quantile),
         normalFactors = leading_above(result$fa.values, normal_factors$quantile)
       ),
       psychSuggested = list(components = result$ncomp, factors = result$nfact))
}

# The reference for the random data of the package: `iterations` data sets whose items are permuted among the persons,
# their Pearson correlations over the pairs with both responses, and the statistics of each eigenvalue.
permutation_reference <- function(x, iterations, seed) {
  set.seed(seed)
  draws <- lapply(seq_len(iterations), function(i) {
    eigenvalues(cor(apply(x, 2, sample), use = "pairwise.complete.obs"))
  })
  list(iterations = iterations,
       components = summarize(t(sapply(draws, function(d) d$components))),
       factors = summarize(t(sapply(draws, function(d) d$factors))))
}

parallel_case <- function(name, x, r, correlation, psych_correlation, psych_iterations, permutation_iterations = NULL) {
  observed <- eigenvalues(r)
  list(
    name = name,
    correlation = correlation,
    itemIds = colnames(x),
    responses = unname(x),
    observed = list(components = observed$components, factors = observed$factors,
                    psychFactors = psych_factor_values(r)),
    psych = psych_parallel(x, psych_correlation, psych_iterations, SEED),
    permutation = if (!is.null(permutation_iterations)) permutation_reference(x, permutation_iterations, SEED)
  )
}

# Correlation matrices of permuted bfi data and their eigenvalues, to test the eigenvalues of random data without the
# random numbers: one of them has a communality at its bound, a Heywood case.
set.seed(SEED)
bfi_random <- lapply(seq_len(60), function(i) cor(apply(bfi_items, 2, sample), use = "pairwise.complete.obs"))
heywood <- which(sapply(bfi_random, function(r) max(abs(minres_loadings(r) - 0)) >= MAX_LOADING - 1e-9))
message("Permuted bfi matrices with a loading at its bound: ", paste(heywood, collapse = ", "))
random_matrices <- lapply(c(1:11, heywood), function(i) {
  e <- eigenvalues(bfi_random[[i]])
  list(matrix = unname(bfi_random[[i]]), components = e$components, factors = e$factors,
       loadings = minres_loadings(bfi_random[[i]]), psychFactors = psych_factor_values(bfi_random[[i]]))
})

bfi8 <- bfi_items[, 1:8]
sat10 <- sat12[, 1:10]
parallel_cases <- list(
  parallel_case("bfi, Pearson", bfi_items, cor(bfi_items, use = "pairwise.complete.obs"), "pearson", "cor", 2000, 2000),
  parallel_case("bfi, 8 items, polychoric", bfi8, cases_polychoric[[4]]$baseR[1:8, 1:8], "polychoric", "poly", 200),
  parallel_case("LSAT7, tetrachoric", lsat7, cases_polychoric[[1]]$baseR[1:5, 1:5], "polychoric", "tet", 300),
  parallel_case("SAT12, 10 items, tetrachoric", sat10, cases_polychoric[[3]]$baseR[1:10, 1:10], "polychoric", "tet",
                100)
)
write_fixture("dimensionality-parallel", list(
  source = paste("psych::bfi, mirt::LSAT7 and mirt::SAT12 as above. Observed eigenvalues: base R eigen and",
                 "L-BFGS-B minres loadings on base R correlations;",
                 "psych::fa.parallel(n.iter, fa = \"both\"); for bfi with",
                 "Pearson",
                 "correlations also 2000 permuted data sets in base R. `randomMatrices`: 60 permuted bfi data sets,",
                 "the first 11 and those with a loading at its bound."),
  maxLoading = MAX_LOADING,
  cases = parallel_cases,
  randomMatrices = random_matrices
))
