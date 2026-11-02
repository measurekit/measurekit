# Residual fit fixtures: the RMSD and mean deviation of each item (fit-rmsd.json) and the local-dependence chi-square
# of each pair of items (fit-local-dependence.json). References: TAM's models with IRT.RMSD, at TAM's 21 nodes, each
# calibration run to 1e-10; mirt's residuals(type = "LD") at mirt's 61 nodes, and at 21 for a population given as
# nodes. The items are the references' own: TAM's are read from its probabilities at the nodes, mirt's from coef.
# Data: seeded simulations of dichotomous items with missing responses and of polytomous items, and mirt's Science.

source("validation/psychometrics/common.R")
suppressPackageStartupMessages({
  library(mirt)
  library(TAM)
})

set.seed(20261104)
TAM_TOLERANCE <- 1e-10
# CDM adds 1e-10 to the posterior count of a node before it divides by it, which moves a mean deviation by about 1e-12.
CDM_EPSILON_EFFECT <- 1e-10
MISSING_SHARE <- 0.03

rows_of <- function(data) unname(lapply(seq_len(nrow(data)), function(i) as.numeric(data[i, ])))

simulate_items <- function(persons, slopes, thresholds) {
  theta <- rnorm(persons)
  data <- sapply(seq_along(thresholds), function(i) {
    vapply(theta, function(t) {
      weights <- exp(c(0, cumsum(slopes[i] * (t - thresholds[[i]]))))
      sample(0:length(thresholds[[i]]), 1, prob = weights / sum(weights))
    }, 0)
  })
  colnames(data) <- paste0("item", seq_along(thresholds))
  data
}

# --- RMSD: TAM --------------------------------------------------------------------------------------------------------

# measurekit's items from TAM's probabilities: ln(P_k / P_(k-1)) = a (θ - b_k) in the partial credit models, so a is
# the slope of the first log ratio between two nodes and b_k the log ratio at θ = 0 over -a. The script checks that the
# items give TAM's probabilities at every node.
tam_items <- function(model, data, family) {
  probabilities <- IRT.irfprob(model)
  nodes <- model$theta[, 1]
  zero <- which(abs(nodes) < 1e-12)
  categories <- apply(data, 2, max, na.rm = TRUE) + 1
  lapply(seq_len(ncol(data)), function(i) {
    K <- categories[[i]]
    ratios <- function(q) log(probabilities[i, 2:K, q] / probabilities[i, 1:(K - 1), q])
    a <- (ratios(zero + 1)[1] - ratios(zero)[1]) / (nodes[zero + 1] - nodes[zero])
    b <- -ratios(zero) / a
    rebuilt <- t(vapply(nodes, function(t) {
      weights <- exp(c(0, cumsum(a * (t - b))))
      weights / sum(weights)
    }, numeric(K)))
    stopifnot(max(abs(rebuilt - t(probabilities[i, 1:K, ]))) < 1e-10)
    if (family == "1pl") stopifnot(abs(a - 1) < 1e-10)
    switch(family,
      "1pl" = list(model = "1pl", parameters = list(a = 1, b = b, c = 0, d = 1)),
      "2pl" = list(model = "2pl", parameters = list(a = a, b = b, c = 0, d = 1)),
      pcm = list(model = "pcm", parameters = list(a = 1, b = I(b))),
      gpcm = list(model = "gpcm", parameters = list(a = a, b = I(b)))
    )
  })
}

# The mean deviation written out from TAM's posterior counts: Σ_q f_q Σ_k k (O_qk / O_q - P_k(θ_q)) / (K - 1), with f the
# mean posterior, TAM's pi.k, and K the item's categories.
mean_deviation <- function(model, data) {
  counts <- IRT.expectedCounts(model)[, , , 1]
  probabilities <- IRT.irfprob(model)
  weights <- model$pi.k[, 1]
  categories <- apply(data, 2, max, na.rm = TRUE) + 1
  vapply(seq_len(ncol(data)), function(i) {
    K <- categories[[i]]
    shares <- t(counts[i, 1:K, ]) / colSums(counts[i, 1:K, ])
    sum(weights * ((shares - t(probabilities[i, 1:K, ])) %*% (0:(K - 1)))) / (K - 1)
  }, 0)
}

# TAM's E-step weighs each node by the normal density of the estimated mean and variance; its pi.k is the mean posterior,
# which IRT.RMSD weighs the squared deviations by. IRT.RMSD's MD divides a matrix of nodes by items by the vector of the
# items' categories less 1, which R recycles down the nodes: with items of unequal categories each node of an item takes
# another item's divisor. The script checks that this reproduces IRT.RMSD's MD, and that the written-out mean deviation
# equals it for items of equal categories: the reference is the mean deviation written out.
tam_case <- function(name, data, family, fit) {
  model <- fit(data)
  nodes <- model$theta[, 1]
  density <- dnorm(nodes, model$beta[1, 1], sqrt(model$variance[1, 1]))
  rmsd <- IRT.RMSD(model)
  written <- mean_deviation(model, data)
  categories <- apply(data, 2, max, na.rm = TRUE) + 1
  if (length(unique(categories)) == 1) {
    stopifnot(max(abs(written - rmsd$MD$Group1)) < CDM_EPSILON_EFFECT)
  } else {
    counts <- IRT.expectedCounts(model)[, , , 1]
    probabilities <- IRT.irfprob(model)
    terms <- vapply(seq_len(ncol(data)), function(i) {
      K <- categories[[i]]
      shares <- t(counts[i, 1:K, ]) / colSums(counts[i, 1:K, ])
      model$pi.k[, 1] * drop((shares - t(probabilities[i, 1:K, ])) %*% (0:(K - 1)))
    }, numeric(length(nodes)))
    recycled <- colSums(terms / (categories - 1))
    stopifnot(max(abs(recycled - rmsd$MD$Group1)) < CDM_EPSILON_EFFECT, max(abs(written - rmsd$MD$Group1)) > 1e-3)
    message(name, ": IRT.RMSD's MD differs from the written-out mean deviation by up to ",
            signif(max(abs(written - rmsd$MD$Group1)), 3))
  }
  list(name = name, responses = rows_of(data), items = tam_items(model, data, family),
       population = list(nodes = nodes, weights = density / sum(density)),
       rmsd = rmsd$RMSD$Group1, meanDeviation = written, tamMeanDeviation = rmsd$MD$Group1)
}

control <- list(conv = TAM_TOLERANCE, convD = TAM_TOLERANCE, maxiter = 10000, progress = FALSE)
dichotomous <- simulate_items(1000, runif(10, 0.6, 1.8), as.list(seq(-1.8, 1.8, length.out = 10)))
dichotomous[matrix(runif(length(dichotomous)) < MISSING_SHARE, nrow(dichotomous))] <- NA
polytomous <- simulate_items(800, rep(1, 5), list(c(-1, 0.5), c(-0.5, 0, 0.8), c(0.2, 1), c(-1.2, -0.3, 0.4), 0.3))
generalized <- simulate_items(800, c(0.7, 1.4, 1, 1.8, 0.9), list(c(-1, 0.5), c(-0.5, 0, 0.8), c(0.2, 1),
                                                                   c(-1.2, -0.3, 0.4), -0.3))

write_fixture("fit-rmsd", list(
  source = paste("TAM 4.3-25: tam.mml and tam.mml.2pl run to 1e-10 at their 21 nodes; IRT.RMSD's RMSD and MD, with",
                 "the population the normal density of TAM's mean and variance at the nodes."),
  cases = list(
    tam_case("rasch, sparse", dichotomous, "1pl", function(data) tam.mml(data, control = control, verbose = FALSE)),
    tam_case("2pl, sparse", dichotomous, "2pl", function(data) tam.mml.2pl(data, control = control, verbose = FALSE)),
    tam_case("partial credit", polytomous, "pcm", function(data) tam.mml(data, control = control, verbose = FALSE)),
    tam_case("generalized partial credit", generalized, "gpcm",
             function(data) tam.mml.2pl(data, irtmodel = "GPCM", control = control, verbose = FALSE))
  )
))

# --- Local dependence: mirt -------------------------------------------------------------------------------------------

mirt_case <- function(name, data, itemtype, quadpts = NULL) {
  model <- mirt(data, 1, itemtype = itemtype, TOL = 1e-8, verbose = FALSE)
  coefficients <- coef(model, IRTpars = TRUE, simplify = TRUE)$items
  items <- lapply(seq_len(ncol(data)), function(i) {
    type <- tolower(extract.mirt(model, "itemtype")[i])
    family <- if (type == "graded") "grm" else type
    list(model = family, parameters = psychometrics_parameters(family, coefficients[i, !is.na(coefficients[i, ])]))
  })
  arguments <- list(model, type = "LD", df.p = TRUE, verbose = FALSE)
  if (!is.null(quadpts)) arguments$quadpts <- quadpts
  result <- do.call(residuals, arguments)
  statistic <- result$LD
  pairs <- which(upper.tri(statistic), arr.ind = TRUE)
  pairs <- pairs[order(pairs[, 1], pairs[, 2]), , drop = FALSE]
  categories <- extract.mirt(model, "K")
  entry <- list(name = name, responses = rows_of(data), items = items,
                chiSquare = statistic[pairs[, c(2, 1)]], cramersV = statistic[pairs],
                degreesOfFreedom = (categories[pairs[, 1]] - 1) * (categories[pairs[, 2]] - 1),
                significance = result$df.p[pairs])
  if (!is.null(quadpts)) {
    nodes <- seq(-6, 6, length.out = quadpts)
    entry$population <- list(nodes = nodes, weights = dnorm(nodes) / sum(dnorm(nodes)))
  }
  entry
}

science <- as.matrix(Science) - 1
colnames(science) <- paste0("item", seq_len(ncol(science)))
mixed <- simulate_items(1000, c(1.2, 0.8, 1.5, 1, 1.3, 0.9), list(0, -0.5, c(-1, 0.6), c(-0.4, 0.4), c(-1, 0, 1), 0.7))

write_fixture("fit-local-dependence", list(
  source = paste("mirt 1.48: residuals(type = \"LD\", df.p = TRUE) of mirt fits at TOL = 1e-8, at 61 nodes, and at 21",
                 "for the case with a population; the items are mirt's coefficients."),
  cases = list(
    mirt_case("2pl, sparse", dichotomous, "2PL"),
    mirt_case("2pl, sparse, 21 nodes", dichotomous, "2PL", quadpts = 21),
    mirt_case("graded, Science", science, "graded"),
    mirt_case("gpcm, mixed categories", mixed, "gpcm")
  )
))
