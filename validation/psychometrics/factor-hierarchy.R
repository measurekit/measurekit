# The Schmid-Leiman transformation and omega hierarchical (dimensionality-hierarchy.json). References: psych's schmid
# and omega(fm = "minres", flip = FALSE), and a base R write-up on the base R oblimin solutions of factor-solutions.R:
# the general factor's loadings on the group factors are the one-factor minres solution of their correlations, the
# general loading of an item is its oblique loadings times those, and its loading on a group factor the oblique loading
# times √(1 - g²). Omega hierarchical is (Σ general)² over the sum of the correlations, and omega total is 1 less the
# sum of the uniquenesses of the factors over it, as psych's omega computes both. psych reverses the items whose general
# loading is negative unless flip = FALSE; the package takes items as scored, so bfi's reversed items are scored back.
#
# Data: the 24 tests of datasets::Harman74.cor; psych::bfi with its reversed items scored back, with missing responses;
# and a seeded test of 12 items on a general factor and 3 group factors, whose omega hierarchical is known.

source("validation/psychometrics/common.R")
source("validation/psychometrics/factor-solutions.R")
suppressPackageStartupMessages(library(psych))

SEED <- 20261036

rows_of <- function(X) unname(lapply(seq_len(nrow(X)), function(i) as.numeric(X[i, ])))

# The signs and the order that orient() gives the factors of an oblique solution.
orientation <- function(loadings, phi) {
  signs <- ifelse(colSums(loadings) < 0, -1, 1)
  oriented <- loadings %*% diag(signs, ncol(loadings))
  turned <- diag(signs, ncol(loadings)) %*% phi %*% diag(signs, ncol(loadings))
  list(signs = signs, order = order(diag(turned %*% crossprod(oriented)), decreasing = TRUE))
}

# The transformation in base R, from an oriented oblique solution and its communalities.
schmid_leiman <- function(loadings, phi, communalities, r) {
  g <- drop(principal_axes(phi, 1, minres_communalities(phi, 1)))
  general <- drop(loadings %*% g)
  specific <- sign(loadings) * sqrt(pmax(loadings^2 %*% diag(1 - g^2, length(g)), 0))
  total <- sum(r)
  list(general = general, specific = unname(specific), communalities = communalities,
       hierarchical = sum(general)^2 / total, total = (total - sum(1 - communalities)) / total)
}

base_of <- function(r, m) {
  unrotated <- principal_axes(r, m, minres_communalities(r, m))
  oriented <- rotate_solution(unrotated, "oblimin")
  c(list(loadings = unname(oriented$loadings), phi = unname(oriented$phi)),
    schmid_leiman(oriented$loadings, oriented$phi, unname(rowSums(unrotated^2)), r))
}

# psych's schmid, its group factors put in the package's order and signs.
psych_schmid <- function(r, m) {
  s <- suppressWarnings(suppressMessages(schmid(r, m, fm = "minres")))
  turn <- orientation(unclass(s$oblique), s$phi)
  specific <- s$sl[, 1 + seq_len(m), drop = FALSE] %*% diag(turn$signs, m)
  list(general = unname(s$sl[, "g"]), specific = unname(specific[, turn$order, drop = FALSE]),
       communalities = unname(s$sl[, "h2"]))
}

psych_omega <- function(x, m) {
  o <- suppressWarnings(suppressMessages(omega(x, m, fm = "minres", flip = FALSE, plot = FALSE)))
  list(hierarchical = o$omega_h, total = o$omega.tot)
}

matrix_case <- function(name, r, m) {
  list(name = name, factors = m, matrix = unname(r), base = base_of(r, m), psych = psych_schmid(r, m))
}

data_case <- function(name, X, m, truth = NULL) {
  r <- cor(X, use = "pairwise")
  list(name = name, itemIds = colnames(X), responses = rows_of(X), factors = m, base = base_of(r, m),
       psych = c(psych_schmid(r, m), psych_omega(X, m)), truth = truth)
}

data(bfi, package = "psych")
reversed <- c("A1", "C4", "C5", "E1", "E2", "O2", "O5")
bfi_items <- as.matrix(bfi[, 1:25])
bfi_items[, reversed] <- 7 - bfi_items[, reversed]

# A general factor of loading 0.6 and three group factors of loading 0.4, four items each, for 2,000 persons, scored in
# hundredths from 1,000. Its omega hierarchical is (12 × 0.6)² over the sum of the population's correlations.
set.seed(SEED)
general_loading <- 0.6
group_loading <- 0.4
hierarchical <- local({
  n <- 2000
  general <- rnorm(n)
  groups <- matrix(rnorm(3 * n), n)
  X <- sapply(1:12, function(j) {
    unique <- sqrt(1 - general_loading^2 - group_loading^2)
    round(100 * (general_loading * general + group_loading * groups[, (j - 1) %/% 4 + 1] + unique * rnorm(n))) + 1000
  })
  colnames(X) <- sprintf("s%02d", 1:12)
  X
})
population <- local({
  block <- kronecker(diag(3), matrix(1, 4, 4))
  sigma <- general_loading^2 + group_loading^2 * block
  diag(sigma) <- 1
  sigma
})
truth <- (12 * general_loading)^2 / sum(population)

write_fixture("dimensionality-hierarchy", list(
  source = paste("psych::schmid and omega(fm = \"minres\", flip = FALSE) and a base R write-up on the base R oblimin",
                 "solutions, on datasets::Harman74.cor, psych::bfi with its reversed items scored back, and 12 seeded",
                 "items of a general and three group factors for 2,000 persons (set.seed(20261036))."),
  matrices = list(matrix_case("Harman's 24 tests, four factors", Harman74.cor$cov, 4)),
  data = list(
    data_case("bfi, five factors", bfi_items, 5),
    data_case("a general and three group factors", hierarchical, 3, truth = truth)
  )
))
