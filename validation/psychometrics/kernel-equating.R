# Kernel equating fixtures (equating-kernel.json). Reference: kequate (González and Wiberg) in each design, with
# getEq, getSee, getH, getPre and genseed, on log-linear fits by glm of the counts on powers of the scores scaled to
# [-1, 1], which span the space of the plain powers without their ill-conditioning. Data: kequate's simeq for the
# nonequivalent groups (its X margin and its Y margin as equivalent groups), and simulations with a fixed seed for the
# single group and the counterbalanced designs.
#
# kequate's counterbalanced design weighs the first group by wcb in r and in s, but its standard errors weigh that
# group's s by 1 - wcb. Below, the standard errors are written out in base R by the delta method through the
# parameters of the log-linear fits, with the bandwidths fixed, as kequate holds them: the write-up reproduces kequate
# at wcb = 1/2, and gives the reference for any other weight.

source("validation/psychometrics/common.R")
suppressPackageStartupMessages({
  library(kequate)
  library(numDeriv)
})

data(simeq, package = "kequate")
X_SCORES <- 0:20
A_SCORES <- 0:10
GLM_CONTROL <- glm.control(epsilon = 1e-14, maxit = 100)

# Powers of the scores scaled to [-1, 1].
scaled <- function(scores) (scores - mean(range(scores))) / (diff(range(scores)) / 2)
power_terms <- function(name, degree) sprintf("I(%s^%d)", name, seq_len(degree))

# The log-linear fit of a univariate frequency table of kefreq.
univariate_fit <- function(table, degree) {
  table$z <- scaled(table$X)
  formula <- as.formula(paste("frequency ~", paste(power_terms("z", degree), collapse = " + ")))
  glm(formula, family = "poisson", data = table, x = TRUE, control = GLM_CONTROL)
}

# The log-linear fit of a bivariate table of kefreq, its first score varying fastest: the powers of each score and the
# cross products z^i w^j for i and j from 1 to `cross`.
bivariate_fit <- function(table, degrees, cross) {
  table$z <- scaled(table[[1]])
  table$w <- scaled(table[[2]])
  products <- if (cross == 0) character(0) else
    as.vector(outer(seq_len(cross), seq_len(cross), function(i, j) sprintf("I(z^%d * w^%d)", i, j)))
  terms <- c(power_terms("z", degrees[1]), power_terms("w", degrees[2]), products)
  glm(as.formula(paste("frequency ~", paste(terms, collapse = " + "))), family = "poisson", data = table, x = TRUE,
      control = GLM_CONTROL)
}

# A table of kefreq, its first score in rows, as rows of counts.
table_rows <- function(table, rows) {
  counts <- matrix(table$frequency, nrow = rows)
  unname(lapply(seq_len(nrow(counts)), function(i) as.numeric(counts[i, ])))
}

# --- The data -------------------------------------------------------------------------------------------------------

from_neat <- kefreq(simeq$bivar1$X, X_SCORES, simeq$bivar1$A, A_SCORES)
to_neat <- kefreq(simeq$bivar2$Y, X_SCORES, simeq$bivar2$A, A_SCORES)
from_random <- kefreq(simeq$bivar1$X, X_SCORES)
to_random <- kefreq(simeq$bivar2$Y, X_SCORES)

# Two forms of 20 items, form Y harder by 0.25, taken by the same persons; in the counterbalanced design the form taken
# second is answered 0.1 lower in ability.
set.seed(20261102)
difficulties <- seq(-1.8, 1.8, length.out = 20)
form_score <- function(theta, shift) {
  rowSums(matrix(rbinom(length(theta) * 20, 1, plogis(outer(theta, difficulties + shift, "-"))), length(theta)))
}
single <- rnorm(1200)
joint <- kefreq(form_score(single, 0), X_SCORES, form_score(single, 0.25), X_SCORES)
first_group <- rnorm(800, 0.1)
second_group <- rnorm(900, -0.1)
first <- kefreq(form_score(first_group, 0), X_SCORES, form_score(first_group - 0.1, 0.25), X_SCORES)
second <- kefreq(form_score(second_group - 0.1, 0), X_SCORES, form_score(second_group, 0.25), X_SCORES)

# --- Base R: kernel equating and its standard errors through the log-linear parameters ------------------------------

# The Gaussian kernel continuization of probabilities r on scores x, with variance factor n / (n - 1) as kequate.
continuized <- function(r, x, n, h) {
  mean <- sum(x * r)
  variance <- n / (n - 1) * (sum(x^2 * r) - mean^2)
  a <- sqrt(variance / (variance + h^2))
  list(cdf = function(t) sum(r * pnorm((t - a * x - (1 - a) * mean) / (a * h))))
}

kernel_equated <- function(r, s, x, y, n, m, hx, hy) {
  from <- continuized(r, x, n, hx)
  to <- continuized(s, y, m, hy)
  sapply(x, function(score) {
    u <- from$cdf(score)
    uniroot(function(t) to$cdf(t) - u, c(min(y) - 50, max(y) + 50), tol = 1e-13)$root
  })
}

# r and s of the counterbalanced design from the coefficients of the two fits, the first group weighted by w in both.
counterbalanced_scores <- function(beta, fits, w) {
  sizes <- sapply(fits, function(fit) ncol(fit$x))
  probabilities <- lapply(seq_along(fits), function(k) {
    coefficients <- beta[(c(0, cumsum(sizes))[k] + 1):cumsum(sizes)[k]]
    m <- exp(fits[[k]]$x %*% coefficients)
    matrix(m / sum(m), nrow = length(X_SCORES))
  })
  list(r = w * rowSums(probabilities[[1]]) + (1 - w) * rowSums(probabilities[[2]]),
       s = w * colSums(probabilities[[1]]) + (1 - w) * colSums(probabilities[[2]]))
}

counterbalanced_errors <- function(fits, w, h) {
  beta <- unlist(lapply(fits, coef))
  n <- sum(fits[[1]]$y)
  equate <- function(b) {
    scores <- counterbalanced_scores(b, fits, w)
    kernel_equated(scores$r, scores$s, X_SCORES, X_SCORES, n, n, h[[1]], h[[2]])
  }
  jacobian <- numDeriv::jacobian(equate, beta)
  covariance <- as.matrix(Matrix::bdiag(lapply(fits, vcov)))
  list(equated = equate(beta), standardErrors = sqrt(diag(jacobian %*% covariance %*% t(jacobian))))
}

# --- The cases ------------------------------------------------------------------------------------------------------

expected_of <- function(result, design) {
  h <- getH(result)
  bandwidths <- if (design == "chained") {
    list(from = h$hxP, to = h$hyQ, fromAnchor = h$haP, toAnchor = h$haQ)
  } else list(from = h$hx, to = h$hy)
  pre <- getPre(result)
  expected <- list(equated = getEq(result), standardErrors = getSee(result), bandwidths = bandwidths,
                   percentRelativeError = pre[[1]])
  if (design == "chained") expected$anchorPercentRelativeError <- pre[[2]]
  expected
}

NO_OPTIONS <- setNames(list(), character(0))

kernel_case <- function(name, data, options, result) {
  list(name = name, data = data, options = options, expected = expected_of(result, data$design))
}

random_fits <- list(univariate_fit(from_random, 3), univariate_fit(to_random, 3))
random_data <- list(design = "equivalent-groups", from = from_random$frequency, to = to_random$frequency,
                    degrees = c(3, 3))
random <- kequate("EG", X_SCORES, X_SCORES, random_fits[[1]], random_fits[[2]])
random_linear <- kequate("EG", X_SCORES, X_SCORES, random_fits[[1]], random_fits[[2]], linear = TRUE)
random_penalty <- kequate("EG", X_SCORES, X_SCORES, random_fits[[1]], random_fits[[2]], KPEN = 0.01)
random_fixed <- kequate("EG", X_SCORES, X_SCORES, random_fits[[1]], random_fits[[2]], hx = 0.8, hy = 1.2)

joint_fit <- bivariate_fit(joint, c(3, 3), 1)
single_group <- kequate("SG", X_SCORES, X_SCORES, joint_fit)

counterbalanced_fits <- list(bivariate_fit(first, c(3, 3), 1), bivariate_fit(second, c(3, 3), 1))
counterbalanced <- kequate("CB", X_SCORES, X_SCORES, counterbalanced_fits[[1]], counterbalanced_fits[[2]])
weighted <- kequate("CB", X_SCORES, X_SCORES, counterbalanced_fits[[1]], counterbalanced_fits[[2]], wcb = 0.7)

# The write-up reproduces kequate's standard errors at the weight 1/2, to the 1e-5 by which the factor n / (n - 1) of
# the variances, which kequate's derivatives leave out, moves them, and its equated scores at any weight, where
# kequate's standard errors differ by far more.
written_half <- counterbalanced_errors(counterbalanced_fits, 0.5, getH(counterbalanced))
stopifnot(max(abs(written_half$standardErrors - getSee(counterbalanced))) < 1e-5)
written_weighted <- counterbalanced_errors(counterbalanced_fits, 0.7, getH(weighted))
stopifnot(max(abs(written_weighted$equated - getEq(weighted))) < 1e-6,
          max(abs(written_weighted$standardErrors - getSee(weighted))) > 1e-3)

neat_fits <- function(degrees, cross) list(bivariate_fit(from_neat, degrees, cross), bivariate_fit(to_neat, degrees, cross))
neat_one <- neat_fits(c(3, 3), 1)
neat_two <- neat_fits(c(4, 3), 2)
chained <- kequate("NEAT_CE", X_SCORES, X_SCORES, A_SCORES, neat_one[[1]], neat_one[[2]])
stratified <- kequate("NEAT_PSE", X_SCORES, X_SCORES, neat_one[[1]], neat_one[[2]])
stratified_weighted <- kequate("NEAT_PSE", X_SCORES, X_SCORES, neat_two[[1]], neat_two[[2]], w = 0.3)
chained_linear <- kequate("NEAT_CE", X_SCORES, X_SCORES, A_SCORES, neat_one[[1]], neat_one[[2]], linear = TRUE)

neat_data <- function(design, degrees, cross) {
  list(design = design, from = table_rows(from_neat, length(X_SCORES)), to = table_rows(to_neat, length(X_SCORES)),
       degrees = degrees, crossDegree = cross)
}
joint_data <- list(design = "single-group", joint = table_rows(joint, length(X_SCORES)), degrees = c(3, 3))
counterbalanced_data <- list(design = "counterbalanced", first = table_rows(first, length(X_SCORES)),
                             second = table_rows(second, length(X_SCORES)), degrees = c(3, 3))

cases <- list(
  kernel_case("equivalent groups", random_data, NO_OPTIONS, random),
  kernel_case("equivalent groups, linear", random_data, list(linear = TRUE), random_linear),
  kernel_case("equivalent groups, penalty", random_data, list(penaltyWeight = 0.01), random_penalty),
  kernel_case("equivalent groups, bandwidths", random_data, list(bandwidths = list(from = 0.8, to = 1.2)), random_fixed),
  kernel_case("single group", joint_data, NO_OPTIONS, single_group),
  kernel_case("counterbalanced", counterbalanced_data, NO_OPTIONS, counterbalanced),
  kernel_case("chained", neat_data("chained", c(3, 3), 1), NO_OPTIONS, chained),
  kernel_case("chained, linear", neat_data("chained", c(3, 3), 1), list(linear = TRUE), chained_linear),
  kernel_case("post-stratification", neat_data("post-stratification", c(3, 3), 1), NO_OPTIONS, stratified),
  kernel_case("post-stratification, weighted", neat_data("post-stratification", c(4, 3), 2), list(weight = 0.3),
              stratified_weighted)
)
weighted_case <- kernel_case("counterbalanced, weighted", counterbalanced_data, list(weight = 0.7), weighted)
weighted_case$expected$standardErrors <- written_weighted$standardErrors
weighted_case$expected$kequate <- getSee(weighted)
cases <- c(cases, list(weighted_case))
names(cases) <- sapply(cases, `[[`, "name")

# The standard error of the difference of two equatings of one design: kequate's of kernel and linear equating, which
# its equipercentile result holds, and genseed's of post-stratification and chained equating.
stopifnot(max(abs(random@equating$eqYxLIN - getEq(random_linear))) < 1e-8)
differences <- list(
  list(first = "equivalent groups", second = "equivalent groups, linear", expected = random@equating$SEEDYx),
  list(first = "post-stratification", second = "chained", expected = genseed(stratified, chained)@out$SEEDYx)
)

write_fixture("equating-kernel", list(
  source = paste("kequate 1.6.4 on glm fits of scaled powers: simeq's NEAT data and its margins as equivalent",
                 "groups; a single group and a counterbalanced design simulated with set.seed(20261102); the",
                 "counterbalanced standard errors at weight 0.7 by the delta method in base R."),
  cases = cases,
  differences = differences
))
