# Longitudinal item response theory: the same persons answer the same items on two or three occasions, each with its own
# trait. measurekit's calibrateLongitudinal against mirt's mirt, with one dimension per occasion, the occasion's items as
# columns, the parameters of an item equal across the occasions that have it (constrain), the first occasion N(0, 1) and
# the means, variances and covariances of the others free (MEAN, COV); and, for the Rasch model, TAM's tam.mml with one
# dimension per occasion and the item parameters tied by A. The data are simulated with set.seed(20261101).

source("validation/psychometrics/common.R")
source("validation/psychometrics/longitudinal-mirt.R")
suppressPackageStartupMessages(library(TAM))

cases <- list()

# --- Two occasions, attrition ----------------------------------------------------------------------------------------------

# 1,500 persons, 10 items on both occasions; 12% of them do not take part in the second.
cases$two <- local({
  set.seed(20261101)
  items <- simulate_items(10)
  thetas <- simulate_abilities(1500, c(0, 0.4), c(1, 1.2), autoregressive(2, 0.8))
  occasions <- simulate_responses(thetas, items, list(1:10, 1:10), list(integer(), sort(sample(1500, 180))))
  run <- fit_mirt(occasions, "2PL", 31, errors = TRUE)
  crossproduct <- mirt_errors(crossproduct_errors(run, "2PL", 31))
  c(
    list(
      source = "Simulated with set.seed(20261101): 1,500 persons, 10 dichotomous items on 2 occasions, abilities N((0, 0.4), sds (1, 1.2), correlation 0.8); 180 persons absent from the second occasion.",
      model = "2pl", points = 31, occasions = occasion_records(occasions), items = mirt_items(run, occasions, FALSE),
      standardErrors = I(mirt_errors(run$fit)), crossProductStandardErrors = I(crossproduct)
    ),
    mirt_moments(run), mirt_statistics(run)
  )
})

# --- Three occasions ---------------------------------------------------------------------------------------------------------

# 300 persons, 8 items on three occasions; 15% do not take part in the third. Three occasions are on 13 nodes, whose grid
# of 13 ** 3 nodes keeps measurekit's calibration to about a second; the tests hold the default of 21 apart.
cases$three <- local({
  set.seed(20261101)
  items <- simulate_items(8)
  thetas <- simulate_abilities(300, c(0, 0.3, 0.7), c(1, 1.15, 1.3), autoregressive(3, 0.75))
  occasions <- simulate_responses(thetas, items, list(1:8, 1:8, 1:8), list(integer(), integer(), sort(sample(300, 45))))
  run <- fit_mirt(occasions, "2PL", 13)
  c(
    list(
      source = "Simulated with set.seed(20261101): 300 persons, 8 dichotomous items on 3 occasions, abilities N((0, 0.3, 0.7), sds (1, 1.15, 1.3), autoregressive correlation 0.75); 45 persons absent from the third occasion.",
      model = "2pl", points = 13, occasions = occasion_records(occasions), items = mirt_items(run, occasions, FALSE)
    ),
    mirt_moments(run), mirt_statistics(run)
  )
})

# --- Items on one occasion only ----------------------------------------------------------------------------------------------

# Items q09 and q10 are only on the first occasion, q11 and q12 only on the second; the two occasions share q01 to q08.
cases$anchored <- local({
  set.seed(20261101)
  items <- simulate_items(12)
  thetas <- simulate_abilities(1200, c(0, 0.5), c(1, 0.9), autoregressive(2, 0.7))
  occasions <- simulate_responses(thetas, items, list(c(1:8, 9, 10), c(1:8, 11, 12)), list(integer(), integer()))
  run <- fit_mirt(occasions, "2PL", 31)
  c(
    list(
      source = "Simulated with set.seed(20261101): 1,200 persons, 12 dichotomous items, 8 on both of 2 occasions and 2 on each only, abilities N((0, 0.5), sds (1, 0.9), correlation 0.7).",
      model = "2pl", points = 31, occasions = occasion_records(occasions), items = mirt_items(run, occasions, FALSE)
    ),
    mirt_moments(run), mirt_statistics(run)
  )
})

# --- The graded model ----------------------------------------------------------------------------------------------------------

cases$graded <- local({
  set.seed(20261101)
  items <- simulate_items(6, categories = 4)
  thetas <- simulate_abilities(1200, c(0, 0.5), c(1, 1.1), autoregressive(2, 0.8))
  occasions <- simulate_responses(thetas, items, list(1:6, 1:6), list(integer(), integer()))
  run <- fit_mirt(occasions, "graded", 31)
  c(
    list(
      source = "Simulated with set.seed(20261101): 1,200 persons, 6 items of 4 categories on 2 occasions, abilities N((0, 0.5), sds (1, 1.1), correlation 0.8).",
      model = "grm", points = 31, occasions = occasion_records(occasions), items = mirt_items(run, occasions, TRUE)
    ),
    mirt_moments(run), mirt_statistics(run)
  )
})

# --- The Rasch model, against mirt and TAM --------------------------------------------------------------------------------------

# TAM's Rasch model on one dimension per occasion has slopes 1 and a free variance for the first occasion; measurekit's 1PL
# has the first occasion N(0, 1) and one slope a. They are one model: a is the first occasion's standard deviation in
# TAM, and its difficulties, means and standard deviations are TAM's over it.
tam_rasch <- function(occasions, points) {
  data <- wide_data(occasions)
  T <- length(occasions)
  ids <- unique(unlist(lapply(occasions, colnames)))
  columns <- colnames(data)
  design <- matrix(0, ncol(data), T, dimnames = list(columns, paste0("F", seq_len(T))))
  A <- array(0, c(ncol(data), 2, length(ids)), dimnames = list(columns, c("Cat0", "Cat1"), ids))
  for (t in seq_len(T)) {
    for (id in colnames(occasions[[t]])) {
      design[paste0(id, "_t", t), t] <- 1
      A[paste0(id, "_t", t), "Cat1", id] <- -1
    }
  }
  fit <- tam.mml(data, Q = design, A = A, beta.fixed = cbind(1, 1, 0), irtmodel = "1PL", verbose = FALSE,
                 control = list(nodes = seq(-6, 6, length.out = points), maxiter = 5000, conv = 1e-8, convD = 1e-8))
  stopifnot(fit$iter < 5000)
  scale <- sqrt(fit$variance[1, 1])
  covariance <- fit$variance / scale^2
  list(
    slope = scale,
    difficulties = I(unname(fit$xsi$xsi[match(ids, rownames(fit$xsi))] / scale)),
    means = I(unname(fit$beta[1, ] / scale)),
    sds = I(unname(sqrt(diag(covariance)))),
    correlations = unname(cov2cor(covariance)),
    logLikelihood = -fit$deviance / 2
  )
}

# Two occasions, as the model's slope and scale do not depend on their number and the three case holds three: on 15 nodes
# per occasion mirt's estimates of three occasions are 4e-3 from measurekit's, and on 21 the calibration takes many seconds.
cases$rasch <- local({
  set.seed(20261101)
  items <- simulate_items(8)
  items$a <- rep(1, 8)
  thetas <- simulate_abilities(800, c(0, 0.4), c(1.1, 1.2), autoregressive(2, 0.8))
  occasions <- simulate_responses(thetas, items, list(1:8, 1:8), list(integer(), sort(sample(800, 80))))
  run <- fit_mirt(occasions, "2PL", 31, common_slope = TRUE)
  c(
    list(
      source = "Simulated with set.seed(20261101): 800 persons, 8 Rasch items on 2 occasions, abilities N((0, 0.4), sds (1.1, 1.2), correlation 0.8); 80 persons absent from the second occasion. mirt: slopes constrained equal; TAM: tam.mml with slopes 1, converted to the scale of the first occasion.",
      model = "1pl", points = 31, occasions = occasion_records(occasions), items = mirt_items(run, occasions, FALSE),
      tam = tam_rasch(occasions, 31)
    ),
    mirt_moments(run), mirt_statistics(run)
  )
})

write_fixture("longitudinal", cases)
