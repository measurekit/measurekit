# How well mirt recovers the moments of the occasions of a longitudinal model, which fixes what measurekit's calibration
# is held to and why it stops at three occasions. Persons answer the same dichotomous items on 2 to 5 occasions with an
# autoregressive correlation of 0.85; each fit is mirt's at its defaults (its `quadpts` for the number of dimensions, a
# tolerance of 1e-4 and 500 cycles), with the parameters of an item tied across the occasions and the moments of the
# later occasions free. The error of a fit is the largest distance of an estimate from the moments of the simulated
# abilities themselves, so that the sampling of the persons is no part of it: the means and standard deviations of the
# occasions after the first, in units of the first occasion's sample standard deviation and from its sample mean, and the
# correlations. The data are simulated with set.seed(20261101 + seed).
#
# The fits of the seeds of a row run in parallel on 4 cores, and each row is written to the fixture as it finishes, so
# that a run cut short keeps its rows. The whole run stops after 60 minutes. mirt is not fitted at 40 items per occasion,
# where its time grows with the cube of the nodes; the guide shows that grid with measurekit's own convergence.

source("validation/psychometrics/common.R")
source("validation/psychometrics/longitudinal-mirt.R")
suppressPackageStartupMessages(library(parallel))

CORES <- 4
LIMIT_MINUTES <- 60
MEANS <- 0.3 * (0:4)
SDS <- 1 + 0.15 * (0:4)
RHO <- 0.85
setTimeLimit(elapsed = LIMIT_MINUTES * 60, transient = FALSE)

# The error of a mirt fit, by what it estimates, from the moments of the abilities the persons were given.
moment_errors <- function(run, thetas) {
  moments <- coef(run$fit, simplify = TRUE)
  sds <- sqrt(diag(moments$cov))
  correlations <- cov2cor(moments$cov)
  first <- sd(thetas[, 1])
  truth <- cor(thetas)
  later <- seq_len(ncol(thetas))[-1]
  list(mean = max(abs(moments$means[later] - (colMeans(thetas)[later] - mean(thetas[, 1])) / first)),
       sd = max(abs(sds[later] - apply(thetas, 2, sd)[later] / first)),
       correlation = max(abs((correlations - truth)[lower.tri(truth)])))
}

# One fit: `persons` persons answering `items` items on `occasions` occasions, at `points` nodes per occasion.
fit_seed <- function(seed, persons, items, occasions, points) {
  set.seed(20261101 + seed)
  master <- simulate_items(items)
  thetas <- simulate_abilities(persons, MEANS[seq_len(occasions)], SDS[seq_len(occasions)], autoregressive(occasions, RHO))
  data <- simulate_responses(thetas, master, rep(list(seq_len(items)), occasions), rep(list(integer()), occasions))
  started <- proc.time()[["elapsed"]]
  run <- fit_mirt(data, "2PL", points, tolerance = 1e-4)
  c(list(seed = seed), moment_errors(run, thetas),
    list(iterations = extract.mirt(run$fit, "iterations"), converged = extract.mirt(run$fit, "converged"),
         seconds = proc.time()[["elapsed"]] - started))
}

# A row: the seeds fitted in parallel, and the worst error of each kind.
study <- function(persons, items, occasions, points, seeds) {
  fits <- mclapply(seeds, fit_seed, persons = persons, items = items, occasions = occasions, points = points,
                   mc.cores = min(CORES, length(seeds)), mc.preschedule = FALSE)
  failed <- vapply(fits, inherits, TRUE, what = "try-error")
  stopifnot(!any(failed))
  message(sprintf("%d persons, %d items, %d occasions, %d nodes: %d fits, %.0f s the slowest", persons, items, occasions,
                  points, length(fits), max(sapply(fits, `[[`, "seconds"))))
  list(persons = persons, itemsPerOccasion = items, occasions = occasions, points = points, fits = fits,
       worst = list(mean = max(sapply(fits, `[[`, "mean")), sd = max(sapply(fits, `[[`, "sd")),
                    correlation = max(sapply(fits, `[[`, "correlation"))))
}

fixture <- list(
  source = "mirt at its defaults, set.seed(20261101 + seed): persons answer the same 2PL items on 2 to 5 occasions, abilities with means 0, 0.3, 0.6, 0.9, 1.2, standard deviations 1, 1.15, 1.3, 1.45, 1.6 and an autoregressive correlation of 0.85. The error is the largest distance of an estimate from the mean, standard deviation or correlation of the simulated abilities.",
  truth = list(means = I(MEANS), sds = I(SDS), rho = RHO),
  recovery = list(),
  grid = list()
)
save_row <- function(part, row) {
  fixture[[part]] <<- c(fixture[[part]], list(row))
  write_fixture("longitudinal-recovery", fixture)
}

# What the recovery tests hold measurekit to: 5,000 persons and 10 items, at the nodes of measurekit's defaults. With
# more items the posteriors narrow and the errors fall, so these bound the tests at 40 items too.
save_row("recovery", study(5000, 10, 2, 31, 1:3))
save_row("recovery", study(5000, 10, 3, 21, 1:3))

# Why a calibration has at most three occasions: 1,500 persons and 10 items, at mirt's default nodes for 2 to 5
# dimensions (31, 15, 9 and 7), and at 21 for 3. The fits converge.
save_row("grid", study(1500, 10, 2, 31, 1:3))
save_row("grid", study(1500, 10, 3, 15, 1:3))
save_row("grid", study(1500, 10, 3, 21, 1:3))
save_row("grid", study(1500, 10, 4, 9, 1))
save_row("grid", study(1500, 10, 5, 7, 1))
