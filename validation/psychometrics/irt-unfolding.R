# Generalized graded unfolding fixtures (irt-unfolding.json). Reference: the GGUM package's GGUM(C), its α, δ and τ_1 to
# τ_C read as a, b and t_1 to t_C, on data its GenData.GGUM simulates: 10 seeds of 2,000 persons and 15 items of four
# categories, and one seed of 1,000 persons and 12 items of two to six categories; the starts of its first stage, by its
# GUM.initprm. Written out in base R where GGUM cannot be the reference:
# - GGUM stops when an EM cycle changes no parameter by more than `tol`. At its default, 0.001, a slow item stops short
#   of the maximum, so the cases take tol = 1e-6, within 1.4e-4 of tol = 1e-7 on seed 1; seed 1 is also fitted at the
#   default, and its log-likelihood, written out below, is lower.
# - GGUM integrates over 30 nodes from -4 to 4; the log-likelihood is written out at measurekit's 61 nodes from -6 to 6,
#   after the same code reproduces GGUM's at its own nodes.
# - GGUM's standard errors invert, item by item, that item's block of the information of the observed patterns, so
#   they leave out the covariance between items. The same information inverted whole is the cross-product estimator of
#   `calibrate`, written out at measurekit's nodes after the code reproduces GGUM's item by item. GGUM labels its
#   threshold errors by the upper half of the thresholds: SE.tau1 to SE.tauC are those of τ_C to τ_1.
# mirt's ggum item type writes the thresholds with the opposite sign, which the script checks on its probtrace.
# GGUM imports xlsx, which needs Java: run with JAVA_HOME set to a JDK. Each seed takes about 3 minutes, two at a time.

source("validation/psychometrics/common.R")
suppressPackageStartupMessages({
  library(GGUM)
  library(parallel)
})

SEEDS <- 1:10
PERSONS <- 2000
ITEMS <- 15
HIGHEST <- 3
MIXED <- list(seed = 11, persons = 1000, highest = rep(1:5, length.out = 12))
TOL <- 1e-6
MAX_OUTER <- 2000
NODES <- seq(-6, 6, length.out = 61)
GGUM_NODES <- seq(-4, 4, length.out = 30)
CORES <- 2
STEP <- 1e-5

# --- The GGUM written out ----------------------------------------------------------------------------------------

# P(z | θ) of an item with thresholds t at each θ, by θ then category.
item_probabilities <- function(a, b, t, theta) {
  highest <- length(t)
  z <- 0:highest
  cumulative <- matrix(cumsum(c(0, t)), length(theta), highest + 1, byrow = TRUE)
  lower <- a * (outer(theta - b, z) - cumulative)
  upper <- a * (outer(theta - b, 2 * highest + 1 - z) - cumulative)
  largest <- pmax(apply(lower, 1, max), apply(upper, 1, max))
  weights <- exp(lower - largest) + exp(upper - largest)
  weights / rowSums(weights)
}

# Each item's log P(x_n | θ_q) for every person and node, by person then node.
item_logs <- function(data, items, i, nodes) {
  logs <- log(item_probabilities(items$a[i], items$b[i], items$t[[i]], nodes))
  t(logs[, data[, i] + 1, drop = FALSE])
}

# Each person's log P(x_n | θ_q) by person then node, and the log weights of the normal density at the nodes.
pattern_logs <- function(data, items, nodes) {
  Reduce(`+`, lapply(seq_len(ncol(data)), function(i) item_logs(data, items, i, nodes)))
}
log_weights <- function(nodes) log(dnorm(nodes) / sum(dnorm(nodes)))

# Each person's log marginal likelihood.
person_likelihoods <- function(data, items, nodes) {
  joint <- sweep(pattern_logs(data, items, nodes), 2, log_weights(nodes), `+`)
  largest <- apply(joint, 1, max)
  largest + log(rowSums(exp(joint - largest)))
}

log_likelihood <- function(data, items, nodes) sum(person_likelihoods(data, items, nodes))

# Each person's score, the gradient of the log marginal likelihood, by person then parameter: a, b, t_1 to t_C of each
# item in turn, each by central differences of the item's log-probabilities under the person's posterior.
scores <- function(data, items, nodes) {
  joint <- sweep(pattern_logs(data, items, nodes), 2, log_weights(nodes), `+`)
  posterior <- exp(joint - apply(joint, 1, max))
  posterior <- posterior / rowSums(posterior)
  do.call(cbind, lapply(seq_len(ncol(data)), function(i) {
    vector <- c(items$a[i], items$b[i], items$t[[i]])
    sapply(seq_along(vector), function(p) {
      at <- function(shift) {
        moved <- vector
        moved[p] <- moved[p] + shift
        shifted <- items
        shifted$a[i] <- moved[1]
        shifted$b[i] <- moved[2]
        shifted$t[[i]] <- moved[-(1:2)]
        item_logs(data, shifted, i, nodes)
      }
      rowSums(posterior * (at(STEP) - at(-STEP)) / (2 * STEP))
    })
  }))
}

# The standard errors of the information of the persons' scores: inverted whole, or item by item as GGUM.
cross_product_errors <- function(data, items, nodes, by_item = FALSE) {
  information <- crossprod(scores(data, items, nodes))
  sizes <- 2 + lengths(items$t)
  if (!by_item) return(sqrt(diag(solve(information))))
  ends <- cumsum(sizes)
  unlist(lapply(seq_along(sizes), function(i) {
    block <- (ends[i] - sizes[i] + 1):ends[i]
    sqrt(diag(solve(information[block, block])))
  }))
}

# mirt's ggum item type writes the thresholds with the opposite sign: its probtrace of a, b and -t is the GGUM's.
local({
  data <- matrix(rep(0:3, 8), 8, 4, dimnames = list(NULL, paste0("item", 1:4)))
  values <- mirt::mirt(data, 1, itemtype = "ggum", pars = "values", verbose = FALSE)
  first <- values$item == "item1"
  values$value[first] <- c(1.3, 0.4, 1.2, 0.8, 0.3)[match(values$name[first], c("a1", "b1", "t1", "t2", "t3"))]
  item <- mirt::extract.item(mirt::mirt(data, 1, itemtype = "ggum", pars = values, TOL = NaN, verbose = FALSE), 1)
  theta <- c(-1, 0.5, 2)
  stopifnot(max(abs(mirt::probtrace(item, theta) - item_probabilities(1.3, 0.4, c(-1.2, -0.8, -0.3), theta))) < 1e-14)
})

# --- Fits -------------------------------------------------------------------------------------------------------

# The persons GGUM keeps: those with a response at or above an item's middle category, or its first agree category.
kept <- function(data, highest) which(colSums(t(data) >= ceiling(highest / 2)) > 0)

# GGUM's estimates as items, its thresholds τ_1 to τ_C of each item, which it aligns right when C differs.
items_of <- function(fit, highest) {
  most <- max(highest)
  list(a = fit$alpha, b = fit$delta,
       t = lapply(seq_along(highest), function(i) unname(fit$taus[i, (most - highest[i] + 1):most])))
}

# GGUM's standard errors in the order of the items' vectors: SE.tau_j is that of τ_(C+1-j).
errors_of <- function(fit, highest) {
  unlist(lapply(seq_along(highest), function(i) {
    se <- fit$SE[i, ]
    c(se[1:2], rev(se[2 + seq_len(highest[i])]))
  }), use.names = FALSE)
}

fit_ggum <- function(data, highest, tol = TOL) {
  output <- capture.output(fit <- GGUM(data, highest, SE = TRUE, precision = 15, tol = tol, max.outer = MAX_OUTER,
                                       max.inner = 200))
  stopifnot(fit$tol.conv < tol)
  fit
}

# A case: the responses, the true items, GGUM's items and standard errors, the persons it kept, the log-likelihood and
# the cross-product errors at measurekit's nodes.
unfolding_case <- function(generated, highest, fit = fit_ggum(generated$data, highest)) {
  data <- generated$data
  persons <- kept(data, highest)
  stopifnot(length(persons) == nrow(data) - length(fit$rows.rm))
  used <- data[persons, , drop = FALSE]
  items <- items_of(fit, highest)
  stopifnot(abs(log_likelihood(used, items, GGUM_NODES) - fit$InformationCrit$log.L) < 1e-6)
  ggum_errors <- errors_of(fit, highest)
  stopifnot(max(abs(cross_product_errors(used, items, GGUM_NODES, by_item = TRUE) / ggum_errors - 1)) < 1e-6)
  errors <- cross_product_errors(used, items, NODES)
  truth <- list(a = generated$alpha.gen, b = generated$delta.gen,
                t = lapply(seq_along(highest), function(i) {
                  most <- max(highest)
                  unname(generated$taus.gen[i, (most - highest[i] + 1):most])
                }))
  list(
    responses = apply(data, 1, paste, collapse = ""),
    truth = item_list(truth),
    persons = length(persons),
    expected = list(items = item_list(items, ggum_errors), logLikelihood = log_likelihood(used, items, NODES),
                    crossProductErrors = split_errors(errors, highest))
  )
}

# Items as a list of { a, b, t }, with GGUM's standard errors by the names of the vector where given.
item_list <- function(items, errors = NULL) {
  starts <- cumsum(c(0, 2 + lengths(items$t)))
  lapply(seq_along(items$a), function(i) {
    item <- list(a = items$a[i], b = items$b[i], t = I(items$t[[i]]))
    if (is.null(errors)) return(item)
    entries <- errors[(starts[i] + 1):starts[i + 1]]
    c(item, list(standardErrors = setNames(as.list(entries), c("a", "b", paste0("t", seq_along(items$t[[i]]))))))
  })
}

split_errors <- function(errors, highest) {
  starts <- cumsum(c(0, 2 + highest))
  lapply(seq_along(highest), function(i) {
    setNames(as.list(errors[(starts[i] + 1):starts[i + 1]]), c("a", "b", paste0("t", seq_len(highest[i]))))
  })
}

# Case names as arguments run only those cases and keep the others of the fixture: "seed 1" to "seed 10", "mixed",
# "starts" and "default tolerance", which reads seed 1's.
requested <- commandArgs(trailingOnly = TRUE)
selected <- function(name) length(requested) == 0 || name %in% requested
PATH <- "packages/psychometrics/test/fixtures/irt-unfolding.json"
fixture <- if (length(requested) > 0 && file.exists(PATH)) {
  jsonlite::fromJSON(PATH, simplifyVector = FALSE)
} else {
  list(cases = list())
}

seed_names <- Filter(selected, paste("seed", SEEDS))
seed_cases <- mclapply(seed_names, function(name) {
  seed <- as.integer(sub("seed ", "", name))
  unfolding_case(GenData.GGUM(PERSONS, ITEMS, HIGHEST, seed = seed), rep(HIGHEST, ITEMS))
}, mc.cores = CORES)
fixture$cases[seed_names] <- seed_cases

if (selected("mixed")) {
  generated <- GenData.GGUM(MIXED$persons, length(MIXED$highest), MIXED$highest, seed = MIXED$seed)
  fixture$cases$mixed <- unfolding_case(generated, MIXED$highest)
}

# The starts of GGUM's first stage, Roberts and Laughlin's procedure in GUM.initprm, on all the persons of each seed.
if (selected("starts")) {
  fixture$starts <- setNames(lapply(SEEDS, function(seed) {
    data <- GenData.GGUM(PERSONS, ITEMS, HIGHEST, seed = seed)$data
    starts <- GGUM:::GUM.initprm(GGUM:::GGUM.data.condense(data)$data.condensed, rep(HIGHEST, ITEMS))
    list(b = starts$delta.ini, t = lapply(seq_len(ITEMS), function(i) I(starts$taus.ini[i, 1:HIGHEST])))
  }), paste("seed", SEEDS))
}

# Seed 1 at GGUM's defaults: its estimates, and its log-likelihood and that of tol = 1e-6, both at GGUM's nodes.
if (selected("default tolerance")) {
  first <- GenData.GGUM(PERSONS, ITEMS, HIGHEST, seed = 1)
  used <- first$data[kept(first$data, rep(HIGHEST, ITEMS)), ]
  output <- capture.output(default_fit <- GGUM(first$data, HIGHEST, SE = FALSE, precision = 15))
  default_items <- items_of(default_fit, rep(HIGHEST, ITEMS))
  tight <- fixture$cases[["seed 1"]]$expected$items
  tight_items <- list(a = sapply(tight, function(item) item$a), b = sapply(tight, function(item) item$b),
                      t = lapply(tight, function(item) unlist(item$t)))
  default_likelihood <- log_likelihood(used, default_items, GGUM_NODES)
  tight_likelihood <- log_likelihood(used, tight_items, GGUM_NODES)
  stopifnot(tight_likelihood > default_likelihood)
  fixture$defaultTolerance <- list(case = "seed 1", items = item_list(default_items),
                                   logLikelihood = default_likelihood, tightLogLikelihood = tight_likelihood)
}

fixture$source <- paste("The GGUM package's GGUM(C, tol = 1e-6) on data of its GenData.GGUM; log-likelihoods and",
                        "cross-product standard errors written out in base R at 61 nodes from -6 to 6.")
fixture$cases <- fixture$cases[intersect(c(paste("seed", SEEDS), "mixed"), names(fixture$cases))]
write_fixture("irt-unfolding", fixture[intersect(c("source", "cases", "starts", "defaultTolerance"), names(fixture))])
