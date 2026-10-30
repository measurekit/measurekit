# IRT fixtures: ability estimates of learners answering calibrated items one at a time. The posterior on the grid, its
# mean and the exact posterior mode are written out here in base R; mirt's fscores, with the items' parameters fixed,
# is the reference the written-out estimates are checked against.

source("validation/knowledge-tracing/common.R")
suppressPackageStartupMessages(library(mirt))

# Items in the IRT parameterization, D = 1: dichotomous 1PL to 4PL, graded response and generalized partial credit.
ITEMS <- list(
  i1 = list(a = 1, b = -1),
  i2 = list(a = 1.4, b = -0.5),
  i3 = list(a = 0.8, b = 0.3),
  i4 = list(a = 2, b = 1),
  i5 = list(a = 1.2, b = 0, c = 0.2),
  i6 = list(a = 1.6, b = 0.8, c = 0.25),
  i7 = list(a = 1.1, b = -0.3, c = 0.15, d = 0.95),
  i8 = list(a = 1.8, b = 1.5, c = 0.1, d = 0.9),
  g1 = list(model = "graded", a = 1.3, b = c(-1, 0.2, 1.1)),
  g2 = list(model = "graded", a = 0.9, b = c(-0.5, 0.8)),
  p1 = list(model = "gpcm", a = 1, b = c(-0.7, 0.4, 1.2)),
  p2 = list(model = "gpcm", a = 1.5, b = c(0.5, -0.2))
)
categories <- function(item) if (is.null(item$model)) 2 else length(item$b) + 1

# P(category k) for k = 0 to K - 1 at theta, with arithmetic that also takes a complex theta.
probabilities <- function(item, theta) {
  if (is.null(item$model)) {
    lower <- item$c %||% 0
    upper <- item$d %||% 1
    p <- lower + (upper - lower) * logistic(item$a * (theta - item$b))
    return(list(1 - p, p))
  }
  if (item$model == "graded") {
    above <- c(list(1), lapply(item$b, function(b) logistic(item$a * (theta - b))), list(0))
    return(lapply(seq_len(categories(item)), function(k) above[[k]] - above[[k + 1]]))
  }
  z <- Reduce(`+`, lapply(item$b, function(b) item$a * (theta - b)), accumulate = TRUE)
  numerators <- c(list(exp(0 * theta)), lapply(z, exp))
  total <- Reduce(`+`, numerators)
  lapply(numerators, function(x) x / total)
}
category_probability <- function(item, k, theta) probabilities(item, theta)[[k + 1]]
expected_outcome <- function(item, theta) {
  p <- probabilities(item, theta)
  Reduce(`+`, lapply(seq_along(p), function(k) (k - 1) * p[[k]])) / (categories(item) - 1)
}

# The log-posterior of theta given responses (item ids and categories) and a normal prior, up to a constant.
log_posterior <- function(theta, ids, ks, prior) {
  value <- -((theta - prior$mean) / prior$sd)^2 / 2
  for (j in seq_along(ids)) value <- value + log(category_probability(ITEMS[[ids[j]]], ks[j], theta))
  value
}
# Its derivative by the complex step, exact to rounding, and the second derivative by central differences of it.
slope <- function(theta, ids, ks, prior) Im(log_posterior(complex(real = theta, imaginary = 1e-20), ids, ks, prior)) / 1e-20
curvature <- function(theta, ids, ks, prior) {
  (slope(theta + 1e-5, ids, ks, prior) - slope(theta - 1e-5, ids, ks, prior)) / 2e-5
}

# One learner's estimates after each response: EAP and its SD on the grid of `points` nodes from mean - 6 sd to mean +
# 6 sd, the exact MAP and 1 / sqrt(-curvature), and before each response the predicted outcome and its log-probability.
trace_learner <- function(ids, ks, prior, points) {
  grid <- prior$mean + prior$sd * seq(-6, 6, length.out = points)
  weights <- exp(-((grid - prior$mean) / prior$sd)^2 / 2)
  rows <- lapply(seq_along(ids), function(j) {
    item <- ITEMS[[ids[j]]]
    w <- weights / sum(weights)
    prediction <- sum(w * expected_outcome(item, grid))
    log_probability <- log(sum(w * category_probability(item, ks[j], grid)))
    weights <<- weights * category_probability(item, ks[j], grid)
    w <- weights / sum(weights)
    eap <- sum(w * grid)
    mode <- uniroot(slope, c(prior$mean - 6 * prior$sd, prior$mean + 6 * prior$sd), ids = ids[1:j], ks = ks[1:j],
                    prior = prior, tol = 1e-15)$root
    data.frame(prediction = prediction, logProbability = log_probability, eap = eap,
               eapStandardError = sqrt(sum(w * (grid - eap)^2)), map = mode,
               mapStandardError = 1 / sqrt(-curvature(mode, ids[1:j], ks[1:j], prior)))
  })
  do.call(rbind, rows)
}

set.seed(20261029)
ids <- names(ITEMS)
abilities <- c(rnorm(8), 2.5, -2.5)
learners <- sprintf("learner%02d", seq_along(abilities))
answers <- lapply(seq_along(abilities), function(l) {
  order <- sample(ids)
  ks <- vapply(order, function(id) {
    p <- unlist(probabilities(ITEMS[[id]], abilities[l]))
    sample(seq_along(p) - 1, 1, prob = p)
  }, numeric(1))
  list(ids = order, ks = unname(ks))
})
# Two learners at the extremes: every response in the top category, and every response in the lowest.
answers[[9]]$ks <- vapply(answers[[9]]$ids, function(id) categories(ITEMS[[id]]) - 1, numeric(1), USE.NAMES = FALSE)
answers[[10]]$ks <- rep(0, length(ids))

standard <- list(mean = 0, sd = 1)
shifted <- list(mean = 0.5, sd = 1.5)
traces <- lapply(answers, function(x) trace_learner(x$ids, x$ks, standard, 61))
shifted_traces <- lapply(answers, function(x) trace_learner(x$ids, x$ks, shifted, 41))

# mirt with the items' parameters fixed, in its slope-intercept form: d = -a b; graded d_k = -a b_k; gpcm d_k = -a
# (b_1 + ... + b_k).
dummy <- as.data.frame(lapply(ITEMS, function(item) rep(seq_len(categories(item)) - 1, length.out = 12)))
itemtype <- vapply(ITEMS, function(item) item$model %||% "4PL", character(1))
values <- mirt(dummy, 1, itemtype = unname(itemtype), pars = "values")
for (id in ids) {
  item <- ITEMS[[id]]
  set <- function(name, value) values$value[values$item == id & values$name == name] <<- value
  set("a1", item$a)
  if (is.null(item$model)) {
    set("d", -item$a * item$b)
    set("g", item$c %||% 0)
    set("u", item$d %||% 1)
  } else {
    d <- if (item$model == "graded") -item$a * item$b else -item$a * cumsum(item$b)
    for (k in seq_along(d)) set(paste0("d", k), d[k])
  }
}
values$est <- FALSE
fixed <- mirt(dummy, 1, itemtype = unname(itemtype), pars = values, TOL = NaN, verbose = FALSE)
pattern_rows <- do.call(rbind, lapply(answers, function(x) {
  t(vapply(seq_along(x$ids), function(j) {
    row <- setNames(rep(NA_real_, length(ids)), ids)
    row[x$ids[1:j]] <- x$ks[1:j]
    row
  }, numeric(length(ids))))
}))
mirt_eap <- fscores(fixed, method = "EAP", response.pattern = pattern_rows, quadpts = 61, theta_lim = c(-6, 6))
mirt_map <- fscores(fixed, method = "MAP", response.pattern = pattern_rows)
written <- do.call(rbind, traces)
gaps <- c(eap = max(abs(mirt_eap[, "F1"] - written$eap)),
          eapStandardError = max(abs(mirt_eap[, "SE_F1"] - written$eapStandardError)),
          map = max(abs(mirt_map[, "F1"] - written$map)),
          mapStandardError = max(abs(mirt_map[, "SE_F1"] - written$mapStandardError)))
print(gaps)
stopifnot(gaps["eap"] < 1e-8, gaps["eapStandardError"] < 1e-8, gaps["map"] < 1e-4, gaps["mapStandardError"] < 1e-3)

time <- 0
events <- list()
for (l in seq_along(answers)) {
  x <- answers[[l]]
  for (j in seq_along(x$ids)) {
    time <- time + 1
    events[[length(events) + 1]] <- list(learner = learners[l], item = x$ids[j], skills = I(character(0)), time = time,
                                         outcome = x$ks[j] / (categories(ITEMS[[x$ids[j]]]) - 1))
  }
}
write_fixture("irt-abilities", list(
  items = ITEMS,
  events = events,
  standard = list(prior = standard, points = 61, trace = as.list(written)),
  shifted = list(prior = shifted, points = 41, trace = as.list(do.call(rbind, shifted_traces))),
  mirt = list(eap = unname(mirt_eap[, "F1"]), eapStandardError = unname(mirt_eap[, "SE_F1"]),
              map = unname(mirt_map[, "F1"]), mapStandardError = unname(mirt_map[, "SE_F1"]))
))
