# IRT fixtures. `abilities`: ability estimates of learners answering calibrated items one at a time. The posterior on
# the grid, its mean and the exact posterior mode are written out here in base R; mirt's fscores, with the items'
# parameters fixed, is the reference the written-out estimates are checked against. `drift`: the posterior of a θ that
# drifts between events, by forward passes in logs on the model's grid and on a grid of 1921 points, and a particle
# filter.
# Case names on the command line run only those cases: `Rscript validation/knowledge-tracing/irt.R drift`.

source("validation/knowledge-tracing/common.R")
suppressPackageStartupMessages(library(mirt))

requested <- commandArgs(trailingOnly = TRUE)
selected <- function(name) length(requested) == 0 || name %in% requested

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

if (selected("abilities")) {
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
}

# --- Drift ------------------------------------------------------------------------------------------------------------

# Online IRT with drift: θ is a random walk whose variance grows by VARIANCE every PER units of time. On a grid of n
# points with spacing h, the chain jumps to either neighbor at the rate VARIANCE / (2 PER h²) and stays at an end when
# a jump would leave the grid; over a gap with t = VARIANCE Δt / (PER h²) jumps on average, a point moves d cells on the
# infinite line with probability e^-t I_d(t), and folding the line at the grid's ends, by images, gives the reflecting
# chain. Each pass below is a forward pass in logs with that kernel, so that no tail underflows.

VARIANCE <- 0.02
PER <- 7 * 24 * 3600 * 1000 # a week of epoch milliseconds
MINUTE <- 60 * 1000
HOUR <- 60 * MINUTE
DAY <- 24 * HOUR
WEEK <- 7 * DAY
START <- 1767225600000 # 2026-01-01T00:00:00Z
FINE_POINTS <- 1921
KERNEL_CUT <- log(1e-300) # the fine pass leaves out moves less likely than this

# ln(e^-t I_d(t)) for distances d ≥ 0: the positive series of I_d in logs, with lgamma.
log_kernel <- function(d, t) {
  m <- 0:ceiling(t / 2 + 12 * sqrt(t + 1) + 60)
  terms <- outer(d, m, function(d, m) (2 * m + d) * log(t / 2) - lgamma(m + 1) - lgamma(m + d + 1))
  top <- apply(terms, 1, max)
  top + log(rowSums(exp(terms - top))) - t
}
# The series against besselI(expon.scaled = TRUE), where that keeps its precision.
check_kernel <- function(t) {
  d <- 0:400
  series <- log_kernel(d, t)
  usable <- series > -600
  gap <- max(abs(series[usable] - log(besselI(t, d[usable], expon.scaled = TRUE))))
  stopifnot(gap < 1e-11)
  gap
}

log_sum_exp <- function(x) {
  top <- max(x)
  if (!is.finite(top)) return(top)
  top + log(sum(exp(x - top)))
}
# The grid cell of a position on the infinite line, folded at -1/2 and n - 1/2.
fold <- function(x, n) {
  m <- x %% (2 * n)
  ifelse(m < n, m, 2 * n - 1 - m)
}

# The transition of the reflecting chain on n points over t jumps on average, in logs: entry (i, j) is ln P(i to j),
# summed over the images of j within 4n cells, beyond which no term reaches rounding.
log_transition <- function(n, t) {
  reach <- 4 * n
  kernel <- log_kernel(0:reach, t)
  out <- matrix(-Inf, n, n)
  from <- 0:(n - 1)
  for (offset in -reach:reach) {
    index <- cbind(from + 1, fold(from + offset, n) + 1)
    value <- kernel[abs(offset) + 1]
    current <- out[index]
    top <- pmax(current, value)
    out[index] <- top + log(exp(current - top) + exp(value - top))
  }
  out
}

# A step of the same-grid pass: ln p'_j = log_sum_exp_i (ln p_i + ln P(i to j)).
same_grid_step <- function(log_density, transition) {
  terms <- transition + log_density
  top <- apply(terms, 2, max)
  top + log(colSums(exp(terms - rep(top, each = nrow(terms)))))
}

# The kernel of the fine pass over t jumps on average, cut where it falls below 1e-300.
fine_kernel <- function(t) {
  kernel <- log_kernel(0:(4 * FINE_POINTS), t)
  kernel[seq_len(max(which(kernel >= KERNEL_CUT)))]
}
# A step of the fine pass, the density extended by images: ln p'_j = log_sum_exp_y (ln p_fold(y) + ln K(j - y)).
fine_step <- function(log_density, kernel) {
  n <- length(log_density)
  reach <- length(kernel) - 1
  extended <- log_density[fold(-reach:(n - 1 + reach), n) + 1]
  base <- (0:(n - 1)) + reach + 1
  top <- rep(-Inf, n)
  for (offset in -reach:reach) top <- pmax(top, extended[base - offset] + kernel[abs(offset) + 1])
  total <- numeric(n)
  for (offset in -reach:reach) total <- total + exp(extended[base - offset] + kernel[abs(offset) + 1] - top)
  top + log(total)
}

log_probability <- function(theta, a, b, y) plogis(a * (theta - b), log.p = TRUE, lower.tail = y == 1)

# The posterior mode on a fine grid: the vertex of the parabola through the highest point and its neighbors.
grid_mode <- function(log_density, z) {
  q <- which.max(log_density)
  if (q == 1 || q == length(z)) return(z[q])
  h <- z[2] - z[1]
  left <- log_density[q - 1]
  right <- log_density[q + 1]
  z[q] - h * (right - left) / (2 * (right - 2 * log_density[q] + left))
}

# Runs a learner's events, sorted by time, through a pass on n points; `step` advances a log-density by t jumps on
# average. Returns the mean, SD and mode of θ at the checkpoints, and the forward log-likelihood.
forward_pass <- function(events, n, step) {
  z <- seq(-6, 6, length.out = n)
  h <- z[2] - z[1]
  log_density <- -z^2 / 2
  lowest <- log_density
  last <- NA
  log_likelihood <- 0
  rows <- list()
  for (e in seq_len(nrow(events))) {
    event <- events[e, ]
    gap <- if (is.na(last)) 0 else max(0, event$time - last)
    last <- max(last, event$time, na.rm = TRUE)
    if (gap > 0) log_density <- step(log_density, VARIANCE * gap / (PER * h^2))
    log_density <- log_density - max(log_density)
    terms <- log_probability(z, event$a, event$b, event$outcome)
    log_likelihood <- log_likelihood + log_sum_exp(log_density + terms) - log_sum_exp(log_density)
    log_density <- log_density + terms
    log_density <- log_density - max(log_density)
    lowest <- pmin(lowest, log_density)
    if (event$checkpoint) {
      w <- exp(log_density) / sum(exp(log_density))
      mean <- sum(w * z)
      rows[[length(rows) + 1]] <- c(mean = mean, sd = sqrt(sum(w * (z - mean)^2)), mode = grid_mode(log_density, z))
    }
  }
  trace <- as.data.frame(do.call(rbind, rows))
  list(mean = trace$mean, sd = trace$sd, mode = trace$mode, logLikelihood = log_likelihood,
       depth = -lowest[which.max(log_density)])
}

# Returns f(t), computed once for each t.
cached <- function(f) {
  cache <- new.env()
  function(t) {
    key <- sprintf("%.17g", t)
    if (is.null(cache[[key]])) cache[[key]] <- f(t)
    cache[[key]]
  }
}
same_grid_pass <- function(events, n) {
  transition <- cached(function(t) log_transition(n, t))
  forward_pass(events, n, function(log_density, t) same_grid_step(log_density, transition(t)))
}
fine_pass <- function(events) {
  kernel <- cached(fine_kernel)
  forward_pass(events, FINE_POINTS, function(log_density, t) fine_step(log_density, kernel(t)))
}

# The patterns of evidence. Events from offsets of time and items, with outcomes drawn from a true θ that follows
# `truth(offsets)`.
simulate_outcomes <- function(offsets, a, spread, truth) {
  theta <- truth(offsets)
  b <- theta + rnorm(length(offsets), 0, spread)
  outcome <- as.numeric(runif(length(offsets)) < plogis(a * (theta - b)))
  data.frame(time = START + offsets, a = a, b = b, outcome = outcome)
}
# A true θ that walks with the drift, at the given offsets.
walking_truth <- function(start) {
  function(offsets) start + cumsum(rnorm(length(offsets), 0, sqrt(VARIANCE * diff(c(0, offsets)) / PER)))
}
# Ordinary practice: 5 items a week for 26 weeks; 3 a day for 16 weeks with a 40-item mock every 4 weeks; 30 a day for 8
# weeks, the items 1.5 minutes apart.
ordinary <- function(pattern, seed) {
  set.seed(20261101 + seed)
  offsets <- switch(pattern,
    weekly = (1:130) * WEEK / 5,
    spaced = sort(c(
      as.vector(outer(c(9, 13, 19) * HOUR, (0:111) * DAY, `+`)),
      as.vector(outer((0:39) * 1.5 * MINUTE + 15 * HOUR, (c(27, 55, 83, 111)) * DAY, `+`))
    )),
    massed = as.vector(outer((0:29) * 1.5 * MINUTE + 18 * HOUR, (0:55) * DAY, `+`))
  )
  a <- runif(length(offsets), 0.8, 1.6)
  simulate_outcomes(offsets, a, 0.5, walking_truth(-1))
}
# A true rise of 1.5 over 14 days of 40 items a day, a minute apart, at one discrimination.
steady <- function(a, seed) {
  set.seed(20261111 + seed)
  offsets <- as.vector(outer((0:39) * MINUTE + 17 * HOUR, (0:13) * DAY, `+`))
  simulate_outcomes(offsets, a, 0.5, function(offsets) -0.75 + 1.5 * offsets / (14 * DAY))
}
# A learner who fails easy items, b from -3 to -1, then answers hard ones, b from 1 to 3, correctly: `burst` puts the
# failures at one time, a gap before the first correct answer.
reversal <- function(failed, correct, a, gap, burst, seed) {
  set.seed(20261121 + seed)
  failures <- if (burst) rep(0, failed) else (0:(failed - 1)) * gap
  offsets <- c(failures, max(failures) + (1:correct) * gap)
  data.frame(time = START + offsets, a = a, b = c(runif(failed, -3, -1), runif(correct, 1, 3)),
             outcome = rep(c(0, 1), c(failed, correct)))
}

# A bootstrap particle filter of the continuous model, without a grid: θ drawn from the prior, moved by the drift's
# normal increments, weighted by each outcome and resampled systematically when the effective sample size falls below
# half. Returns the weighted mean and SD of θ after the last event.
particle_filter <- function(events, particles, seed) {
  set.seed(seed)
  theta <- rnorm(particles)
  log_weight <- numeric(particles)
  last <- NA
  for (e in seq_len(nrow(events))) {
    event <- events[e, ]
    gap <- if (is.na(last)) 0 else max(0, event$time - last)
    last <- max(last, event$time, na.rm = TRUE)
    if (gap > 0) theta <- theta + rnorm(particles, 0, sqrt(VARIANCE * gap / PER))
    log_weight <- log_weight + log_probability(theta, event$a, event$b, event$outcome)
    w <- exp(log_weight - max(log_weight))
    if (sum(w)^2 / sum(w^2) < particles / 2) {
      theta <- theta[findInterval((runif(1) + 0:(particles - 1)) / particles, cumsum(w) / sum(w)) + 1]
      log_weight <- numeric(particles)
    }
  }
  w <- exp(log_weight - max(log_weight))
  w <- w / sum(w)
  mean <- sum(w * theta)
  c(mean = mean, sd = sqrt(sum(w * (theta - mean)^2)))
}

if (selected("drift")) {
  # The patterns at 3 seeds, the steady rise at three discriminations, and reversals at the discriminations and gaps of
  # the design's tables, deep tails included.
  cases <- c(
    unlist(lapply(c("weekly", "spaced", "massed"), function(pattern) {
      lapply(1:3, function(seed) {
        list(name = sprintf("%s-%d", pattern, seed), kind = "ordinary", events = ordinary(pattern, seed))
      })
    }), recursive = FALSE),
    unlist(lapply(c(1.2, 2, 3), function(a) {
      lapply(1:3, function(seed) {
        list(name = sprintf("steady-%g-%d", a, seed), kind = "steady", events = steady(a, seed))
      })
    }), recursive = FALSE),
    lapply(list(
      list("reversal-60-120", 60, 120, 1.2, MINUTE, FALSE),
      list("reversal-90-200", 90, 200, 2, MINUTE, FALSE),
      list("burst-80-160-a2.5", 80, 160, 2.5, MINUTE, TRUE),
      list("burst-80-160-a3", 80, 160, 3, MINUTE, TRUE),
      list("reversal-200-400-hourly", 200, 400, 1.5, HOUR, FALSE),
      list("burst-120-300-daily", 120, 300, 2, DAY, TRUE),
      list("reversal-300-600", 300, 600, 3, MINUTE, FALSE),
      list("reversal-150-300", 150, 300, 3, MINUTE, FALSE)
    ), function(x) list(name = x[[1]], kind = "reversal", events = reversal(x[[2]], x[[3]], x[[4]], x[[5]], x[[6]], 1)))
  )
  for (t in c(1e-4, 0.05, 1, 8, 73, 512)) check_kernel(t)

  POINTS <- 241
  CHECKPOINTS <- 10
  results <- lapply(cases, function(case) {
    events <- case$events
    count <- nrow(events)
    checkpoints <- unique(round(seq(count / CHECKPOINTS, count, length.out = CHECKPOINTS)))
    events$checkpoint <- seq_len(count) %in% checkpoints
    same <- same_grid_pass(events, POINTS)
    fine <- fine_pass(events)
    message(sprintf("%-24s same grid %.4f, depth %4.0f; fine %.4f, SD %.4f", case$name, tail(same$mean, 1), same$depth,
                    tail(fine$mean, 1), tail(fine$sd, 1)))
    list(name = case$name, kind = case$kind,
         events = list(time = events$time, a = events$a, b = events$b, outcome = events$outcome),
         checkpoints = checkpoints - 1,
         sameGrid = same[c("mean", "sd", "logLikelihood", "depth")], fine = fine[c("mean", "sd", "mode")])
  })

  # The particle filter, as 10 filters of 10^5 particles, whose spread gives the Monte Carlo error of their mean.
  filtered <- cases[[1]]
  filters <- sapply(1:10, function(k) particle_filter(filtered$events, 1e5, 20261131 + k))
  particle <- list(case = filtered$name, particles = 1e6, mean = mean(filters["mean", ]), sd = mean(filters["sd", ]),
                   meanError = sd(filters["mean", ]) / sqrt(10))
  fine_mean <- tail(results[[1]]$fine$mean, 1)
  message(sprintf("particle filter %.5f, Monte Carlo error %.5f; fine %.5f", particle$mean, particle$meanError,
                  fine_mean))
  stopifnot(abs(particle$mean - fine_mean) < 3 * particle$meanError)

  write_fixture("irt-drift", list(
    drift = list(variance = VARIANCE, per = PER),
    points = POINTS,
    finePoints = FINE_POINTS,
    cases = results,
    particleFilter = particle
  ))
}
