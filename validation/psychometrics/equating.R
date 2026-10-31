# Linking and equating fixtures. References: equateIRT (direc: mean-mean, mean-sigma, Haebara and Stocking-Lord
# linking of dichotomous items; score: true-score and observed-score equating), equate (linear, equipercentile and
# anchor-test equating, bootstrap), mirt (the covariance of a concurrent calibration), numDeriv (Jacobians), and base R
# written out below: the linking criteria for mixed-format items, true-score equating by uniroot, and the equating
# methods that no package implements for the designs used.

source("validation/psychometrics/common.R")
suppressPackageStartupMessages({
  library(equateIRT)
  library(equate)
  library(mirt)
  library(numDeriv)
})

SEED <- 20261029
METHODS <- c(meanMean = "mean-mean", meanSigma = "mean-sigma", haebara = "Haebara", stockingLord = "Stocking-Lord")
POINTS <- 30 # equateIRT's quadrature: 30 Gauss-Hermite points for N(0, 1)
data(est2pl, est3pl, package = "equateIRT")
data(ACTmath, KBneat, package = "equate")

# --- Dichotomous linking: equateIRT ---------------------------------------------------------------------------------
# Forms 1 and 2 of equateIRT's est2pl and est3pl share items I1 to I10. Form 1 is linked to the scale of form 2.

form_items <- function(mods, form, items, model) {
  coef <- mods[[form]]$coef
  lapply(items, function(i) {
    get <- function(name, default) if (paste0(name, ".", i) %in% names(coef)) coef[[paste0(name, ".", i)]] else default
    list(model = model, parameters = list(a = get("Dscrmn", 1), b = get("Dffclt", NA), c = get("Gussng", 0), d = 1))
  })
}

linking_case <- function(name, estimates, model) {
  mods <- modIRT(coef = estimates$coef, var = estimates$var, display = FALSE)
  common <- paste0("I", 1:10)
  transformations <- lapply(METHODS, function(method) {
    result <- direc(mods = mods, which = c(1, 2), method = method, nq = POINTS)
    list(slope = result$A, intercept = result$B)
  })
  list(name = name, from = form_items(mods, 1, common, model), to = form_items(mods, 2, common, model),
       transformations = transformations)
}

# --- Mixed-format linking: base R -----------------------------------------------------------------------------------
# Six 3PL, two GRM and two GPCM items on the `to` scale; their `from` estimates are the items on the scale
# θ_from = (θ_to - 0.4) / 1.25 with noise added to every parameter, so that the methods disagree.

category_probabilities <- function(item, theta) {
  p <- item$parameters
  # a (θ - b_k) for each theta (row) and threshold (column).
  steps <- function() matrix(outer(theta, p$b, function(t, b) p$a * (t - b)), nrow = length(theta))
  switch(item$model,
    "3pl" = {
      correct <- p$c + (1 - p$c) * plogis(p$a * (theta - p$b))
      cbind(1 - correct, correct)
    },
    grm = {
      above <- cbind(1, matrix(plogis(steps()), nrow = length(theta)), 0)
      above[, -ncol(above), drop = FALSE] - above[, -1, drop = FALSE]
    },
    gpcm = {
      logits <- cbind(0, t(apply(steps(), 1, cumsum)))
      exponentials <- exp(logits - apply(logits, 1, max))
      exponentials / rowSums(exponentials)
    }
  )
}

rescale <- function(item, A, B) {
  p <- item$parameters
  p$a <- p$a / A
  p$b <- A * p$b + B
  list(model = item$model, parameters = p)
}

set.seed(SEED)
mixed_to <- c(
  lapply(1:6, function(i) {
    list(model = "3pl", parameters = list(a = runif(1, 0.8, 2), b = rnorm(1), c = runif(1, 0.1, 0.25), d = 1))
  }),
  lapply(1:2, function(i) list(model = "grm", parameters = list(a = runif(1, 0.8, 2), b = sort(rnorm(3))))),
  lapply(1:2, function(i) list(model = "gpcm", parameters = list(a = runif(1, 0.8, 2), b = rnorm(3))))
)
mixed_from <- lapply(mixed_to, function(item) {
  rescaled <- rescale(item, 1 / 1.25, -0.4 / 1.25)
  p <- rescaled$parameters
  p$a <- p$a * exp(rnorm(1, 0, 0.1))
  p$b <- p$b + rnorm(length(p$b), 0, 0.1)
  if (item$model == "grm") p$b <- sort(p$b)
  list(model = item$model, parameters = p)
})

# measurekit's default quadrature: 61 points from -6 to 6 with normal density weights.
nodes <- seq(-6, 6, length.out = 61)
weights <- dnorm(nodes) / sum(dnorm(nodes))

# Haebara's criterion sums the squared differences of every category's probability; Stocking and Lord's, the squared
# difference of the expected total score. x is (ln A, B).
criterion <- function(x, type) {
  differences <- lapply(seq_along(mixed_to), function(j) {
    category_probabilities(mixed_to[[j]], nodes) -
      category_probabilities(rescale(mixed_from[[j]], exp(x[1]), x[2]), nodes)
  })
  if (type == "haebara") return(sum(weights * Reduce(`+`, lapply(differences, function(d) rowSums(d^2)))))
  scores <- Reduce(`+`, lapply(differences, function(d) d %*% (seq_len(ncol(d)) - 1)))
  sum(weights * scores^2)
}

minimize <- function(type) {
  fit <- nlminb(c(0, 0), criterion, type = type,
                control = list(rel.tol = 1e-15, x.tol = 1e-15, eval.max = 1e4, iter.max = 1e4))
  list(slope = exp(fit$par[1]), intercept = fit$par[2])
}

moments <- function(items) {
  list(slopes = sapply(items, function(item) item$parameters$a),
       locations = unlist(lapply(items, function(item) item$parameters$b)))
}
from_moments <- moments(mixed_from)
to_moments <- moments(mixed_to)
mean_mean <- mean(from_moments$slopes) / mean(to_moments$slopes)
mean_sigma <- sd(to_moments$locations) / sd(from_moments$locations)
moment_transformation <- function(A) {
  list(slope = A, intercept = mean(to_moments$locations) - A * mean(from_moments$locations))
}

mixed_case <- list(
  name = "mixed format",
  from = mixed_from,
  to = mixed_to,
  transformations = list(
    meanMean = moment_transformation(mean_mean),
    meanSigma = moment_transformation(mean_sigma),
    haebara = minimize("haebara"),
    stockingLord = minimize("stocking-lord")
  )
)

write_fixture("equating-linking", list(
  source = paste("equateIRT's est2pl and est3pl, form 1 to form 2 over items I1 to I10, with 30 Gauss-Hermite points;",
                 "mixed-format items simulated with set.seed(20261029), base R criteria with 61 rectangular points."),
  cases = list(linking_case("2PL", est2pl, "2pl"), linking_case("3PL", est3pl, "3pl"), mixed_case)
))

# --- True-score equating: base R ------------------------------------------------------------------------------------
# The true score of `from` solved for theta by uniroot, the true score of `to` there; below the sum of the lower
# asymptotes, the line from 0 to the lower limit of `to` (Kolen and Brennan).

true_score <- function(items, theta) {
  sum(sapply(items, function(item) {
    probabilities <- category_probabilities(item, theta)
    sum(probabilities * (seq_len(ncol(probabilities)) - 1))
  }))
}
lower_limit <- function(items) sum(sapply(items, function(item) if (item$model == "3pl") item$parameters$c else 0))
highest_score <- function(items) sum(sapply(items, function(item) length(item$parameters$b)))

equate_true_scores <- function(from, to) {
  sapply(0:highest_score(from), function(x) {
    if (x <= lower_limit(from)) return(x * lower_limit(to) / lower_limit(from))
    if (x >= highest_score(from)) return(highest_score(to))
    theta <- uniroot(function(t) true_score(from, t) - x, c(-30, 30), tol = 1e-14)$root
    true_score(to, theta)
  })
}

# Form 2 of est3pl, and form 1 on its scale by equateIRT's Haebara transformation. equateIRT's score() gives the same
# equivalents for the scores above the sum of the lower asymptotes.
mods3 <- modIRT(coef = est3pl$coef, var = est3pl$var, display = FALSE)
haebara3 <- direc(mods = mods3, which = c(1, 2), method = "Haebara", nq = POINTS)
form2 <- lapply(form_items(mods3, 2, sub("Dffclt.", "", grep("Dffclt", names(mods3[[2]]$coef), value = TRUE)), "3pl"),
                function(item) { item$model <- "3pl"; item })
form1 <- lapply(form_items(mods3, 1, sub("Dffclt.", "", grep("Dffclt", names(mods3[[1]]$coef), value = TRUE)), "3pl"),
                function(item) rescale(item, haebara3$A, haebara3$B))
three_pl <- equate_true_scores(form2, form1)
reference <- score(haebara3, method = "TSE", se = FALSE)
attainable <- reference$T2 + 1
stopifnot(max(abs(reference$T1.as.T2 - three_pl[attainable])) < 1e-8)

mixed_linked <- lapply(mixed_from, function(item) rescale(item, mixed_case$transformations$haebara$slope,
                                                          mixed_case$transformations$haebara$intercept))

write_fixture("equating-true-score", list(
  source = paste("Form 2 of equateIRT's est3pl and form 1 linked to it by Haebara; the mixed-format items of the",
                 "linking fixture."),
  cases = list(
    list(name = "3PL", from = form2, to = form1, equated = three_pl),
    list(name = "mixed format", from = mixed_to, to = mixed_linked,
         equated = equate_true_scores(mixed_to, mixed_linked))
  )
))

# --- Observed-score equating: equate --------------------------------------------------------------------------------
# Random groups: form X to the scale of form Y. ACTmath holds the frequencies; KBneat's total scores are tabulated.

# Kolen and Brennan's equipercentile equating written out: the Y score at each percentile rank, through the smallest Y
# score whose cumulative share exceeds the rank and the largest whose share is below it, averaged; the rank of an X
# score is F_X(x - 1) + f(x) / 2. equate rounds P and F_Y to 10 decimals first, which moves its results by up to about
# 1e-7.
inverse_ranks <- function(ranks, y) {
  cumulative <- cumsum(as.numeric(y)) / sum(y)
  share <- function(score) if (score < 0) 0 else cumulative[score + 1]
  highest <- length(cumulative) - 1
  sapply(ranks, function(p) {
    if (p == 0) return(-0.5)
    if (p == 1) return(highest + 0.5)
    upper <- min(which(cumulative > p)) - 1
    from_upper <- upper - 0.5 + (p - share(upper - 1)) / (share(upper) - share(upper - 1))
    below <- which(cumulative < p) - 1
    if (length(below) == 0) return(from_upper)
    lower <- max(below)
    (from_upper + lower + 0.5 + (p - share(lower)) / (share(lower + 1) - share(lower))) / 2
  })
}

equipercentile <- function(x, y) {
  frequencies <- as.numeric(x) / sum(x)
  inverse_ranks(cumsum(frequencies) - frequencies / 2, y)
}

observed_case <- function(name, x, y) {
  reference <- equate(x, y, type = "equipercentile", verbose = FALSE)
  exact <- equipercentile(x, y)
  stopifnot(max(abs(reference - exact)) < 1e-6)
  list(
    name = name,
    from = as.numeric(x),
    to = as.numeric(y),
    linear = equate(x, y, type = "linear", verbose = FALSE),
    equipercentile = reference,
    equipercentileUnrounded = exact
  )
}

write_fixture("equating-observed-score", list(
  source = "The equate package's ACTmath (40 items) and the total scores of KBneat (36 items).",
  cases = list(
    observed_case("ACTmath", as.freqtab(ACTmath[, 1:2]), as.freqtab(ACTmath[, c(1, 3)])),
    observed_case("KBneat", freqtab(KBneat$x$total, scales = 0:36), freqtab(KBneat$y$total, scales = 0:36))
  )
))

# Log-linear presmoothing: base R's Poisson glm on the powers of the score, iterated to a relative deviance change of
# 1e-15; equate's presmoothing, which runs glm at its default 1e-8, agrees within 1e-6.
presmoothing_case <- function(name, x, degrees) {
  counts <- as.numeric(x)
  score <- seq_along(counts) - 1
  list(name = name, distribution = counts, fits = lapply(degrees, function(degree) {
    powers <- sapply(seq_len(degree), function(k) score^k)
    fit <- glm(counts ~ powers, family = poisson, control = glm.control(epsilon = 1e-15, maxit = 100))
    smoothed <- as.numeric(presmoothing(x, smoothmethod = "loglinear", degrees = list(degree)))
    stopifnot(max(abs(smoothed - fitted(fit))) < 1e-6)
    list(degree = degree, distribution = unname(fitted(fit)))
  }))
}

write_fixture("equating-presmoothing", list(
  source = "The equate package's ACTmath (40 items) and the total scores of KBneat (36 items).",
  cases = list(
    presmoothing_case("ACTmath X", as.freqtab(ACTmath[, 1:2]), c(2, 3, 4, 6)),
    presmoothing_case("KBneat X", freqtab(KBneat$x$total, scales = 0:36), c(2, 3, 4, 6))
  )
))

# --- Nonequivalent groups with an anchor: equate --------------------------------------------------------------------
# Form X (group P) to form Y (group Q) through a common anchor V. Tucker, Levine and chained linear equating are closed
# forms, compared with equate to 1e-12. Chained equipercentile and frequency estimation are written out in base R from
# Kolen and Brennan's definitions and compared with equate to 1e-6, which rounds percentile ranks to 10 decimals.
# equate's frequency estimation weighs the two components of each synthetic distribution by the group sizes as well as
# by ws, so ws = 1/2 gives the weights w1 = n1 / (n1 + n2) of the definition; the case passes the ws that reproduces
# w1 = `weight`: ws n1 : (1 - ws) n2 = w1 : w2.

as_matrix <- function(x) matrix(as.numeric(x), nrow = dim(x)[1])

# The synthetic distributions of frequency estimation by the definition, f(x) = w1 f1(x) + w2 sum_v f1(x | v) h2(v) and
# g(y) = w1 sum_v g2(y | v) h1(v) + w2 g2(y), from the matrices of counts by total, then anchor score.
synthetic_distributions <- function(x, y, w1) {
  f1 <- x / sum(x)
  g2 <- y / sum(y)
  h1 <- colSums(f1)
  h2 <- colSums(g2)
  list(from = w1 * rowSums(f1) + (1 - w1) * as.vector(sweep(f1, 2, h1, "/") %*% h2),
       to = w1 * as.vector(sweep(g2, 2, h2, "/") %*% h1) + (1 - w1) * rowSums(g2))
}

frequency_estimation <- function(x, y, w1) {
  synthetic <- synthetic_distributions(x, y, w1)
  equipercentile(synthetic$from, synthetic$to)
}

# Chained equipercentile equating: X to V in P, the percentile rank of that anchor equivalent in the continuized anchor
# distribution of Q (linear between the cumulative shares at the ends of its score's interval), then back to Y in Q.
chained_equipercentile <- function(x, y) {
  anchor_equivalent <- equipercentile(rowSums(x), colSums(x))
  cumulative <- cumsum(colSums(y)) / sum(y)
  share <- function(score) if (score < 0) 0 else if (score >= length(cumulative)) 1 else cumulative[score + 1]
  ranks <- sapply(anchor_equivalent, function(v) {
    score <- floor(v + 0.5)
    share(score - 1) + (v - (score - 0.5)) * (share(score) - share(score - 1))
  })
  inverse_ranks(ranks, rowSums(y))
}

neat_case <- function(name, x, y, anchor, smoothed_degrees) {
  internal <- anchor == "internal"
  n1 <- sum(x)
  n2 <- sum(y)
  other_weight <- 0.3
  linear_entry <- function(method, weight = NA, true_score = FALSE) {
    ws <- if (is.na(weight)) -1 else weight
    list(method = method, weight = weight, trueScore = true_score,
         equated = equate(x, y, type = "linear", method = method, ws = ws, internal = internal, lts = true_score,
                          verbose = FALSE))
  }
  frequency_entry <- function(w1) {
    ws <- w1 * n2 / (w1 * n2 + (1 - w1) * n1)
    reference <- equate(x, y, type = "equipercentile", method = "frequency estimation", ws = ws, verbose = FALSE)
    exact <- frequency_estimation(as_matrix(x), as_matrix(y), w1)
    stopifnot(max(abs(reference - exact)) < 1e-6)
    list(method = "frequency estimation", weight = w1, equate = reference, exact = exact)
  }
  chained <- equate(x, y, type = "equipercentile", method = "chained", verbose = FALSE)
  chained_exact <- chained_equipercentile(as_matrix(x), as_matrix(y))
  stopifnot(max(abs(chained - chained_exact)) < 1e-6)
  list(
    name = name,
    anchor = anchor,
    share = n1 / (n1 + n2),
    from = as_matrix(x),
    to = as_matrix(y),
    linear = list(
      linear_entry("tucker"), linear_entry("tucker", other_weight),
      linear_entry("levine"), linear_entry("levine", other_weight), linear_entry("levine", true_score = TRUE),
      linear_entry("chained")
    ),
    equipercentile = list(
      list(method = "chained", weight = NA, equate = chained, exact = chained_exact),
      frequency_entry(n1 / (n1 + n2)), frequency_entry(other_weight)
    )
  )
}

# Bivariate log-linear presmoothing: base R's Poisson glm on the powers of the total and anchor scores and their cross
# products, scaled to [0, 1] for conditioning, iterated to a relative deviance change of 1e-15; equate's presmoothing
# agrees within 1e-6. `degrees` is c(total, anchor) and `cross` the highest power in the cross products.
presmoothing_fit <- function(x, degrees, cross) {
  counts <- as.numeric(x)
  cells <- expand.grid(total = seq_len(dim(x)[1]) - 1, anchor = seq_len(dim(x)[2]) - 1)
  total <- cells$total / max(cells$total)
  anchor <- cells$anchor / max(cells$anchor)
  terms <- c(lapply(seq_len(degrees[1]), function(i) total^i), lapply(seq_len(degrees[2]), function(j) anchor^j))
  if (cross > 0) {
    for (i in seq_len(cross)) for (j in seq_len(cross)) terms[[length(terms) + 1]] <- total^i * anchor^j
  }
  design <- do.call(cbind, terms)
  fit <- suppressWarnings(glm(counts ~ design, family = poisson, control = glm.control(epsilon = 1e-15, maxit = 100)))
  smoothed <- suppressWarnings(as.numeric(presmoothing(x, smoothmethod = "loglinear",
                                                       degrees = list(degrees, cross))))
  stopifnot(max(abs(smoothed - fitted(fit))) < 1e-6)
  list(degrees = degrees, crossDegree = cross,
       distribution = matrix(unname(fitted(fit)), nrow = dim(x)[1]))
}

# KBneat: forms X and Y of 36 items with an internal anchor of 12 items, 1655 and 1638 persons.
kb_x <- freqtab(KBneat$x, scales = list(0:36, 0:12))
kb_y <- freqtab(KBneat$y, scales = list(0:36, 0:12))

# A simulation with an external anchor: two 20-item forms and a 10-item anchor under a 2PL; group P, 700 persons, theta
# ~ N(0, 1) took form X; group Q, 500 persons, theta ~ N(0.4, 1) took form Y; both took the anchor.
set.seed(SEED)
two_pl_score <- function(theta, a, b) {
  rowSums(sapply(seq_along(a), function(j) rbinom(length(theta), 1, plogis(a[j] * (theta - b[j])))))
}
a_x <- runif(20, 0.7, 1.5)
b_x <- rnorm(20)
a_y <- runif(20, 0.7, 1.5)
b_y <- rnorm(20, 0.3)
a_v <- runif(10, 0.7, 1.5)
b_v <- rnorm(10, 0, 0.8)
theta_p <- rnorm(700)
theta_q <- rnorm(500, 0.4)
sim_x <- freqtab(data.frame(total = two_pl_score(theta_p, a_x, b_x), anchor = two_pl_score(theta_p, a_v, b_v)),
                 scales = list(0:20, 0:10))
sim_y <- freqtab(data.frame(total = two_pl_score(theta_q, a_y, b_y), anchor = two_pl_score(theta_q, a_v, b_v)),
                 scales = list(0:20, 0:10))
stopifnot(all((colSums(as_matrix(sim_x)) > 0) == (colSums(as_matrix(sim_y)) > 0)))

write_fixture("equating-neat", list(
  source = paste("equate's KBneat (internal anchor) and a 2PL simulation with an external anchor,",
                 "set.seed(20261029): equate's linear, chained and frequency estimation equating, and base R",
                 "written out for the equipercentile methods."),
  cases = list(neat_case("KBneat", kb_x, kb_y, "internal"), neat_case("simulation", sim_x, sim_y, "external"))
))

write_fixture("equating-bivariate-presmoothing", list(
  source = "The bivariate frequencies of KBneat's X and the simulation's X, as the equating-neat fixture.",
  cases = list(
    list(name = "KBneat X", distribution = as_matrix(kb_x),
         fits = list(presmoothing_fit(kb_x, c(4, 3), 1), presmoothing_fit(kb_x, c(3, 3), 2),
                     presmoothing_fit(kb_x, c(2, 2), 0))),
    list(name = "simulation X", distribution = as_matrix(sim_x),
         fits = list(presmoothing_fit(sim_x, c(3, 3), 1), presmoothing_fit(sim_x, c(4, 2), 2)))
  )
))

# --- Bootstrap standard errors: equate -----------------------------------------------------------------------------
# equate's bootstrap draws as many persons as the sample has, with replacement, from the cells in proportion to their
# counts, and reports the standard deviation of the equated scores over 2000 replications. The Monte Carlo error of
# each standard error is about 1.6%. The NEAT case smooths each replication's distributions before frequency estimation,
# as a resample can leave an anchor score without persons in one group; ws = 1/2 gives equate's frequency estimation
# the weights of the definition at the groups' sizes.

REPLICATIONS <- 2000

bootstrap_errors <- function(x, y, args) {
  set.seed(SEED)
  suppressWarnings(bootstrap(x, y, reps = REPLICATIONS, args = args)$se)
}

neat_args <- list(
  tucker = list(type = "linear", method = "tucker"),
  levine = list(type = "linear", method = "levine"),
  chainedLinear = list(type = "linear", method = "chained"),
  chainedEquipercentile = list(type = "equipercentile", method = "chained"),
  frequencyEstimation = list(type = "equipercentile", method = "frequency estimation", ws = 0.5,
                             smoothmethod = "loglinear", degrees = list(c(4, 3), 1))
)
neat_errors <- bootstrap_errors(kb_x, kb_y, neat_args)

act_x <- as.freqtab(ACTmath[, 1:2])
act_y <- as.freqtab(ACTmath[, c(1, 3)])
random_groups_errors <- bootstrap_errors(act_x, act_y, list(
  linear = list(type = "linear"),
  equipercentile = list(type = "equipercentile")
))

# A single group of 800 persons took forms X and Y of 20 items under a 2PL.
set.seed(SEED)
theta_sg <- rnorm(800)
joint <- freqtab(data.frame(x = two_pl_score(theta_sg, a_x, b_x), y = two_pl_score(theta_sg, a_y, b_y)),
                 scales = list(0:20, 0:20))
set.seed(SEED)
single_group_errors <- suppressWarnings(bootstrap(joint, reps = REPLICATIONS, args = list(
  linear = list(type = "linear"),
  equipercentile = list(type = "equipercentile")
))$se)

as_lists <- function(errors) lapply(seq_len(ncol(errors)), function(j) unname(errors[, j])) |>
  setNames(colnames(errors))

write_fixture("equating-bootstrap", list(
  source = paste("equate's bootstrap, 2000 replications, set.seed(20261029): KBneat, ACTmath (random groups) and a",
                 "single-group 2PL simulation of 800 persons."),
  replications = REPLICATIONS,
  neat = list(from = as_matrix(kb_x), to = as_matrix(kb_y), errors = as_lists(neat_errors)),
  randomGroups = list(from = as.numeric(act_x), to = as.numeric(act_y), errors = as_lists(random_groups_errors)),
  singleGroup = list(joint = as_matrix(joint), errors = as_lists(single_group_errors))
))

# --- Robust z of common items: base R ---------------------------------------------------------------------------------
# Huynh and Meyer's robust z of the difference d in an item's mean location between two calibrations: (d - median d) /
# (0.74 IQR d), with R's quantiles of type 7, flagged beyond 1.645.

robust_z <- function(from, to) {
  location <- function(item) mean(item$parameters$b)
  d <- mapply(function(f, t) location(t) - location(f), from, to)
  z <- (d - median(d)) / (0.74 * IQR(d))
  list(from = from, to = to, differences = unname(d), z = unname(z), flagged = unname(abs(z) > 1.645))
}

# Fifteen Rasch items on two scales that differ by a shift of 0.3, with calibration noise, and drift of 0.9 in item 4 and
# -0.7 in item 11.
set.seed(SEED)
rasch_b <- rnorm(15)
drift <- replace(numeric(15), c(4, 11), c(0.9, -0.7))
rasch_item <- function(b) list(model = "2pl", parameters = list(a = 1, b = b, c = 0, d = 1))
rasch_from <- lapply(rasch_b, rasch_item)
rasch_to <- lapply(rasch_b + 0.3 + drift + rnorm(15, 0, 0.08), rasch_item)

# Form 1 and form 2 of equateIRT's est2pl over items I1 to I10, as the linking fixture.
est2pl_mods <- modIRT(coef = est2pl$coef, var = est2pl$var, display = FALSE)
est2pl_items <- paste0("I", 1:10)

write_fixture("equating-robust-z", list(
  source = paste("A Rasch simulation with drift, set.seed(20261029); equateIRT's est2pl, forms 1 to 2 over I1-I10;",
                 "the 10 mixed-format items of the linking fixture."),
  cases = list(
    c(list(name = "Rasch with drift"), robust_z(rasch_from, rasch_to)),
    c(list(name = "est2pl"), robust_z(form_items(est2pl_mods, 1, est2pl_items, "2pl"),
                                      form_items(est2pl_mods, 2, est2pl_items, "2pl"))),
    c(list(name = "mixed format"), robust_z(mixed_from, mixed_to))
  )
))

# --- IRT observed-score equating: base R and equateIRT -------------------------------------------------------------------
# Lord and Wingersky's recursion at each node of a population, the groups' score distributions mixed with weight w for
# the group of the form to equate, and the equipercentile equating of the two mixtures. equateIRT's score(method = "OSE")
# takes the populations N(0, 1) on the nodes theta and N(B, A^2) on A theta + B with the same weights; its results are
# compared with the base R ones for form 2 of est3pl and form 1 moved to its scale.

response_probabilities <- function(item, theta) {
  if (item$model == "2pl") item$model <- "3pl"
  category_probabilities(item, theta)
}

score_probabilities <- function(items, theta) {
  distribution <- 1
  for (item in items) {
    p <- as.vector(response_probabilities(item, theta))
    following <- numeric(length(distribution) + length(p) - 1)
    for (k in seq_along(p)) {
      positions <- k - 1 + seq_along(distribution)
      following[positions] <- following[positions] + distribution * p[k]
    }
    distribution <- following
  }
  distribution
}

population <- function(mean, sd) list(nodes = mean + sd * nodes, weights = weights)

form_distribution <- function(items, pop) {
  rowSums(mapply(function(theta, w) w * score_probabilities(items, theta), pop$nodes, pop$weights))
}

irt_observed_score <- function(from, to, populations, w) {
  mixture <- function(items) {
    w * form_distribution(items, populations$from) + (1 - w) * form_distribution(items, populations$to)
  }
  equipercentile(mixture(from), mixture(to))
}

ose_populations <- function(transformation) {
  list(from = population(0, 1), to = population(transformation$intercept, transformation$slope))
}

est3pl_populations <- ose_populations(list(slope = haebara3$A, intercept = haebara3$B))
ose_case <- function(name, from, to, populations, ws, reference = NULL) {
  list(name = name, from = from, to = to, populations = populations, results = lapply(ws, function(w) {
    equated <- irt_observed_score(from, to, populations, w)
    if (!is.null(reference)) {
      stopifnot(max(abs(equated - reference(w))) < 1e-8)
    }
    list(weight = w, equated = equated)
  }))
}

equateirt_ose <- function(w) {
  score(haebara3, method = "OSE", se = FALSE, theta = nodes, weights = weights, w = w)[[2]]
}

write_fixture("equating-irt-observed-score", list(
  source = paste("Form 2 of equateIRT's est3pl and form 1 moved to its scale by Haebara, with equateIRT's OSE as a",
                 "second reference; the mixed-format items of the linking fixture, from `to` to `from` moved by Haebara;",
                 "populations N(0, 1) and N(B, A^2) on 61 rectangular points."),
  cases = list(
    ose_case("3PL", form2, form1, est3pl_populations, c(0.5, 0.3), equateirt_ose),
    ose_case("mixed format", mixed_to, mixed_linked, ose_populations(mixed_case$transformations$haebara), c(0.5, 0.7))
  )
))

# --- Delta-method standard errors of IRT equating: base R and mirt ------------------------------------------------------
# The Jacobian of the equated scores with respect to the item parameters by numDeriv, times the covariance of the
# parameters: SE^2 = J V J'. A pool lists the items of both forms once, a common item once; the parameter vector is
# the estimated parameters of the pool's items in order, named as measurekit names them.

parameter_names <- function(item) {
  switch(item$model,
    "2pl" = c("a", "b"),
    "3pl" = c("a", "b", "c"),
    c("a", paste0("b", seq_along(item$parameters$b)))
  )
}
parameter_vector <- function(item) {
  p <- item$parameters
  switch(item$model, "2pl" = c(p$a, p$b), "3pl" = c(p$a, p$b, p$c), c(p$a, p$b))
}
with_vector <- function(item, v) {
  p <- item$parameters
  item$parameters <- switch(item$model,
    "2pl" = modifyList(p, list(a = v[1], b = v[2])),
    "3pl" = modifyList(p, list(a = v[1], b = v[2], c = v[3])),
    list(a = v[1], b = v[-1])
  )
  item
}

# The true-score equating of two forms of the pool, with the limit at 0 of a form without lower asymptotes taken as 0.
total_score <- function(items, theta) {
  sum(sapply(items, function(item) {
    probabilities <- response_probabilities(item, theta)
    sum(probabilities * (seq_len(ncol(probabilities)) - 1))
  }))
}
lower_asymptotes <- function(items) sum(sapply(items, function(item) if (item$model %in% c("3pl")) item$parameters$c else 0))
true_score_equating <- function(from, to) {
  highest <- highest_score(from)
  sapply(0:highest, function(x) {
    lower <- lower_asymptotes(from)
    if (x <= lower) return(if (lower > 0) x * lower_asymptotes(to) / lower else lower_asymptotes(to))
    if (x >= highest) return(highest_score(to))
    theta <- uniroot(function(t) total_score(from, t) - x, c(-30, 30), tol = 1e-14)$root
    total_score(to, theta)
  })
}

delta_case <- function(name, pool, from_ids, to_ids, covariance, populations, w) {
  ids <- sapply(pool, function(item) item$id)
  sizes <- sapply(pool, function(item) length(parameter_names(item)))
  starts <- cumsum(c(0, sizes))[seq_along(pool)]
  forms <- function(vector) {
    moved <- lapply(seq_along(pool), function(j) with_vector(pool[[j]], vector[starts[j] + seq_len(sizes[j])]))
    list(from = moved[match(from_ids, ids)], to = moved[match(to_ids, ids)])
  }
  start <- unlist(lapply(pool, parameter_vector))
  labels <- data.frame(item = rep(ids, sizes), name = unlist(lapply(pool, parameter_names)))
  stopifnot(nrow(covariance) == length(start))
  results <- lapply(c("true-score", "observed-score"), function(method) {
    equate <- function(vector) {
      f <- forms(vector)
      if (method == "true-score") true_score_equating(f$from, f$to)
      else irt_observed_score(f$from, f$to, populations, w)
    }
    jacobian <- numDeriv::jacobian(equate, start)
    list(method = method, equated = equate(start), standardErrors = sqrt(rowSums((jacobian %*% covariance) * jacobian)))
  })
  list(
    name = name,
    items = lapply(pool, function(item) c(list(id = item$id, model = item$model), parameters = list(item$parameters))),
    from = from_ids, to = to_ids,
    covariance = list(parameters = lapply(seq_len(nrow(labels)), function(k) as.list(labels[k, ])),
                      matrix = unname(covariance)),
    populations = populations, weight = w, results = results
  )
}

# A concurrent calibration of two 20-item 2PL forms that share 8 items: group P, 1500 persons, theta ~ N(0, 1), took
# form X; group Q, 1500 persons, theta ~ N(0.3, 1.1^2), took form Y. Both groups' items are one set of estimates, the
# group means and variances free, and mirt's covariance of the slopes and intercepts becomes that of a and b = -d / a1.
set.seed(SEED)
n_unique <- 12
n_common <- 8
pool_size <- 2 * n_unique + n_common
true_a <- runif(pool_size, 0.8, 1.8)
true_b <- rnorm(pool_size)
pool_ids <- c(paste0("X", seq_len(n_unique)), paste0("Y", seq_len(n_unique)), paste0("C", seq_len(n_common)))
x_columns <- c(seq_len(n_unique), 2 * n_unique + seq_len(n_common))
y_columns <- n_unique + c(seq_len(n_unique), n_unique + seq_len(n_common))
simulate_group <- function(theta, columns) {
  sapply(columns, function(j) rbinom(length(theta), 1, plogis(true_a[j] * (theta - true_b[j]))))
}
concurrent <- matrix(NA_integer_, 3000, pool_size, dimnames = list(NULL, pool_ids))
concurrent[1:1500, x_columns] <- simulate_group(rnorm(1500), x_columns)
concurrent[1501:3000, y_columns] <- simulate_group(rnorm(1500, 0.3, 1.1), y_columns)
calibration <- multipleGroup(concurrent, 1, group = rep(c("P", "Q"), each = 1500), itemtype = "2PL",
                             invariance = c("slopes", "intercepts", "free_means", "free_var"), SE = TRUE,
                             verbose = FALSE)
estimates <- mod2values(calibration)
estimates <- estimates[estimates$group == "P" & estimates$name %in% c("a1", "d"), ]
slopes <- estimates$value[estimates$name == "a1"]
intercepts <- estimates$value[estimates$name == "d"]
stopifnot(identical(as.character(estimates$item[estimates$name == "a1"]), pool_ids))

# The rows of vcov are named name.parnum.parnum; the first parnum of the first group gives the item, four parameters
# (a1, d, g, u) to each.
full <- vcov(calibration)
parnum <- as.integer(sapply(strsplit(rownames(full), ".", fixed = TRUE), function(parts) parts[2]))
item_rows <- which(substr(rownames(full), 1, 1) %in% c("a", "d"))
stopifnot(length(item_rows) == 2 * pool_size)
item_of_row <- (parnum[item_rows] - 1) %/% 4 + 1
is_slope <- substr(rownames(full)[item_rows], 1, 2) == "a1"
sub_covariance <- full[item_rows, item_rows]
# Parameters in the order of the pool, a then b for each item: b = -d / a1.
position <- function(item, slope) which(item_of_row == item & is_slope == slope)
transform <- matrix(0, 2 * pool_size, 2 * pool_size)
for (j in seq_len(pool_size)) {
  a1 <- slopes[j]
  d <- intercepts[j]
  transform[2 * j - 1, position(j, TRUE)] <- 1
  transform[2 * j, position(j, TRUE)] <- d / a1^2
  transform[2 * j, position(j, FALSE)] <- -1 / a1
}
concurrent_covariance <- transform %*% sub_covariance %*% t(transform)

values_q <- mod2values(calibration)
group_q <- c(values_q$value[values_q$group == "Q" & values_q$name == "MEAN_1"],
             values_q$value[values_q$group == "Q" & values_q$name == "COV_11"])
concurrent_pool <- lapply(seq_len(pool_size), function(j) {
  list(id = pool_ids[j], model = "2pl", parameters = list(a = slopes[j], b = -intercepts[j] / slopes[j], c = 0, d = 1))
})
concurrent_populations <- list(from = population(0, 1), to = population(group_q[1], sqrt(group_q[2])))

# A mixed-format pool of the linking fixture's items (3PL, GRM and GPCM), a form and its moved copy as the two forms,
# with a made-up positive definite covariance: the delta method does not depend on where the covariance comes from.
mixed_pool <- c(
  lapply(seq_along(mixed_to), function(j) c(list(id = paste0("from", j)), mixed_to[[j]])),
  lapply(seq_along(mixed_linked), function(j) c(list(id = paste0("to", j)), mixed_linked[[j]]))
)
mixed_size <- sum(sapply(mixed_pool, function(item) length(parameter_names(item))))
set.seed(SEED)
draws <- matrix(rnorm(mixed_size * (mixed_size + 4)), mixed_size)
mixed_covariance <- 0.0004 * tcrossprod(draws) / (mixed_size + 4) + diag(0.0004, mixed_size)
mixed_transformation <- mixed_case$transformations$haebara

write_fixture("equating-irt-standard-errors", list(
  source = paste("A concurrent mirt multipleGroup calibration of two 2PL forms sharing 8 items, set.seed(20261029):",
                 "1500 persons per group, covariance of a and b; the mixed-format items of the linking fixture with a",
                 "made-up covariance. numDeriv's Jacobian times the covariance."),
  cases = list(
    delta_case("concurrent 2PL", concurrent_pool, pool_ids[c(seq_len(n_unique), 2 * n_unique + seq_len(n_common))],
               pool_ids[c(n_unique + seq_len(n_unique), 2 * n_unique + seq_len(n_common))], concurrent_covariance,
               concurrent_populations, 0.5),
    delta_case("mixed format", mixed_pool, paste0("from", seq_along(mixed_to)), paste0("to", seq_along(mixed_linked)),
               mixed_covariance, ose_populations(mixed_transformation), 0.5)
  )
))
