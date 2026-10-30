# Linking and equating fixtures. References: equateIRT (direc: mean-mean, mean-sigma, Haebara and Stocking-Lord
# linking of dichotomous items; score: true-score equating), equate (linear and equipercentile equating), and base R
# written out below: the linking criteria for mixed-format items and true-score equating by uniroot.

source("validation/psychometrics/common.R")
suppressPackageStartupMessages({
  library(equateIRT)
  library(equate)
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

# the package's default quadrature: 61 points from -6 to 6 with normal density weights.
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

# Kolen and Brennan's equipercentile equating written out: the percentile rank P of each X score, F_Y(x - 1) + f(x) / 2,
# inverted through the smallest Y score whose cumulative share exceeds it and the largest whose share is below it,
# averaged. equate rounds P and F_Y to 10 decimals first, which moves its results by up to about 1e-7.
equipercentile <- function(x, y) {
  frequencies <- as.numeric(x) / sum(x)
  cumulative <- cumsum(as.numeric(y)) / sum(y)
  share <- function(score) if (score < 0) 0 else cumulative[score + 1]
  highest <- length(cumulative) - 1
  sapply(cumsum(frequencies) - frequencies / 2, function(p) {
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
