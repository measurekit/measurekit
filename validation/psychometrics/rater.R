# Rater agreement fixtures. References: irr (kappa2, kappam.fleiss, icc, agree) and psych (cohen.kappa, ICC) as a check;
# effsize's cohen.d for the standardized mean difference between two raters, checked against base R; base R's cor for the
# correlation between two raters.
# Data: irr's anxiety (20 subjects, 3 raters, ratings 1 to 6), recoded 0 to 5, and diagnoses (30 subjects, 6 raters,
# 5 categories), coded 0 to 4; anxiety again with two ratings missing.

source("validation/psychometrics/common.R")
suppressPackageStartupMessages({
  library(irr)
  library(psych)
})

data(anxiety, diagnoses, package = "irr")
anxiety <- as.matrix(anxiety) - 1
diagnoses <- sapply(diagnoses, function(x) as.integer(substr(as.character(x), 1, 1)) - 1)
sparse <- anxiety
sparse[3, 2] <- NA
sparse[11, 1] <- NA

icc_values <- function(x) {
  coefficient <- function(model, type, unit) icc(x, model = model, type = type, unit = unit)$value
  values <- list(
    oneWay = coefficient("oneway", "consistency", "single"),
    agreement = coefficient("twoway", "agreement", "single"),
    consistency = coefficient("twoway", "consistency", "single"),
    oneWayAverage = coefficient("oneway", "consistency", "average"),
    agreementAverage = coefficient("twoway", "agreement", "average"),
    consistencyAverage = coefficient("twoway", "consistency", "average")
  )
  reference <- suppressMessages(ICC(na.omit(x), lmer = FALSE))$results$ICC
  stopifnot(max(abs(unlist(values) - reference[c(1, 2, 3, 4, 5, 6)])) < 1e-12)
  values
}

kappa_values <- function(x) {
  values <- list(
    none = kappa2(x, "unweighted")$value,
    linear = kappa2(x, "equal")$value,
    quadratic = kappa2(x, "squared")$value
  )
  stopifnot(abs(values$none - cohen.kappa(na.omit(x))$kappa) < 1e-12,
            abs(values$quadratic - cohen.kappa(na.omit(x))$weighted.kappa) < 1e-12)
  values
}

ratings <- function(x) unname(lapply(seq_len(nrow(x)), function(i) as.numeric(x[i, ])))

write_fixture("rater", list(
  source = "irr's anxiety, recoded 0 to 5, also with two ratings missing; irr's diagnoses, coded 0 to 4.",
  cases = list(
    list(name = "anxiety", ratings = ratings(anxiety), cohen = kappa_values(anxiety[, 1:2]),
         fleiss = kappam.fleiss(anxiety)$value, icc = icc_values(anxiety)),
    list(name = "anxiety with missing ratings", ratings = ratings(sparse), cohen = kappa_values(sparse[, 1:2]),
         fleiss = kappam.fleiss(sparse)$value, icc = icc_values(sparse)),
    list(name = "diagnoses", ratings = ratings(diagnoses), cohen = kappa_values(diagnoses[, 1:2]),
         fleiss = kappam.fleiss(diagnoses)$value, icc = icc_values(diagnoses))
  )
))

# --- Many-facet Rasch model -----------------------------------------------------------------------------------------
# References: the marginal likelihood written out in base R, with theta = mu + sigma z on the 61 rectangular points of
# the package, maximized by BFGS and polished by Newton's method on numDeriv's gradient and Hessian, whose inverse gives
# the standard errors; and TAM's tam.mml.mfr (constraint = "items"), run on the same scores as a check. Everything else
# is base R written out at the maximum: person EAP, fair averages, element fit, separation and reliability.
# Data: sirt's data.ratings1 (135 students, 5 criteria, 7 raters, each student rated by 1 to 7 of them), TAM's
# data.sim.mfr (100 persons, 5 items, 5 raters and 3 topics, each person rated by one rater), and a simulation with an
# incomplete design.

suppressPackageStartupMessages({
  library(numDeriv)
  library(sirt)
  library(TAM)
})

SEED <- 20261029
source("validation/psychometrics/many-facet-likelihood.R")

# Measures and standard errors by the delta method, and the thresholds with theirs.
estimates <- function(design, fit) {
  par <- unpack(design, fit$phi)
  covariance <- solve(fit$information)
  p <- ncol(design$X)
  at <- 0
  facets <- lapply(seq_along(design$facets), function(f) {
    contrast <- design$contrasts[[f]]
    index <- at + seq_len(ncol(contrast))
    at <<- at + ncol(contrast)
    list(facet = design$facets[f], ids = design$elements[[f]], measure = drop(contrast %*% par$beta[index]),
         standardError = sqrt(diag(contrast %*% covariance[index, index, drop = FALSE] %*% t(contrast))))
  })
  at <- p
  thresholds <- lapply(seq_along(design$thresholds), function(s) {
    contrast <- rbind(diag(design$thresholds[s] - 1), -1)
    index <- at + seq_len(design$thresholds[s] - 1)
    at <<- at + design$thresholds[s] - 1
    list(element = design$sets[s], thresholds = par$thresholds[[s]],
         standardErrors = sqrt(diag(contrast %*% covariance[index, index, drop = FALSE] %*% t(contrast))))
  })
  list(facets = facets, thresholds = thresholds, mean = par$mu, sd = par$sigma)
}

# P(category) of a partial credit item with location `location` and thresholds `tau` at theta, a = 1.
category_probabilities <- function(theta, location, tau) {
  logits <- c(0, cumsum(theta - location - tau))
  weights <- exp(logits - max(logits))
  weights / sum(weights)
}

# The expected rating at theta with the elements `levels`, a character vector with an NA for each facet at its mean;
# with the threshold facet at NA, the expected ratings under the thresholds of each of its elements are averaged.
expected_rating <- function(est, threshold_facet, theta, levels) {
  names <- sapply(est$facets, `[[`, "facet")
  location <- sum(sapply(seq_along(levels), function(f) {
    if (is.na(levels[f])) 0 else est$facets[[f]]$measure[match(levels[f], est$facets[[f]]$ids)]
  }))
  owner <- if (is.null(threshold_facet)) NA_character_ else levels[match(threshold_facet, names)]
  keep <- if (is.na(owner)) rep(TRUE, length(est$thresholds)) else sapply(est$thresholds, function(s) identical(s$element, owner))
  mean(sapply(est$thresholds[keep], function(s) {
    sum(0:length(s$thresholds) * category_probabilities(theta, location, s$thresholds))
  }))
}

# EAP estimates of the persons under N(mu, sigma^2) on the rule: the posterior mean and SD of theta.
person_estimates <- function(design, fit) {
  par <- unpack(design, fit$phi)
  theta <- par$mu + par$sigma * Z
  joint <- sweep(person_node_loglik(design, fit$phi), 2, LOG_WEIGHTS, "+")
  posterior <- exp(joint - apply(joint, 1, max))
  posterior <- posterior / rowSums(posterior)
  mean_theta <- drop(posterior %*% theta)
  list(id = unique(design$long$person), theta = mean_theta,
       standardError = sqrt(rowSums(posterior * outer(mean_theta, theta, "-")^2)))
}

# Infit and outfit mean squares and their Wilson-Hilferty standardized forms (Wright and Masters, 1982) of every
# element of every facet, from the residuals of its scores at the persons' estimates.
element_fit <- function(design, fit, persons) {
  par <- unpack(design, fit$phi)
  location <- drop(design$X %*% par$beta)
  moments <- t(sapply(seq_len(nrow(design$long)), function(i) {
    c <- design$virtual[i]
    tau <- par$thresholds[[design$set_of[c]]]
    p <- category_probabilities(persons$theta[design$person[i]], location[c], tau)
    k <- seq_along(p) - 1
    mean <- sum(k * p)
    c(design$long$score[i] - mean, sum((k - mean)^2 * p), sum((k - mean)^4 * p))
  }))
  standardized <- function(ms, q) (ms^(1 / 3) - 1) * (3 / q) + q / 3
  result <- list()
  for (f in seq_along(design$facets)) {
    codes <- match(as.character(design$long[[design$facets[f]]]), design$elements[[f]])
    for (e in seq_along(design$elements[[f]])) {
      rows <- moments[codes == e, , drop = FALSE]
      residual <- rows[, 1]; variance <- rows[, 2]; fourth <- rows[, 3]
      n <- nrow(rows)
      outfit <- mean(residual^2 / variance)
      infit <- sum(residual^2) / sum(variance)
      outfit_q <- sqrt(sum(fourth / variance^2) / n^2 - 1 / n)
      infit_q <- sqrt(sum(fourth - variance^2) / sum(variance)^2)
      result[[length(result) + 1]] <- list(facet = design$facets[f], id = design$elements[[f]][e],
        infitMeanSquare = infit, outfitMeanSquare = outfit, infitStandardized = standardized(infit, infit_q),
        outfitStandardized = standardized(outfit, outfit_q))
    }
  }
  result
}

# Separation, reliability and the homogeneity chi-square of a facet's measures with their standard errors.
facet_summary <- function(measure, error) {
  observed <- mean((measure - mean(measure))^2)
  mean_square_error <- mean(error^2)
  true_variance <- max(observed - mean_square_error, 0)
  weights <- 1 / error^2
  chi_square <- max(sum(weights * measure^2) - sum(weights * measure)^2 / sum(weights), 0)
  list(separation = sqrt(true_variance / mean_square_error), reliability = true_variance / observed,
       homogeneity = list(chiSquare = chi_square, degreesOfFreedom = length(measure) - 1,
                          significance = pchisq(chi_square, length(measure) - 1, lower.tail = FALSE)))
}

# TAM on the same scores: `wide` has a row per person and rater and a column per item, `facets` the rater of each row.
tam_check <- function(wide, facets, pid, formula, design, est, fit) {
  model <- tam.mml.mfr(resp = wide, facets = facets, pid = pid, formulaA = formula, constraint = "items",
                       control = list(nodes = Z, conv = 1e-10, convD = 1e-10, maxiter = 3000), verbose = FALSE)
  table <- model$xsi.facets
  differences <- unlist(lapply(est$facets, function(facet) {
    rows <- table[table$facet == facet$facet, ]
    names <- if (facet$facet == "item") rows$parameter else sub(paste0("^", facet$facet), "", rows$parameter)
    rows$xsi[match(facet$ids, names)] - facet$measure
  }))
  steps <- lapply(est$thresholds, function(set) {
    rows <- table[table$facet == (if (is.na(set$element)) "step" else paste0(design$threshold_facet, ":step")), ]
    if (!is.na(set$element)) rows <- rows[startsWith(rows$parameter, paste0(set$element, ":")), ]
    rows$xsi[seq_along(set$thresholds)] - set$thresholds
  })
  list(logLikelihood = -model$deviance / 2, mean = drop(model$beta), sd = sqrt(drop(model$variance)),
       iterations = model$iter, converged = model$iter < 3000,
       measureDifference = max(abs(differences)), thresholdDifference = max(abs(unlist(steps))),
       logLikelihoodDifference = -model$deviance / 2 - fit$logLikelihood,
       sdDifference = sqrt(drop(model$variance)) - est$sd, meanDifference = drop(model$beta) - est$mean)
}

# One model of a dataset: the base R maximum with its summaries, fit and fair averages, and TAM's check.
mfrm_case <- function(long, facets, threshold_facet, tam) {
  design <- facet_design(long, facets, threshold_facet)
  fit <- maximize(design)
  est <- estimates(design, fit)
  persons <- person_estimates(design, fit)
  summaries <- lapply(seq_along(facets), function(f) {
    codes <- match(as.character(long[[facets[f]]]), design$elements[[f]])
    facet <- est$facets[[f]]
    average <- function(e) {
      levels <- rep(NA_character_, length(facets))
      levels[f] <- facet$ids[e]
      expected_rating(est, threshold_facet, est$mean, levels)
    }
    c(list(facet = facets[f], ids = facet$ids, measures = facet$measure, standardErrors = facet$standardError,
           counts = as.numeric(table(factor(codes, seq_along(facet$ids)))),
           observedAverages = as.numeric(tapply(long$score, factor(codes, seq_along(facet$ids)), mean)),
           fairAverages = sapply(seq_along(facet$ids), average)),
      if (length(facet$ids) > 1) facet_summary(facet$measure, facet$standardError))
  })
  probes <- lapply(c(-1, 0, 1.5), function(theta) {
    levels <- rep(NA_character_, length(facets))
    levels[1] <- est$facets[[1]]$ids[1]
    levels[length(levels)] <- est$facets[[length(levels)]]$ids[2]
    list(theta = theta, levels = setNames(as.list(levels[!is.na(levels)]), facets[!is.na(levels)]),
         value = expected_rating(est, threshold_facet, theta, levels),
         fair = expected_rating(est, threshold_facet, theta, rep(NA_character_, length(facets))))
  })
  message(paste(facets, collapse = " + "), if (is.null(threshold_facet)) "" else paste0(" (thresholds by ", threshold_facet, ")"),
          ": ", design$persons, " persons, ", nrow(long), " scores, log-likelihood ", round(fit$logLikelihood, 6))
  # TAM 4.3-25 cannot build the design for thresholds by a facet of more than three elements other than the items.
  tam_result <- tryCatch(do.call(tam_check, c(tam, list(design = design, est = est, fit = fit))), error = function(e) {
    message("  TAM failed: ", conditionMessage(e))
    NULL
  })
  if (!is.null(tam_result)) {
    message("  TAM: ", tam_result$iterations, " iterations; differences from the maximum: log-likelihood ",
            signif(tam_result$logLikelihoodDifference, 3), ", mean ", signif(tam_result$meanDifference, 3), ", SD ",
            signif(tam_result$sdDifference, 3), ", measures ", signif(tam_result$measureDifference, 3), ", thresholds ",
            signif(tam_result$thresholdDifference, 3))
  }
  list(
    facets = facets, thresholdFacet = if (is.null(threshold_facet)) NULL else threshold_facet,
    summaries = summaries,
    thresholds = lapply(est$thresholds, function(s) list(element = if (is.na(s$element)) NULL else s$element,
                                                         thresholds = s$thresholds, standardErrors = s$standardErrors)),
    mean = est$mean, sd = est$sd, logLikelihood = fit$logLikelihood, parameters = fit$size,
    persons = persons, fit = element_fit(design, fit, persons), expectedRatings = probes,
    tam = tam_result
  )
}

# Scores in long format from a wide matrix with a row per person and rater, row by row.
long_scores <- function(wide, person, facets) {
  items <- ncol(wide)
  data.frame(person = rep(as.character(person), each = items), item = rep(colnames(wide), times = nrow(wide)),
             lapply(facets, function(f) rep(as.character(f), each = items)), score = as.vector(t(wide)),
             stringsAsFactors = FALSE)
}

# Dataset 1: sirt's data.ratings1.
data(data.ratings1)
ratings1_wide <- as.matrix(data.ratings1[, paste0("k", 1:5)])
ratings1 <- long_scores(ratings1_wide, data.ratings1$idstud, list(rater = data.ratings1$rater))
ratings1_tam <- function(formula) {
  list(wide = data.ratings1[, paste0("k", 1:5)], facets = data.frame(rater = as.character(data.ratings1$rater)),
       pid = data.ratings1$idstud, formula = formula)
}

# Dataset 2: TAM's data.sim.mfr, a person per row, with its rater and topic.
data(data.sim.mfr, data.sim.facets)
sim_wide <- data.sim.mfr
colnames(sim_wide) <- paste0("i", 1:5)
sim_ids <- paste0("p", seq_len(nrow(sim_wide)))
sim_facets <- list(rater = paste0("r", data.sim.facets$rater), topic = paste0("t", data.sim.facets$topic))
tam_sim <- long_scores(sim_wide, sim_ids, sim_facets)
tam_sim_tam <- function(formula) {
  list(wide = as.data.frame(sim_wide), facets = data.frame(sim_facets), pid = sim_ids, formula = formula)
}

# Dataset 3: 300 persons, 4 items, 6 raters, 4 categories; each person is rated on every item by 2 raters of 6, and the
# raters use the scale differently: their thresholds are the common ones times 0.8, 0.9, 1, 1.1, 1.2 and 1.
set.seed(SEED)
sim_severity <- c(-0.6, -0.3, 0, 0.1, 0.3, 0.5)
sim_difficulty <- c(-0.8, -0.2, 0.3, 0.7)
sim_tau <- c(-1.2, 0.1, 1.1)
sim_spread <- c(0.8, 0.9, 1, 1.1, 1.2, 1)
sim_theta <- rnorm(300, 0.3, 1.2)
incomplete_rows <- list()
for (person in seq_along(sim_theta)) {
  for (rater in sample(6, 2)) {
    scores <- sapply(sim_difficulty, function(difficulty) {
      probabilities <- category_probabilities(sim_theta[person], difficulty + sim_severity[rater], sim_tau * sim_spread[rater])
      sample(0:3, 1, prob = probabilities)
    })
    incomplete_rows[[length(incomplete_rows) + 1]] <- list(person = paste0("p", person), rater = paste0("r", rater), scores = scores)
  }
}
incomplete_wide <- do.call(rbind, lapply(incomplete_rows, `[[`, "scores"))
colnames(incomplete_wide) <- paste0("i", 1:4)
incomplete_person <- sapply(incomplete_rows, `[[`, "person")
incomplete_rater <- sapply(incomplete_rows, `[[`, "rater")
incomplete <- long_scores(incomplete_wide, incomplete_person, list(rater = incomplete_rater))
incomplete_tam <- function(formula) {
  list(wide = as.data.frame(incomplete_wide), facets = data.frame(rater = incomplete_rater), pid = incomplete_person,
       formula = formula)
}

scores_of <- function(long, facets) {
  c(list(person = long$person), setNames(lapply(facets, function(f) long[[f]]), facets), list(score = long$score))
}

write_fixture("rater-many-facet", list(
  source = "sirt's data.ratings1 (7 raters); TAM's data.sim.mfr (5 raters, 3 topics); a simulation, set.seed(20261029): 300 persons, 4 items, 6 raters, 2 raters per person.",
  cases = list(
    list(name = "ratings1", scores = scores_of(ratings1, c("item", "rater")),
         models = list(
           mfrm_case(ratings1, c("item", "rater"), NULL, ratings1_tam(~ item + rater + step)),
           mfrm_case(ratings1, c("item", "rater"), "item", ratings1_tam(~ item + rater + item:step)))),
    list(name = "data.sim.mfr", scores = scores_of(tam_sim, c("item", "rater", "topic")),
         models = list(
           mfrm_case(tam_sim, c("item", "rater"), NULL, tam_sim_tam(~ item + rater + step)),
           mfrm_case(tam_sim, c("item", "rater", "topic"), NULL, tam_sim_tam(~ item + rater + topic + step)))),
    list(name = "incomplete design", scores = scores_of(incomplete, c("item", "rater")),
         models = list(
           mfrm_case(incomplete, c("item", "rater"), NULL, incomplete_tam(~ item + rater + step)),
           mfrm_case(incomplete, c("item", "rater"), "rater", incomplete_tam(~ item + rater + rater:step))))
  )
))

# --- Exact and adjacent agreement, standardized mean difference --------------------------------------------------------
# irr's agree, the percentage of subjects whose ratings are all within `tolerance` of each other, on the subjects every
# rater rated, as a proportion; and the standardized mean difference of two raters over the pooled SD of their ratings,
# by base R and by effsize's cohen.d, which agree for ratings of the same subjects.

agreement_case <- function(name, x) {
  agree_values <- function(tolerance) irr::agree(x, tolerance)$value / 100
  complete <- na.omit(x)
  rows <- apply(complete, 1, function(row) max(row) - min(row))
  stopifnot(all(sapply(0:3, function(tolerance) abs(agree_values(tolerance) - mean(rows <= tolerance)) < 1e-15)))
  list(name = name, ratings = ratings(x), subjects = nrow(complete), exact = agree_values(0),
       adjacent = agree_values(1), widths = lapply(0:3, function(width) list(width = width, agreement = agree_values(width))))
}

mean_difference <- function(x) {
  complete <- na.omit(x)
  pooled <- sqrt((var(complete[, 1]) + var(complete[, 2])) / 2)
  value <- (mean(complete[, 1]) - mean(complete[, 2])) / pooled
  stopifnot(abs(value - effsize::cohen.d(complete[, 1], complete[, 2])$estimate) < 1e-12)
  value
}

# Two raters' continuous scores of 150 subjects: the second rates 0.4 higher on average and is noisier; 5% are missing.
set.seed(20261029)
true_scores <- rnorm(150, 3, 1)
scores <- cbind(true_scores + rnorm(150, 0, 0.5), true_scores + 0.4 + rnorm(150, 0, 0.7))
scores[matrix(runif(length(scores)) < 0.05, 150)] <- NA

write_fixture("rater-agreement", list(
  source = "irr's anxiety, recoded 0 to 5, also with two ratings missing; irr's diagnoses, coded 0 to 4; a simulation of two raters' continuous scores, set.seed(20261029).",
  agreement = list(
    agreement_case("anxiety, three raters", anxiety),
    agreement_case("anxiety, two raters", anxiety[, 1:2]),
    agreement_case("anxiety with missing ratings", sparse),
    agreement_case("diagnoses, six raters", diagnoses)
  ),
  meanDifference = lapply(list(
    list(name = "anxiety, raters 1 and 2", x = anxiety[, 1:2]),
    list(name = "anxiety, raters 2 and 3", x = anxiety[, 2:3]),
    list(name = "anxiety with missing ratings", x = sparse[, 1:2]),
    list(name = "continuous scores with missing", x = scores)
  ), function(case) list(name = case$name, ratings = ratings(case$x), value = mean_difference(case$x)))
))

# --- Correlation between two raters ----------------------------------------------------------------------------------
# Pearson's r over the subjects both raters rated, by base R's cor on the complete cases: NA, written as null, when a
# rater's ratings are constant.

rating_correlation <- function(x) {
  complete <- na.omit(x)
  if (any(apply(complete, 2, function(column) length(unique(column)) == 1))) return(NA)
  cor(complete[, 1], complete[, 2])
}

write_fixture("rater-correlation", list(
  source = "The two-rater data of rater-agreement: irr's anxiety recoded 0 to 5, also with two ratings missing, and a simulation of continuous scores, set.seed(20261029); base R's cor, and a constant rater.",
  cases = lapply(list(
    list(name = "anxiety, raters 1 and 2", x = anxiety[, 1:2]),
    list(name = "anxiety, raters 2 and 3", x = anxiety[, 2:3]),
    list(name = "anxiety with missing ratings", x = sparse[, 1:2]),
    list(name = "continuous scores with missing", x = scores),
    list(name = "a constant rater", x = cbind(anxiety[, 1], 3)),
    list(name = "decimal ratings with a large offset", x = scores * 0.1 + 1000)
  ), function(case) list(name = case$name, ratings = ratings(case$x), value = rating_correlation(case$x)))
))
