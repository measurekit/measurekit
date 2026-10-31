# Score reporting and prediction fixtures. References: mirt (fscores(method = "EAPsum"), expected.test, testinfo,
# probtrace), equate (px, the percentile rank function) and base R written out below: the roots of the test
# characteristic curve by uniroot, the conditional standard error of scale scores and the distribution of a total score
# by the recursion of Lord and Wingersky, interpolation by approx, the posterior of theta, and the probabilities of
# reported scores by pnorm and by summing over scores.

source("validation/psychometrics/common.R")
suppressPackageStartupMessages({
  library(mirt)
  library(equate)
  library(equateIRT)
})

POINTS <- 61 # measurekit's default rule: 61 equally spaced nodes on [-6, 6]
RANGE <- c(-6, 6) # the interval of the thetas of total scores that have none

# --- Forms ------------------------------------------------------------------------------------------------------------
# A form is a mirt model with the package's items: LSAT7 under the 2PL and Science under the GPCM as calibrated by mirt,
# and forms 1 and 2 of equateIRT's est3pl (20 3PL items each) set in mirt.

lsat7 <- expand.table(LSAT7)
science <- mirt::Science - 1

package_items <- function(fit, models) {
  irt <- coef(fit, IRTpars = TRUE, simplify = TRUE)$items
  lapply(seq_along(models), function(i) {
    list(model = models[i], parameters = psychometrics_parameters(models[i], irt[i, !is.na(irt[i, ])]))
  })
}

lsat7_fit <- mirt(lsat7, 1, itemtype = "2PL", TOL = 1e-10, quadpts = POINTS, verbose = FALSE)
science_fit <- mirt(science, 1, itemtype = "gpcm", TOL = 1e-10, quadpts = POINTS, verbose = FALSE)

data(est3pl, package = "equateIRT")
est3pl_mods <- modIRT(coef = est3pl$coef, var = est3pl$var, display = FALSE)
est3pl_form <- function(form) {
  coef <- est3pl_mods[[form]]$coef
  ids <- sub("Dffclt.", "", grep("Dffclt", names(coef), value = TRUE))
  a <- unname(coef[paste0("Dscrmn.", ids)])
  b <- unname(coef[paste0("Dffclt.", ids)])
  g <- unname(coef[paste0("Gussng.", ids)])
  dummy <- lsat7[, rep(seq_len(ncol(lsat7)), length.out = length(ids))]
  colnames(dummy) <- ids
  list(
    fit = set_items(dummy, "3PL", data.frame(a1 = a, d = -a * b, g = g)),
    items = lapply(seq_along(ids), function(i) {
      list(model = "3pl", parameters = list(a = a[i], b = b[i], c = g[i], d = 1))
    })
  )
}
est3pl_1 <- est3pl_form(1)
est3pl_2 <- est3pl_form(2)

forms <- list(
  list(name = "LSAT7, 2PL", fit = lsat7_fit, items = package_items(lsat7_fit, rep("2pl", 5))),
  list(name = "Science, GPCM", fit = science_fit, items = package_items(science_fit, rep("gpcm", 4))),
  list(name = "est3pl form 1, 3PL", fit = est3pl_1$fit, items = est3pl_1$items),
  list(name = "est3pl form 2, 3PL", fit = est3pl_2$fit, items = est3pl_2$items)
)
highest <- function(fit) sum(extract.mirt(fit, "K") - 1)

# --- Scales -----------------------------------------------------------------------------------------------------------
# A reported score is slope theta + intercept, rounded to the nearest multiple of the step, halves up, and bounded.
# A bound or step that is absent is -Inf, Inf or 1; a step of 0 leaves the score unrounded.

scales <- list(
  list(slope = 100, intercept = 500, lowest = 200, highest = 800, step = 10),
  list(slope = 15, intercept = 100),
  list(slope = 100, intercept = 500, lowest = 200, highest = 800, step = 0),
  list(slope = 100, intercept = 500, lowest = 205, highest = 795, step = 10)
)
scale_field <- function(scale, name, default) if (is.null(scale[[name]])) default else scale[[name]]
reported <- function(x, scale) {
  step <- scale_field(scale, "step", 1)
  rounded <- if (step > 0) floor(x / step + 0.5) * step else x
  pmin(pmax(rounded, scale_field(scale, "lowest", -Inf)), scale_field(scale, "highest", Inf))
}

# --- Conversion tables ------------------------------------------------------------------------------------------------
# The theta of each total score. EAP of the summed score: mirt's EAPsum under N(0, 1); base R's recursion under another
# prior, with the nodes placed on it. True score: the root of mirt's expected.test by uniroot, and 1 / sqrt(testinfo);
# a score at or below the sum of the lower asymptotes or at or above the sum of the upper ones has no root and takes an
# end of the range.

eap_sum_thetas <- function(fit, prior) {
  z <- seq(-6, 6, length.out = POINTS)
  nodes <- prior$mean + prior$sd * z
  weights <- dnorm(z) / sum(dnorm(z))
  likelihoods <- score_likelihoods(fit, nodes)
  probability <- drop(likelihoods %*% weights)
  theta <- drop(likelihoods %*% (weights * nodes)) / probability
  sd <- sqrt(sapply(seq_along(theta), function(s) sum(likelihoods[s, ] * weights * (nodes - theta[s])^2)) / probability)
  if (prior$mean == 0 && prior$sd == 1) {
    table <- fscores(fit, method = "EAPsum", full.scores = FALSE, quadpts = POINTS, verbose = FALSE)
    stopifnot(max(abs(theta - table$F1), abs(sd - table$SE_F1)) < 1e-10)
    theta <- table$F1
    sd <- table$SE_F1
  }
  list(theta = theta, standardError = sd)
}

item_limits <- function(items) {
  list(lower = sum(sapply(items, function(item) if (is.null(item$parameters$c)) 0 else item$parameters$c)),
       upper = sum(sapply(items, function(item) {
         if (is.null(item$parameters$d)) length(item$parameters$b) else item$parameters$d
       })))
}

true_score_thetas <- function(fit, items) {
  limits <- item_limits(items)
  scores <- 0:highest(fit)
  theta <- sapply(scores, function(x) {
    if (x <= limits$lower) return(RANGE[1])
    if (x >= limits$upper) return(RANGE[2])
    uniroot(function(t) expected.test(fit, matrix(t)) - x, c(-60, 60), tol = 1e-14)$root
  })
  interior <- scores > limits$lower & scores < limits$upper
  list(theta = theta, standardError = ifelse(interior, sapply(theta, function(t) 1 / sqrt(testinfo(fit, matrix(t)))),
                                             NA))
}

conversion_case <- function(form, method, prior, scale) {
  thetas <- if (method == "eap-sum") eap_sum_thetas(form$fit, prior) else true_score_thetas(form$fit, form$items)
  scale_scores <- reported(scale$slope * thetas$theta + scale$intercept, scale)
  likelihoods <- score_likelihoods(form$fit, thetas$theta)
  scale_error <- sapply(seq_along(thetas$theta), function(j) {
    p <- likelihoods[, j]
    sqrt(sum(p * (scale_scores - sum(p * scale_scores))^2))
  })
  list(
    form = form$name, method = method, prior = prior, scale = scale,
    rows = data.frame(score = seq_along(thetas$theta) - 1, theta = thetas$theta, standardError = thetas$standardError,
                      scaleScore = scale_scores, scaleStandardError = scale_error)
  )
}

standard_prior <- list(mean = 0, sd = 1)
shifted_prior <- list(mean = 0.4, sd = 1.3)
conversion_cases <- c(
  unlist(lapply(forms, function(form) {
    lapply(scales[1:3], function(scale) conversion_case(form, "eap-sum", standard_prior, scale))
  }), recursive = FALSE),
  unlist(lapply(forms, function(form) {
    lapply(scales[c(1, 3, 4)], function(scale) conversion_case(form, "true-score", standard_prior, scale))
  }), recursive = FALSE),
  lapply(forms[c(1, 4)], function(form) conversion_case(form, "eap-sum", shifted_prior, scales[[1]]))
)

# The scale scores of equated scores: interpolation in the unrounded table of form Y, ends held, then rounding.
interpolated_case <- function(form, equivalents, scale) {
  table <- conversion_case(form, "eap-sum", standard_prior, modifyList(scale, list(step = 0)))$rows
  conversion <- table$scaleScore
  clamped <- pmin(pmax(equivalents, 0), length(conversion) - 1)
  interpolated <- approx(0:(length(conversion) - 1), conversion, xout = clamped)$y
  list(form = form$name, scale = scale, conversion = conversion, equivalents = equivalents,
       scaleScores = reported(interpolated, scale))
}
scale_conversion_cases <- list(
  interpolated_case(forms[[4]], c(-0.5, 0, 0.3, 7.5, 12.25, 19.99, 20, 20.4), scales[[1]]),
  interpolated_case(forms[[2]], c(0, 0.5, 4.2, 8.9, 9, 12), scales[[2]]),
  interpolated_case(forms[[1]], c(0.25, 1, 2.5, 4.75), modifyList(scales[[1]], list(step = 0)))
)

write_fixture("scaling-conversion", list(
  source = paste("LSAT7 under the 2PL and Science under the GPCM, as mirt calibrates them, and forms 1 and 2 of",
                 "equateIRT's est3pl set in mirt: tables of the EAP of the summed score (mirt's EAPsum under N(0, 1),",
                 "base R's recursion under N(0.4, 1.3^2)) and of the true score (uniroot on expected.test), with the",
                 "conditional standard error of the scale scores by the recursion of Lord and Wingersky; scale scores",
                 "of equivalents by approx."),
  forms = lapply(forms, function(form) list(name = form$name, items = form$items)),
  tables = conversion_cases,
  scaleConversion = scale_conversion_cases,
  conditionalErrors = lapply(forms, function(form) {
    thetas <- c(-3, -0.5, 0, 1.25, 4)
    table <- conversion_case(form, "eap-sum", standard_prior, scales[[1]])$rows
    likelihoods <- score_likelihoods(form$fit, thetas)
    list(form = form$name, conversion = table$scaleScore, thetas = thetas, errors = sapply(seq_along(thetas), function(j) {
      p <- likelihoods[, j]
      sqrt(sum(p * (table$scaleScore - sum(p * table$scaleScore))^2))
    }))
  })
))

# --- Prediction -------------------------------------------------------------------------------------------------------
# The posterior of theta given practice responses to five 2PL items, on the rule placed on the prior: the rectangular
# weights times the likelihood of the answered responses, normalized.

practice <- data.frame(a = c(1.1, 0.8, 1.4, 1, 1.7), b = c(-1.2, -0.4, 0.1, 0.6, 1.3))
practice_posterior <- function(responses, prior) {
  z <- seq(-6, 6, length.out = POINTS)
  nodes <- prior$mean + prior$sd * z
  weights <- dnorm(z) / sum(dnorm(z))
  for (i in which(!is.na(responses))) {
    p <- plogis(practice$a[i] * (nodes - practice$b[i]))
    weights <- weights * if (responses[i] == 1) p else 1 - p
  }
  list(nodes = nodes, weights = weights / sum(weights))
}

# Predicted form scores: P(x) = sum over nodes of w P(x | theta), pooled over total scores with the same scale score,
# scaled to sum to 1; the mean and the equal-tailed interval from the cumulative probability.
summarize <- function(scores, probabilities, level) {
  probabilities <- probabilities / sum(probabilities)
  cumulative <- cumsum(probabilities)
  list(mean = sum(scores * probabilities),
       lower = scores[which(cumulative >= (1 - level) / 2)[1]],
       upper = scores[which(cumulative >= (1 + level) / 2)[1]],
       scores = scores, probabilities = probabilities)
}

predict_form <- function(fit, posterior, conversion, level) {
  per_score <- drop(score_likelihoods(fit, posterior$nodes) %*% posterior$weights)
  scores <- sort(unique(conversion))
  summarize(scores, sapply(scores, function(s) sum(per_score[conversion == s])), level)
}

# Predicted theta scores: the estimate theta + e, e ~ N(0, se^2), is reported as the rounded and bounded score. Written
# per multiple k of the step: the estimates that round to k have v / step in [k - 1/2, k + 1/2); the multiples at or
# below the lowest score report the lowest, those at or above the highest the highest.
predict_theta <- function(posterior, scale, se, level) {
  step <- scale$step
  multiples <- floor(scale$lowest / step):ceiling(scale$highest / step)
  scores <- sort(unique(pmin(pmax(multiples * step, scale$lowest), scale$highest)))
  probability_of_multiple <- function(k) {
    if (se == 0) {
      return(as.numeric(floor((scale$slope * posterior$nodes + scale$intercept) / step + 0.5) == k))
    }
    upper <- if (k == max(multiples)) 1 else pnorm(((k + 0.5) * step - scale$intercept) / scale$slope, posterior$nodes, se)
    lower <- if (k == min(multiples)) 0 else pnorm(((k - 0.5) * step - scale$intercept) / scale$slope, posterior$nodes, se)
    upper - lower
  }
  reports <- pmin(pmax(multiples * step, scale$lowest), scale$highest)
  per_multiple <- sapply(multiples, function(k) sum(posterior$weights * probability_of_multiple(k)))
  if (se == 0) {
    # Multiples beyond the ends take the mass of every estimate beyond them.
    per_multiple[1] <- sum(posterior$weights * (floor((scale$slope * posterior$nodes + scale$intercept) / step + 0.5) <=
                                                  min(multiples)))
    per_multiple[length(multiples)] <- sum(posterior$weights *
      (floor((scale$slope * posterior$nodes + scale$intercept) / step + 0.5) >= max(multiples)))
  }
  summarize(scores, sapply(scores, function(s) sum(per_multiple[reports == s])), level)
}

patterns <- list(c(1, 1, 0, 1, 0), c(1, NA, 1, NA, 1), c(0, 0, 0, 0, 0), c(NA, NA, NA, NA, NA))
priors <- list(standard_prior, list(mean = 0.3, sd = 1.2))
levels <- c(0.8, 0.95)
posteriors <- unlist(lapply(priors, function(prior) {
  lapply(patterns, function(responses) c(list(responses = responses, prior = prior),
                                         practice_posterior(responses, prior)))
}), recursive = FALSE)

# Each prediction names its posterior by position in `posteriors`.
form_predictions <- unlist(lapply(seq_along(posteriors), function(p) {
  unlist(lapply(forms[c(1, 3, 4)], function(form) {
    table <- conversion_case(form, "eap-sum", standard_prior, scales[[1]])$rows
    lapply(levels, function(level) {
      c(list(posterior = p - 1, form = form$name, conversion = table$scaleScore, level = level),
        predict_form(form$fit, posteriors[[p]], table$scaleScore, level))
    })
  }), recursive = FALSE)
}), recursive = FALSE)

theta_scales <- list(scales[[1]], scales[[4]], list(slope = 15, intercept = 100, lowest = 60, highest = 140, step = 1))
theta_predictions <- unlist(lapply(c(1, 2, 4, 5, 8), function(p) {
  unlist(lapply(theta_scales, function(scale) {
    unlist(lapply(c(0, 0.25, 0.6), function(se) {
      lapply(levels, function(level) {
        c(list(posterior = p - 1, scale = scale, standardError = se, level = level),
          predict_theta(posteriors[[p]], scale, se, level))
      })
    }), recursive = FALSE)
  }), recursive = FALSE)
}), recursive = FALSE)

write_fixture("scaling-prediction", list(
  source = paste("Posteriors of theta given responses to five 2PL practice items, on the 61 rectangular points placed",
                 "on N(0, 1) and N(0.3, 1.2^2), and the predicted scale scores on forms of LSAT7, est3pl form 1 and",
                 "est3pl form 2: sums over nodes of the recursion of Lord and Wingersky for a form's score, and pnorm",
                 "of the estimates that round to each multiple of the step for a theta score."),
  practice = practice,
  posteriors = posteriors,
  formPredictions = form_predictions,
  thetaPredictions = theta_predictions
))

# --- Percentile ranks -------------------------------------------------------------------------------------------------
# PR(x) = 100 (#{s < x} + #{s = x} / 2) / n, written out; equate's px, the share below plus half the share at, on
# ACTmath's form X at the integer scores.

percentile_ranks <- function(sample, x) sapply(x, function(v) 100 * (sum(sample < v) + sum(sample == v) / 2) / length(sample))

act <- as.freqtab(ACTmath[, 1:2])
act_table <- as.data.frame(act)
act_sample <- rep(act_table$total, act_table$count)
stopifnot(max(abs(percentile_ranks(act_sample, 0:40) - 100 * px(act))) < 1e-10)

set.seed(20261029)
scale_sample <- pmin(pmax(round(rnorm(2000, 500, 100), -1), 200), 800)
norm_cases <- list(
  list(name = "ACTmath form X, total scores", sample = act_sample, scores = c(0:40, -1, 0.5, 12.25, 40.5, 41)),
  list(name = "simulated scale scores", sample = scale_sample, scores = c(seq(200, 800, by = 5), 199, 512.5, 801)),
  list(name = "one score", sample = I(7), scores = c(6, 7, 8))
)
for (i in seq_along(norm_cases)) norm_cases[[i]]$ranks <- percentile_ranks(norm_cases[[i]]$sample, norm_cases[[i]]$scores)

write_fixture("scaling-norms", list(
  source = "equate's ACTmath form X total scores (px at the integers) and a simulated normal sample of scale scores.",
  cases = norm_cases
))
