# Differential item functioning fixtures. References: difR (difMH, difLogistic, difLord, purified and not), base R's
# mantelhaen.test for the Mantel-Haenszel chi-square, the ETS categories written out below from difMH's odds ratios and
# variances, and base R's statistics under measurekit's matching rule for sparse data. Data: difR's verbal, with
# Gender 1 as the focal group, and a 2PL simulation with known DIF; both also with 2% of the responses missing.
# Polytomous items (dif-polytomous.json): Mantel's chi-square, the standardized mean difference and the generalized
# Mantel-Haenszel chi-square in base R and mantelhaen.test, and ordinal logistic regression by rms::lrm, with
# MASS::polr and lordif's rundif as checks. Data: psych's bfi N1 to N5 by gender and a graded response simulation with known DIF; both also with 2% of
# the responses missing.

source("validation/psychometrics/common.R")
suppressPackageStartupMessages({
  library(difR)
  library(ltm)
})

SEED <- 20261029
ETS_SIGNIFICANCE <- 0.05

# ETS categories (Zwick, 2012): A when |D| < 1 or the chi-square is not significant; C when |D| >= 1.5 and |D| exceeds
# 1 by more than the one-sided critical value times its standard error; B otherwise.
ets_category <- function(delta, se, p) {
  ifelse(abs(delta) < 1 | p >= ETS_SIGNIFICANCE, "A",
         ifelse(abs(delta) >= 1.5 & (abs(delta) - 1) / se > qnorm(1 - ETS_SIGNIFICANCE), "C", "B"))
}

# difR's printed effect size codes, by |D| alone.
difr_category <- function(delta) as.character(symnum(abs(delta), c(0, 1, 1.5, Inf), symbols = c("A", "B", "C")))

# The chi-square of base R's mantelhaen.test, which subtracts the continuity correction only when |Σ (A - E(A))| is at
# least 0.5; difMH subtracts it always, and agrees elsewhere. The odds ratio and the variance of its log are difMH's.
mantel_haenszel <- function(data, focal, correct) {
  result <- difMH(data, group = focal, focal.name = 1, correct = correct)
  total <- rowSums(data)
  kept <- total %in% as.numeric(names(which(table(total) > 1)))
  excess <- sapply(seq_len(ncol(data)), function(i) {
    sum(sapply(split(seq_len(nrow(data))[kept], total[kept]), function(rows) {
      sum(data[rows, i] == 1 & focal[rows] == 0) - sum(focal[rows] == 0) * sum(data[rows, i] == 1) / length(rows)
    }))
  })
  chi_square <- sapply(seq_len(ncol(data)), function(i) {
    tables <- table(factor(data[kept, i], 1:0), factor(focal[kept], 0:1), total[kept])
    unname(mantelhaen.test(tables, correct = correct)$statistic)
  })
  agree <- !correct | abs(excess) >= 0.5
  stopifnot(max(abs(chi_square - result$MH)[agree]) < 1e-12)
  significance <- pchisq(chi_square, 1, lower.tail = FALSE)
  delta <- -2.35 * log(result$alphaMH)
  se <- 2.35 * sqrt(result$varLambda)
  list(
    continuityCorrection = correct,
    chiSquare = chi_square,
    significance = significance,
    oddsRatio = result$alphaMH,
    delta = delta,
    deltaStandardError = se,
    etsCategory = ets_category(delta, se, significance),
    difrCategory = difr_category(delta)
  )
}

logistic <- function(data, focal) {
  test <- function(type) {
    result <- difLogistic(data, group = focal, focal.name = 1, type = type)
    list(chiSquare = result$Logistik, significance = result$p.value, rSquaredChange = result$deltaR2)
  }
  list(uniform = test("udif"), nonuniform = test("nudif"), both = test("both"))
}

# difR's itemPar2PL with ltm run to convergence: its default control, 40 EM cycles and then BFGS at optim's default
# reltol, stops up to 1.5e-3 short of the maximum on these data. difLord rescales these estimates and compares them.
item_parameters_2pl <- function(data) {
  J <- ncol(data)
  mod <- ltm(data ~ z1, IRT.param = FALSE, control = list(GHk = 21, iter.em = 2000, iter.qN = 2000))
  beta <- summary(mod)$coefficients[1:J, 1]
  alpha <- summary(mod)$coefficients[(J + 1):(2 * J), 1]
  covariance <- vcov(mod)
  Sig1 <- diag(covariance)[1:J]
  Sig2 <- diag(covariance)[(J + 1):(2 * J)]
  Sig12 <- diag(covariance[1:J, (J + 1):(2 * J)])
  parameters <- cbind(alpha, -beta / alpha, sqrt(Sig2),
                      sqrt(Sig1 / alpha^2 + Sig2 * beta^2 / alpha^4 - 2 * beta * Sig12 / alpha^3),
                      beta * Sig2 / alpha^2 - Sig12 / alpha)
  dimnames(parameters) <- list(colnames(data), c("a", "b", "se(a)", "se(b)", "cov(a,b)"))
  parameters
}

lord <- function(data, focal) {
  colnames(data) <- paste0("item", seq_len(ncol(data)))
  reference <- item_parameters_2pl(data[focal == 0, ])
  focal_parameters <- item_parameters_2pl(data[focal == 1, ])
  result <- difLord(data, group = focal, focal.name = 1, model = "2PL", irtParam = rbind(reference, focal_parameters),
                    same.scale = FALSE)
  stopifnot(max(abs(LordChi2(reference, itemRescale(reference, focal_parameters)) - result$LordChi)) < 1e-12)
  A <- sd(reference[, "b"]) / sd(focal_parameters[, "b"])
  B <- mean(reference[, "b"]) - A * mean(focal_parameters[, "b"])
  default <- difLord(data, group = focal, focal.name = 1, model = "2PL")
  list(
    chiSquare = result$LordChi,
    significance = result$p.value,
    reference = list(a = unname(reference[, "a"]), b = unname(reference[, "b"])),
    focal = list(a = unname(focal_parameters[, "a"]), b = unname(focal_parameters[, "b"])),
    transformation = list(slope = A, intercept = B),
    difrDefaultChiSquare = default$LordChi
  )
}

# --- Sparse data and purification: base R ----------------------------------------------------------------------
# measurekit's rule: each item uses the persons who answered it and another matching item; a person's matching score
# is the response to the item plus the total over the other matching items the person answered, scaled to all of
# them by their highest possible scores and rounded, halves up. With complete data it is difR's total score over the
# anchors and the studied item, which the script checks against difMH and difLogistic, purified and not.

# The highest possible score of an item is its highest response, at least 1: 1 for a dichotomous item.
match_scores <- function(data, item, anchors) {
  others <- setdiff(anchors, item)
  highest <- pmax(1, suppressWarnings(apply(data, 2, max, na.rm = TRUE)))
  possible <- drop((!is.na(data[, others, drop = FALSE])) %*% highest[others])
  correct <- rowSums(data[, others, drop = FALSE], na.rm = TRUE)
  keep <- !is.na(data[, item]) & possible > 0
  list(rows = which(keep),
       score = data[keep, item] + floor(sum(highest[others]) * correct[keep] / possible[keep] + 0.5))
}

# The Mantel-Haenszel statistics of every item under the rule: mantelhaen.test's chi-square over the score groups of
# more than one person, and the common odds ratio and the Robins-Breslow-Greenland variance of its log written out.
mh_rule <- function(data, focal, anchors, correct = TRUE) {
  rows <- lapply(seq_len(ncol(data)), function(item) {
    m <- match_scores(data, item, anchors)
    y <- data[m$rows, item]
    g <- focal[m$rows]
    kept <- m$score %in% as.numeric(names(which(table(m$score) > 1)))
    tables <- table(factor(y[kept], 1:0), factor(g[kept], 0:1), m$score[kept])
    # mantelhaen.test fails without score groups and gives NaN without variance; both are undefined.
    chi <- tryCatch(unname(mantelhaen.test(tables, correct = correct)$statistic), error = function(e) NA)
    a <- tables[1, 1, ]; b <- tables[2, 1, ]; c <- tables[1, 2, ]; d <- tables[2, 2, ]
    total <- a + b + c + d
    r <- sum(a * d / total); s <- sum(b * c / total)
    variance <- sum((a + d) * a * d / total^2) / (2 * r^2) +
      sum(((a + d) * b * c + (b + c) * a * d) / total^2) / (2 * r * s) + sum((b + c) * b * c / total^2) / (2 * s^2)
    delta <- if (r > 0 && s > 0) -2.35 * log(r / s) else NA
    se <- if (r > 0 && s > 0) 2.35 * sqrt(variance) else NA
    p <- if (is.na(chi)) NA else pchisq(chi, 1, lower.tail = FALSE)
    c(chi, p, if (s > 0) r / s else NA, delta, se)
  })
  values <- do.call(rbind, rows)
  list(chiSquare = values[, 1], significance = values[, 2], oddsRatio = values[, 3], delta = values[, 4],
       deltaStandardError = values[, 5], etsCategory = ets_category(values[, 4], values[, 5], values[, 2]))
}

# The logistic regression tests of every item under the rule, by glm.
lr_rule <- function(data, focal, anchors) {
  tests <- lapply(seq_len(ncol(data)), function(item) {
    m <- match_scores(data, item, anchors)
    y <- data[m$rows, item]
    score <- m$score
    group <- focal[m$rows]
    if (length(unique(y)) < 2 || length(unique(group)) < 2) return(rep(NA, 6))
    deviances <- c(glm(y ~ 1, family = binomial)$deviance, glm(y ~ score, family = binomial)$deviance,
                   glm(y ~ score + group, family = binomial)$deviance, glm(y ~ score * group, family = binomial)$deviance)
    nagelkerke <- function(deviance) (1 - exp((deviance - deviances[1]) / length(y))) / (1 - exp(-deviances[1] / length(y)))
    c(deviances[2] - deviances[3], deviances[3] - deviances[4], deviances[2] - deviances[4],
      nagelkerke(deviances[3]) - nagelkerke(deviances[2]), nagelkerke(deviances[4]) - nagelkerke(deviances[3]),
      nagelkerke(deviances[4]) - nagelkerke(deviances[2]))
  })
  values <- do.call(rbind, tests)
  test <- function(column, r2, df) {
    list(chiSquare = values[, column], significance = pchisq(values[, column], df, lower.tail = FALSE), rSquaredChange = values[, r2])
  }
  list(uniform = test(1, 4, 1), nonuniform = test(2, 5, 1), both = test(3, 6, 2))
}

# Purification as difR's: flag at 0.05, compute again on the items not flagged, until a round flags the items the
# previous one did, every item is flagged, or 10 rounds.
purify_rule <- function(data, focal, statistics, significance) {
  anchors <- seq_len(ncol(data))
  result <- statistics(anchors)
  flags <- significance(result) < 0.05 & !is.na(significance(result))
  iterations <- 0
  converged <- !any(flags)
  while (!converged && iterations < 10 && !all(flags)) {
    iterations <- iterations + 1
    anchors <- which(!flags)
    result <- statistics(anchors)
    following <- significance(result) < 0.05 & !is.na(significance(result))
    converged <- all(following == flags)
    flags <- following
  }
  c(result, list(anchors = I(anchors), iterations = iterations, converged = converged))
}

sparse_case <- function(data, focal) {
  all_items <- seq_len(ncol(data))
  list(
    responses = unname(lapply(seq_len(nrow(data)), function(i) as.numeric(data[i, ]))),
    mantelHaenszel = mh_rule(data, focal, all_items),
    logistic = lr_rule(data, focal, all_items),
    purifiedMantelHaenszel = purify_rule(data, focal, function(a) mh_rule(data, focal, a), function(r) r$significance),
    purifiedLogistic = purify_rule(data, focal, function(a) lr_rule(data, focal, a), function(r) r$both$significance)
  )
}

# On complete data the rule is difR's: its statistics and its purification.
check_complete <- function(data, focal, purified_mh, purified_lr) {
  stopifnot(max(abs(mh_rule(data, focal, seq_len(ncol(data)))$oddsRatio -
                     difMH(data, group = focal, focal.name = 1)$alphaMH)) < 1e-12)
  difr_mh <- difMH(data, group = focal, focal.name = 1, purify = TRUE)
  stopifnot(max(abs(purified_mh$oddsRatio - difr_mh$alphaMH)) < 1e-12, purified_mh$iterations == difr_mh$nrPur,
            purified_mh$converged == difr_mh$convergence)
  difr_lr <- difLogistic(data, group = focal, focal.name = 1, purify = TRUE)
  stopifnot(max(abs(purified_lr$both$chiSquare - difr_lr$Logistik)) < 1e-8, purified_lr$iterations == difr_lr$nrPur,
            purified_lr$converged == difr_lr$convergence)
}

dif_case <- function(name, data, focal) {
  complete <- sparse_case(data, focal)
  check_complete(data, focal, complete$purifiedMantelHaenszel, complete$purifiedLogistic)
  # 2% of the responses missing at random.
  set.seed(SEED)
  sparse <- data
  sparse[matrix(runif(length(sparse)) < 0.02, nrow(sparse))] <- NA
  message(name, ": ", sum(complete.cases(sparse)), " of ", nrow(sparse), " persons answered every item")
  list(
    name = name,
    responses = unname(lapply(seq_len(nrow(data)), function(i) as.numeric(data[i, ]))),
    focal = as.numeric(focal),
    mantelHaenszel = list(mantel_haenszel(data, focal, TRUE), mantel_haenszel(data, focal, FALSE)),
    logistic = logistic(data, focal),
    purifiedMantelHaenszel = complete$purifiedMantelHaenszel,
    purifiedLogistic = complete$purifiedLogistic,
    sparse = sparse_case(sparse, focal),
    lord = lord(data, focal)
  )
}

data(verbal)
verbal_items <- as.matrix(verbal[, 1:24])

# 2PL simulation: 20 items, 1000 persons per group, the focal group's theta N(-0.5, 1). Items 1 and 2 are 0.6 harder
# for the focal group (uniform DIF); item 3 discriminates less in it, a = 0.5 instead of 1.5 (nonuniform DIF).
set.seed(SEED)
items <- 20
a <- runif(items, 0.8, 2)
b <- rnorm(items)
a[3] <- 1.5
focal_a <- a
focal_a[3] <- 0.5
focal_b <- b
focal_b[1:2] <- b[1:2] + 0.6
simulate <- function(theta, a, b) {
  t(sapply(theta, function(t) as.numeric(runif(items) < plogis(a * (t - b)))))
}
simulated <- rbind(simulate(rnorm(1000), a, b), simulate(rnorm(1000, -0.5), focal_a, focal_b))
simulated_focal <- rep(0:1, each = 1000)

write_fixture("dif", list(
  source = "difR's verbal (24 items, 316 persons, Gender 1 focal) and a 2PL simulation with DIF in items 1 to 3.",
  cases = list(
    dif_case("verbal", verbal_items, verbal$Gender),
    dif_case("simulated", simulated, simulated_focal)
  )
))

# --- Polytomous items ------------------------------------------------------------------------------------------
# Each item's persons and matching scores are those of the rule above, with a given score such as theta in place of
# the total score for ordinal logistic regression.

POLYTOMOUS_ETS <- c(moderate = 0.17, large = 0.25)

# ETS categories (Zwick, 2012): AA when Mantel's chi-square is not significant or |SMD / SD| <= 0.17; CC when it is
# significant and |SMD / SD| > 0.25; BB otherwise.
polytomous_category <- function(effect, p) {
  ifelse(is.na(effect) | is.na(p), NA_character_,
         ifelse(p >= ETS_SIGNIFICANCE | abs(effect) <= POLYTOMOUS_ETS[["moderate"]], "AA",
                ifelse(abs(effect) > POLYTOMOUS_ETS[["large"]], "CC", "BB")))
}

# Mantel's chi-square from the rows of each score group: with n persons, n_F of them focal, the focal group's sum of
# responses F has mean n_F * mean(y) and variance n_R n_F var(y) / n, var the sample variance, which is the sampling
# variance of n_F draws without replacement from the score group (Mantel, 1963).
mantel_chi_square <- function(y, g, groups) {
  groups <- Filter(function(rows) length(rows) > 1, groups)
  excess <- sum(vapply(groups, function(rows) sum(y[rows][g[rows] == 1]) - sum(g[rows] == 1) * mean(y[rows]), 0))
  variance <- sum(vapply(groups, function(rows) {
    n <- length(rows)
    focal <- sum(g[rows] == 1)
    focal * (n - focal) * var(y[rows]) / n
  }, 0))
  if (variance > 0) excess^2 / variance else NA_real_
}

# The standardized mean difference over the score groups with persons of both groups, weighted by the focal group's.
standardized_mean_difference <- function(y, g, groups) {
  groups <- Filter(function(rows) any(g[rows] == 1) && any(g[rows] == 0), groups)
  if (length(groups) == 0) return(NA_real_)
  focal <- vapply(groups, function(rows) sum(g[rows] == 1), 0)
  difference <- vapply(groups, function(rows) mean(y[rows][g[rows] == 1]) - mean(y[rows][g[rows] == 0]), 0)
  sum(focal * difference) / sum(focal)
}

# The generalized Mantel-Haenszel test by mantelhaen.test on the 2 x K x S table of the categories used in the score
# groups of more than one person; its degrees of freedom are the categories used less 1. A singular covariance matrix,
# which mantelhaen.test reports as an error, is undefined.
generalized_mantel_haenszel <- function(y, g, score) {
  kept <- score %in% as.numeric(names(which(table(score) > 1)))
  used <- sort(unique(y[kept]))
  df <- max(length(used) - 1, 0)
  if (df < 1) return(c(NA_real_, df, NA_real_))
  table3 <- table(factor(g[kept], 0:1), factor(y[kept], used), score[kept])
  chi <- tryCatch(unname(mantelhaen.test(table3, correct = FALSE)$statistic), error = function(e) NA_real_)
  c(chi, df, if (is.na(chi)) NA_real_ else pchisq(chi, df, lower.tail = FALSE))
}

mantel_rule <- function(data, focal, anchors) {
  rows <- lapply(seq_len(ncol(data)), function(item) {
    m <- match_scores(data, item, anchors)
    y <- data[m$rows, item]
    g <- focal[m$rows]
    groups <- split(seq_along(y), m$score)
    chi <- mantel_chi_square(y, g, groups)
    p <- if (is.na(chi)) NA_real_ else pchisq(chi, 1, lower.tail = FALSE)
    smd <- standardized_mean_difference(y, g, groups)
    spread <- sd(y)
    effect <- if (is.na(smd) || is.na(spread) || spread == 0) NA_real_ else smd / spread
    c(chi, p, generalized_mantel_haenszel(y, g, m$score), smd, effect)
  })
  values <- do.call(rbind, rows)
  list(chiSquare = values[, 1], significance = values[, 2], generalizedChiSquare = values[, 3],
       generalizedDegreesOfFreedom = values[, 4], generalizedSignificance = values[, 5],
       standardizedMeanDifference = values[, 6], effectSize = values[, 7],
       etsCategory = polytomous_category(values[, 7], values[, 2]))
}

# Mantel's chi-square against mantelhaen.test on the dichotomized item, where it is the Mantel-Haenszel chi-square
# without the continuity correction, and the generalized test against Mantel's where the item has two categories.
check_mantel <- function(data, focal) {
  for (item in seq_len(ncol(data))) {
    m <- match_scores(data, item, seq_len(ncol(data)))
    y <- as.numeric(data[m$rows, item] > 0)
    kept <- m$score %in% as.numeric(names(which(table(m$score) > 1)))
    chi <- mantel_chi_square(y, focal[m$rows], split(seq_along(y), m$score))
    reference <- unname(mantelhaen.test(table(factor(y[kept], 1:0), factor(focal[m$rows][kept], 0:1), m$score[kept]),
                                        correct = FALSE)$statistic)
    stopifnot(abs(chi - reference) < 1e-10 * max(1, reference),
              abs(generalized_mantel_haenszel(y, focal[m$rows], m$score)[1] - reference) < 1e-10 * max(1, reference))
  }
}

# The cumulative logit models of an item's responses by rms::lrm, lordif's engine, run to eps = 1e-14: the deviance
# -2 ln L and the coefficient of the score, which is centered. MASS::polr, which stops where BFGS does, agrees in the
# deviance within 1e-8 and in the coefficient within 1e-5. The deviance of the model with only thresholds is
# -2 sum n_k ln(n_k / n); lrm agrees.
ordinal_models <- function(y, g, score) {
  score <- score - mean(score)
  y <- factor(y, ordered = TRUE)
  fit <- function(formula) {
    model <- rms::lrm(formula, eps = 1e-14, maxit = 100)
    check <- MASS::polr(formula, method = "logistic", control = list(reltol = 1e-14, maxit = 10000))
    stopifnot(abs(deviance(check) - model$deviance[2]) < 1e-8,
              abs(coef(check)[["score"]] - coef(model)[["score"]]) < 1e-5)
    list(deviance = model$deviance[2], slope = unname(coef(model)[["score"]]), null = model$deviance[1])
  }
  counts <- table(y)
  null_deviance <- -2 * sum(counts * log(counts / length(y)))
  fits <- list(score = fit(y ~ score), group = fit(y ~ score + g), interaction = fit(y ~ score * g))
  stopifnot(abs(fits$score$null - null_deviance) < 1e-8)
  c(list(null = null_deviance), fits)
}

# Nagelkerke's R squared of a deviance, given the deviance of the thresholds alone.
nagelkerke_r2 <- function(deviance, null_deviance, n) (1 - exp((deviance - null_deviance) / n)) / (1 - exp(-null_deviance / n))

ordinal_rule <- function(data, focal, anchors, theta = NULL) {
  tests <- lapply(seq_len(ncol(data)), function(item) {
    if (is.null(theta)) {
      m <- match_scores(data, item, anchors)
    } else {
      rows <- which(!is.na(data[, item]) & !is.na(theta))
      m <- list(rows = rows, score = theta[rows])
    }
    y <- data[m$rows, item]
    g <- focal[m$rows]
    if (length(unique(y)) < 2 || length(unique(g)) < 2 || length(unique(m$score)) < 2) return(rep(NA_real_, 7))
    fits <- ordinal_models(y, g, m$score)
    n <- length(y)
    r2 <- function(model) nagelkerke_r2(model$deviance, fits$null, n)
    c(fits$score$deviance - fits$group$deviance, fits$group$deviance - fits$interaction$deviance,
      fits$score$deviance - fits$interaction$deviance,
      r2(fits$group) - r2(fits$score), r2(fits$interaction) - r2(fits$group), r2(fits$interaction) - r2(fits$score),
      abs((fits$group$slope - fits$score$slope) / fits$score$slope))
  })
  values <- do.call(rbind, tests)
  test <- function(column, r2, df) {
    list(chiSquare = values[, column], significance = pchisq(values[, column], df, lower.tail = FALSE),
         rSquaredChange = values[, r2])
  }
  list(uniform = test(1, 4, 1), nonuniform = test(2, 5, 1), both = test(3, 6, 2), slopeChange = values[, 7])
}

# lordif's rundif on complete data, with the matching score of every item given: its p-values, slope change and
# Nagelkerke R squared changes, which it rounds to 4 decimals. Item i is matched on theta[, i].
lordif_rundif <- function(data, focal, theta) {
  rows <- lapply(seq_len(ncol(data)), function(item) {
    stats <- lordif::rundif(item = 1, resp = data[, item, drop = FALSE], theta = theta[, item], gr = focal,
                            criterion = "Chisqr", alpha = 0.05, beta.change = 0.1, pseudo.R2 = "Nagelkerke",
                            R2.change = 0.02, wt = NULL)$stats
    unlist(stats[, c("chi12", "chi23", "chi13", "beta12", "pseudo12.Nagelkerke", "pseudo23.Nagelkerke",
                     "pseudo13.Nagelkerke")])
  })
  values <- do.call(rbind, rows)
  list(uniform = list(significance = values[, 1], rSquaredChange = values[, 5]),
       nonuniform = list(significance = values[, 2], rSquaredChange = values[, 6]),
       both = list(significance = values[, 3], rSquaredChange = values[, 7]), slopeChange = values[, 4])
}

# On complete data the polytomous rule is lordif's: rundif given the rule's matching scores, which are the total score
# over the anchors and the studied item.
check_lordif <- function(data, focal, anchors, ordinal) {
  theta <- sapply(seq_len(ncol(data)), function(item) rowSums(data[, union(anchors, item), drop = FALSE]))
  reference <- lordif_rundif(data, focal, theta)
  expect_rounded(ordinal, reference)
  reference
}

# measurekit's p-values, R squared changes and slope change are within lordif's rounding to 4 decimals of its own.
expect_rounded <- function(ordinal, reference) {
  differences <- c(
    unlist(lapply(c("uniform", "nonuniform", "both"), function(type) {
      c(ordinal[[type]]$significance - reference[[type]]$significance,
        ordinal[[type]]$rSquaredChange - reference[[type]]$rSquaredChange)
    })),
    ordinal$slopeChange - reference$slopeChange)
  stopifnot(max(abs(differences)) < 5e-5 + 1e-9)
}

# The statistics of a response matrix under every matching criterion: the rule's, theta given, and purified.
polytomous_statistics <- function(data, focal, theta) {
  all_items <- seq_len(ncol(data))
  list(
    mantel = mantel_rule(data, focal, all_items),
    ordinal = ordinal_rule(data, focal, all_items),
    ordinalTheta = ordinal_rule(data, focal, all_items, theta),
    purifiedMantel = purify_rule(data, focal, function(a) mantel_rule(data, focal, a), function(r) r$significance),
    purifiedOrdinal = purify_rule(data, focal, function(a) ordinal_rule(data, focal, a), function(r) r$both$significance)
  )
}

# theta: the EAP estimates of one graded response calibration of all persons, the criterion lordif matches on.
eap_theta <- function(data) {
  model <- mirt::mirt(data, 1, itemtype = "graded", TOL = 1e-7, technical = list(NCYCLES = 5000), verbose = FALSE)
  as.numeric(mirt::fscores(model, method = "EAP", verbose = FALSE)[, 1])
}

polytomous_case <- function(name, data, focal) {
  message(name, ": ", nrow(data), " persons, ", ncol(data), " items")
  check_mantel(data, focal)
  theta <- eap_theta(data)
  complete <- polytomous_statistics(data, focal, theta)
  # lordif's rundif on complete data: the total score, theta, and each item's final purified criterion.
  all_items <- seq_len(ncol(data))
  lordif <- check_lordif(data, focal, all_items, complete$ordinal)
  theta_matrix <- matrix(theta, nrow(data), ncol(data))
  lordif_theta <- lordif_rundif(data, focal, theta_matrix)
  expect_rounded(complete$ordinalTheta, lordif_theta)
  check_lordif(data, focal, complete$purifiedOrdinal$anchors, complete$purifiedOrdinal)
  # 2% of the responses missing at random.
  set.seed(SEED)
  sparse <- data
  sparse[matrix(runif(length(sparse)) < 0.02, nrow(sparse))] <- NA
  message(name, ": ", sum(complete.cases(sparse)), " of ", nrow(sparse), " persons answered every item")
  sparse_theta <- eap_theta(sparse)
  list(
    name = name,
    responses = unname(lapply(seq_len(nrow(data)), function(i) as.numeric(data[i, ]))),
    focal = as.numeric(focal),
    theta = theta,
    complete = c(complete, list(lordif = lordif, lordifTheta = lordif_theta)),
    sparse = c(list(responses = unname(lapply(seq_len(nrow(sparse)), function(i) as.numeric(sparse[i, ]))),
                    theta = sparse_theta),
               polytomous_statistics(sparse, focal, sparse_theta))
  )
}

# psych's bfi: N1 to N5, 6 categories recoded from 0, women (gender 2) focal, complete cases.
bfi_data <- psych::bfi[complete.cases(psych::bfi[, c(paste0("N", 1:5), "gender")]), ]
bfi_items <- as.matrix(bfi_data[, paste0("N", 1:5)]) - 1

# Graded response simulation: 15 items of 4 categories, 1000 persons per group, the focal group's theta N(-0.5, 1).
# Items 1 and 2 have thresholds 0.6 higher for the focal group (uniform DIF); item 3, with thresholds close together,
# discriminates less in it, a = 0.5 instead of 1.5 (nonuniform DIF).
set.seed(SEED)
graded_items <- 15
graded_categories <- 4
a <- runif(graded_items, 0.8, 2)
b <- t(apply(matrix(rnorm(graded_items * (graded_categories - 1)), graded_items), 1, sort))
a[3] <- 1.5
b[3, ] <- c(-0.3, 0, 0.3)
focal_a <- a
focal_a[3] <- 0.5
focal_b <- b
focal_b[1:2, ] <- b[1:2, ] + 0.6
simulate_graded <- function(theta, a, b) {
  t(sapply(theta, function(t) rowSums(runif(graded_items) < plogis(a * (t - b)))))
}
graded <- rbind(simulate_graded(rnorm(1000), a, b), simulate_graded(rnorm(1000, -0.5), focal_a, focal_b))
colnames(graded) <- paste0("item", seq_len(graded_items))
graded_focal <- rep(0:1, each = 1000)

write_fixture("dif-polytomous", list(
  source = "psych's bfi N1 to N5 (complete cases, women focal) and a graded response simulation with DIF in items 1 to 3.",
  cases = list(
    polytomous_case("bfi", bfi_items, as.numeric(bfi_data$gender == 2)),
    polytomous_case("simulated", graded, graded_focal)
  )
))
