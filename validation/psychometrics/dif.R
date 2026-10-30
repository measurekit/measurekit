# Differential item functioning fixtures. References: difR (difMH, difLogistic, difLord, purified and not), base R's
# mantelhaen.test for the Mantel-Haenszel chi-square, the ETS categories written out below from difMH's odds ratios and
# variances, and base R's statistics under measurekit's matching rule for sparse data. Data: difR's verbal, with
# Gender 1 as the focal group, and a 2PL simulation with known DIF; both also with 2% of the responses missing.

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
# them and rounded, halves up. With complete data it is difR's total score over the anchors and the studied item,
# which the script checks against difMH and difLogistic, purified and not.

match_scores <- function(data, item, anchors) {
  others <- setdiff(anchors, item)
  answered <- rowSums(!is.na(data[, others, drop = FALSE]))
  correct <- rowSums(data[, others, drop = FALSE], na.rm = TRUE)
  keep <- !is.na(data[, item]) & answered > 0
  list(rows = which(keep), score = data[keep, item] + floor(length(others) * correct[keep] / answered[keep] + 0.5))
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
