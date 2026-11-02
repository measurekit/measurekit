# Fixtures of Liu and Agresti's cumulative common odds ratio and the logistic discriminant function
# (dif-polytomous-tests.json). References: difR's liu_agresti_ccor for the odds ratio, with the strata the matching
# scores mantelTest uses; base R's equation 3 of Liu and Agresti (1996) for the standard error of its log, as difR's
# se = "liu_agresti" is another estimate (see below); base R's glm for the discriminant function. Data: the cases of
# dif.json and dif-polytomous.json, complete and with 2% of their responses missing, so run dif.R first.

source("validation/psychometrics/common.R")
suppressPackageStartupMessages(library(difR))

ALPHA <- 0.05
MAX_ROUNDS <- 10
# glm's iteratively reweighted least squares, run until the deviance changes by less than this relative amount.
GLM_EPSILON <- 1e-14

# A response matrix from the rows of a fixture, with NA for null.
matrix_of <- function(rows) {
  data <- do.call(rbind, lapply(rows, function(row) vapply(row, function(x) if (is.null(x)) NA_real_ else x, 0)))
  colnames(data) <- paste0("item", seq_len(ncol(data)))
  data
}
numbers <- function(values) vapply(values, function(x) if (is.null(x)) NA_real_ else x, 0)

# dif.R's rule: each item uses the persons who answered it and another matching item; a person's matching score is
# the response to the item plus the total over the other matching items the person answered, scaled to all of them by
# their highest possible scores and rounded, halves up. An item's highest possible score is its highest response, at
# least 1.
match_scores <- function(data, item, anchors) {
  others <- setdiff(anchors, item)
  highest <- pmax(1, suppressWarnings(apply(data, 2, max, na.rm = TRUE)))
  possible <- drop((!is.na(data[, others, drop = FALSE])) %*% highest[others])
  correct <- rowSums(data[, others, drop = FALSE], na.rm = TRUE)
  keep <- !is.na(data[, item]) & possible > 0
  list(rows = which(keep),
       score = data[keep, item] + floor(sum(highest[others]) * correct[keep] / possible[keep] + 0.5))
}

# --- Liu and Agresti ---------------------------------------------------------------------------------------------

# Equation 3 of Liu and Agresti (1996), written out: the variance of log psi over the strata, with row 1 the group whose
# low responses raise psi. A stratum without persons of a group adds nothing.
liu_agresti_variance <- function(y, row1, strata, psi) {
  levels_used <- sort(unique(y))
  thresholds <- length(levels_used) - 1
  spread <- 0
  discordant <- 0
  for (k in unique(strata)) {
    in_k <- strata == k
    n1 <- sum(in_k & row1)
    n2 <- sum(in_k & !row1)
    if (n1 == 0 || n2 == 0) next
    N <- n1 + n2
    x1 <- vapply(levels_used[seq_len(thresholds)], function(level) sum(in_k & row1 & y <= level), 0)
    x2 <- vapply(levels_used[seq_len(thresholds)], function(level) sum(in_k & !row1 & y <= level), 0)
    discordant <- discordant + sum((n1 - x1) * x2) / N
    phi <- function(j, s) {
      n1 * n2 / N^2 * (psi * (n1 - x1[s]) * x2[j] / n1 * (1 + (psi - 1) * x2[s] / n2) +
                         x1[j] * (n2 - x2[s]) / n2 * (psi - (psi - 1) * x1[s] / n1))
    }
    for (s in seq_len(thresholds)) {
      for (j in seq_len(s)) spread <- spread + (if (j == s) 1 else 2) * phi(j, s)
    }
  }
  spread / (psi^2 * discordant^2)
}

# The odds ratio by difR with the focal group as its first level, so that psi is above 1 when the item favors the
# reference group; the standard error by equation 3, which takes the same value with the rows reversed (Liu, 1995).
liu_agresti_item <- function(data, focal, item, anchors) {
  m <- match_scores(data, item, anchors)
  y <- data[m$rows, item]
  g <- focal[m$rows]
  difr <- liu_agresti_ccor(y, factor(g, levels = c(1, 0)), strata = m$score, se = "liu_agresti")
  psi <- difr[["Psi_hat"]]
  variance <- liu_agresti_variance(y, g == 1, m$score, psi)
  stopifnot(abs(liu_agresti_variance(y, g == 0, m$score, 1 / psi) - variance) < 1e-12 * variance)
  se <- sqrt(variance)
  c(oddsRatio = psi, logStandardError = se, z = log(psi) / se, significance = 2 * pnorm(-abs(log(psi) / se)),
    difrStandardError = difr[["SE_log_Psi"]],
    difrReversedStandardError = liu_agresti_ccor(y, factor(g, levels = c(0, 1)), strata = m$score,
                                                 se = "liu_agresti")[["SE_log_Psi"]])
}

liu_agresti_items <- function(data, focal, anchors = seq_len(ncol(data))) {
  rows <- sapply(seq_len(ncol(data)), function(item) liu_agresti_item(data, focal, item, anchors))
  as.list(as.data.frame(t(rows)))
}

# --- The logistic discriminant function -------------------------------------------------------------------------

# Nagelkerke's R² of a model with deviance D, given D0 of the intercept alone.
nagelkerke <- function(deviance, null_deviance, n) (1 - exp((deviance - null_deviance) / n)) / (1 - exp(-null_deviance / n))

# The three likelihood-ratio tests of the regressions of the group on the score, then the response, then their product,
# the response numeric; NA where the group, the score or the response does not vary.
discriminant_tests <- function(g, score, y) {
  undefined <- list(chiSquare = NA, significance = NA, rSquaredChange = NA)
  if (length(unique(g)) < 2 || length(unique(score)) < 2 || length(unique(y)) < 2) {
    return(list(uniform = undefined, nonuniform = undefined, both = undefined))
  }
  fit <- function(formula) glm(formula, family = binomial, control = glm.control(epsilon = GLM_EPSILON, maxit = 100))
  deviances <- vapply(list(g ~ 1, g ~ score, g ~ score + y, g ~ score * y), function(formula) deviance(fit(formula)), 0)
  n <- length(g)
  test <- function(smaller, larger, df) {
    chi <- max(0, deviances[smaller] - deviances[larger])
    list(chiSquare = chi, significance = pchisq(chi, df, lower.tail = FALSE),
         rSquaredChange = nagelkerke(deviances[smaller] - chi, deviances[1], n) -
           nagelkerke(deviances[smaller], deviances[1], n))
  }
  list(uniform = test(2, 3, 1), nonuniform = test(3, 4, 1), both = test(2, 4, 2))
}

# The tests of every item, by test, then statistic, then item; matched on the rule or on `theta`.
discriminant_items <- function(data, focal, anchors = seq_len(ncol(data)), theta = NULL) {
  tests <- lapply(seq_len(ncol(data)), function(item) {
    if (is.null(theta)) {
      m <- match_scores(data, item, anchors)
    } else {
      rows <- which(!is.na(data[, item]) & !is.na(theta))
      m <- list(rows = rows, score = theta[rows])
    }
    discriminant_tests(focal[m$rows], m$score, data[m$rows, item])
  })
  by_test <- function(name) {
    lapply(list(chiSquare = "chiSquare", significance = "significance", rSquaredChange = "rSquaredChange"),
           function(statistic) vapply(tests, function(test) test[[name]][[statistic]], 0))
  }
  list(uniform = by_test("uniform"), nonuniform = by_test("nonuniform"), both = by_test("both"))
}

# Purification as difR's, flagging by the test of both terms.
purified_discriminant <- function(data, focal) {
  anchors <- seq_len(ncol(data))
  result <- discriminant_items(data, focal, anchors)
  flagged <- function(tests) !is.na(tests$both$significance) & tests$both$significance < ALPHA
  flags <- flagged(result)
  iterations <- 0
  converged <- !any(flags)
  while (!converged && iterations < MAX_ROUNDS && sum(!flags) >= 1) {
    iterations <- iterations + 1
    anchors <- which(!flags)
    result <- discriminant_items(data, focal, anchors)
    following <- flagged(result)
    converged <- all(following == flags)
    flags <- following
  }
  c(result, list(anchors = I(anchors), iterations = iterations, converged = converged))
}

# --- The standard error against the spread of log psi ------------------------------------------------------------

# On 1,000 seeded sets of 10 score groups, 60 reference and 40 focal persons each, an item of 4 categories with a common
# cumulative odds ratio of 2 favoring the reference group: the spread of log psi, and the mean of each standard error.
# Equation 3 matches the spread; difR's estimate does not, and differs with the order of the groups.
MONTE_CARLO_SETS <- 1000
monte_carlo <- function() {
  set.seed(20261102)
  cuts <- c(-1, 0, 1)
  estimates <- t(replicate(MONTE_CARLO_SETS, {
    strata <- rep(1:10, each = 100)
    focal <- rep(rep(1:0, c(40, 60)), 10)
    location <- (strata - 5.5) / 3 + ifelse(focal == 1, -log(2) / 2, log(2) / 2)
    y <- vapply(location + rlogis(1000), function(latent) sum(latent > cuts), 0)
    psi <- liu_agresti_ccor(y, factor(focal, levels = c(1, 0)), strata = strata, se = "liu_agresti")
    reversed <- liu_agresti_ccor(y, factor(focal, levels = c(0, 1)), strata = strata, se = "liu_agresti")
    c(log = log(psi[["Psi_hat"]]), equation = sqrt(liu_agresti_variance(y, focal == 1, strata, psi[["Psi_hat"]])),
      difr = psi[["SE_log_Psi"]], difrReversed = reversed[["SE_log_Psi"]])
  }))
  result <- list(spread = sd(estimates[, "log"]), equation = mean(estimates[, "equation"]),
                 difr = mean(estimates[, "difr"]), difrReversed = mean(estimates[, "difrReversed"]))
  message("Monte Carlo: spread ", result$spread, ", equation 3 ", result$equation, ", difR ", result$difr,
          " and reversed ", result$difrReversed)
  stopifnot(abs(result$equation / result$spread - 1) < 0.05)
  result
}

# --- Cases -------------------------------------------------------------------------------------------------------

dichotomous <- jsonlite::fromJSON("packages/psychometrics/test/fixtures/dif.json", simplifyVector = FALSE)$cases
polytomous <- jsonlite::fromJSON("packages/psychometrics/test/fixtures/dif-polytomous.json", simplifyVector = FALSE)$cases
case_named <- function(cases, name) Filter(function(case) case$name == name, cases)[[1]]

# A case of a fixture with its complete or its sparse responses. The Mantel test's purified anchors come from
# dif-polytomous.json, base R's; dif.json has none for it, so its cases are not purified.
fixture_case <- function(fixture, name, sparse) {
  case <- case_named(if (fixture == "dif") dichotomous else polytomous, name)
  part <- if (sparse) case$sparse else case$complete
  data <- matrix_of(if (sparse) case$sparse$responses else case$responses)
  focal <- unlist(case$focal)
  entry <- list(input = list(fixture = fixture, case = name, sparse = sparse),
                liuAgresti = liu_agresti_items(data, focal),
                discriminant = discriminant_items(data, focal),
                purifiedDiscriminant = purified_discriminant(data, focal))
  if (fixture == "dif" && !sparse) {
    stopifnot(max(abs(entry$liuAgresti$oddsRatio - difMH(data, group = focal, focal.name = 1)$alphaMH)) < 1e-12)
  }
  if (fixture == "dif-polytomous") {
    anchors <- unlist(part$purifiedMantel$anchors)
    entry$purifiedLiuAgresti <- c(liu_agresti_items(data, focal, anchors), list(anchors = I(anchors)))
    theta <- numbers(if (sparse) case$sparse$theta else case$theta)
    entry$discriminantTheta <- discriminant_items(data, focal, theta = theta)
  }
  entry
}

write_fixture("dif-polytomous-tests", list(
  source = paste("difR's liu_agresti_ccor with the matching scores as strata, and base R's equation 3 of Liu and",
                 "Agresti (1996) for its standard error; base R's glm for the logistic discriminant function."),
  monteCarlo = monte_carlo(),
  cases = list(
    verbal = fixture_case("dif", "verbal", FALSE),
    "verbal, sparse" = fixture_case("dif", "verbal", TRUE),
    simulated = fixture_case("dif", "simulated", FALSE),
    bfi = fixture_case("dif-polytomous", "bfi", FALSE),
    "bfi, sparse" = fixture_case("dif-polytomous", "bfi", TRUE),
    graded = fixture_case("dif-polytomous", "simulated", FALSE),
    "graded, sparse" = fixture_case("dif-polytomous", "simulated", TRUE)
  )
))
