# Fixtures of the Rasch tests (fit-rasch-tests.json): Andersen's likelihood-ratio test, the Martin-Löf test and the Wald
# test of each item. References: eRm's LRtest, MLoef and Waldtest. Where eRm's value is not the statistic, the script
# writes it out in base R and checks why:
# - eRm's PCM does not maximize the conditional likelihood of missing responses of polytomous items (rasch-cml.R), so
#   neither does its LRtest: a small sparse test is fitted by base R's conditional likelihood by every pattern;
# - MLoef counts the parameters of the partial credit model in its degrees of freedom, also for the rating scale model,
#   where each subset estimates steps of its own: base R counts the parameters of eRm's fits;
# - Waldtest compares each item-category parameter; measurekit compares each item's location, the mean of its
#   thresholds, which for a dichotomous item is its difficulty and for the rating scale model is eRm's first
#   item-category parameter: for the partial credit model, base R from eRm's PCM of each group.
# Data: eRm's raschdat1, complete and with 2% of the responses missing, and pcmdat2; seeded simulations of the rating
# scale model, of a Rasch test with an item no person of the upper group misses, and of a small sparse mixed test.

source("validation/psychometrics/common.R")
suppressPackageStartupMessages(library(eRm))

set.seed(20261103)
MISSING_SHARE <- 0.02

rows_of <- function(data) unname(lapply(seq_len(nrow(data)), function(i) as.numeric(data[i, ])))
by_item <- function(values, categories) unname(split(values, rep(seq_along(categories), categories)))

fit_model <- function(data, model) {
  switch(model, rasch = RM(data, sum0 = TRUE), "partial-credit" = PCM(data, sum0 = TRUE),
         "rating-scale" = RSM(data, sum0 = TRUE))
}

# The persons a conditional likelihood uses: two or more answered items, a total neither 0 nor its highest.
used_persons <- function(data) {
  highest <- apply(data, 2, max, na.rm = TRUE)
  answered <- !is.na(data)
  total <- rowSums(data, na.rm = TRUE)
  possible <- drop(answered %*% highest)
  sum(rowSums(answered) >= 2 & total > 0 & total < possible)
}

# eRm's split of the persons: at the median or mean of the total score, within each pattern of missing responses
# when there are missing ones; a total at the median, or at the mean, goes to the lower group.
person_split <- function(data, split) {
  if (length(split) > 1) return(split)
  pattern <- apply(is.na(data), 1, paste, collapse = "")
  total <- rowSums(data, na.rm = TRUE)
  center <- ave(total, pattern, FUN = if (split == "median") median else mean)
  ifelse(total <= center, "low", "high")
}

# --- eRm's thresholds moved to measurekit's scale, as in rasch-cml.R ------------------------------------------------

normalized <- function(thresholds, covariance) {
  n <- length(thresholds)
  projection <- diag(n) - matrix(1 / n, n, n)
  list(thresholds = drop(projection %*% thresholds), covariance = projection %*% covariance %*% t(projection))
}

erm_thresholds <- function(model, categories) {
  covariance <- model$W %*% solve(model$hessian) %*% t(model$W)
  n <- length(model$betapar)
  difference <- matrix(0, n, n)
  start <- 0
  for (m in categories) {
    for (k in seq_len(m)) {
      difference[start + k, start + k] <- -1
      if (k > 1) difference[start + k, start + k - 1] <- 1
    }
    start <- start + m
  }
  normalized(drop(difference %*% model$betapar), difference %*% covariance %*% t(difference))
}

# Each item's location, the mean of its thresholds, and its variance.
locations <- function(model, categories) {
  moved <- erm_thresholds(model, categories)
  index <- rep(seq_along(categories), categories)
  t(vapply(seq_along(categories), function(i) {
    rows <- which(index == i)
    c(mean(moved$thresholds[rows]), sum(moved$covariance[rows, rows]) / length(rows)^2)
  }, numeric(2)))
}

# --- Andersen's likelihood-ratio test -------------------------------------------------------------------------------

andersen_case <- function(name, data, model, split) {
  fit <- fit_model(data, model)
  labels <- person_split(data, split)
  test <- suppressWarnings(LRtest(fit, splitcr = split))
  excluded <- colnames(data)[sort(unique(unlist(lapply(unique(labels), function(level) {
    eRm:::datcheck.LRtest(data[labels == level, , drop = FALSE], data, fit$model)
  }))))]
  kept <- data[, setdiff(colnames(data), excluded), drop = FALSE]
  list(name = name, model = model, responses = rows_of(data), split = if (length(split) > 1) I(split) else split,
       statistic = test$LR, degreesOfFreedom = test$df, significance = test$pvalue, excluded = I(excluded),
       persons = used_persons(kept))
}

# --- The Wald test --------------------------------------------------------------------------------------------------

wald_case <- function(name, data, model, split) {
  fit <- fit_model(data, model)
  test <- suppressWarnings(Waldtest(fit, splitcr = split))
  excluded <- colnames(data)[sort(test$it.ex)]
  kept <- setdiff(colnames(data), excluded)
  labels <- person_split(data, split)
  levels <- sort(unique(labels))
  if (length(split) == 1) levels <- c("low", "high")
  if (model == "partial-credit") {
    categories <- apply(data[, kept], 2, max, na.rm = TRUE)
    groups <- lapply(levels, function(level) locations(fit_model(data[labels == level, kept], model), categories))
    z <- (groups[[2]][, 1] - groups[[1]][, 1]) / sqrt(groups[[1]][, 2] + groups[[2]][, 2])
  } else {
    # The first item-category parameter of each item: its easiness, so minus its location.
    first <- paste0("beta ", kept, if (model == "rating-scale") ".c1" else "")
    z <- unname(test$coef.table[first, "z-statistic"])
    if (model == "rating-scale") {
      categories <- apply(data[, kept], 2, max, na.rm = TRUE)
      groups <- lapply(levels, function(level) locations(fit_model(data[labels == level, kept], model), categories))
      written <- (groups[[2]][, 1] - groups[[1]][, 1]) / sqrt(groups[[1]][, 2] + groups[[2]][, 2])
      stopifnot(max(abs(written - z)) < 1e-6)
    }
  }
  expected <- setNames(rep(NA_real_, ncol(data)), colnames(data))
  expected[kept] <- z
  list(name = name, model = model, responses = rows_of(data), split = if (length(split) > 1) I(split) else split,
       z = unname(expected), significance = unname(2 * pnorm(-abs(expected))), excluded = I(excluded),
       persons = used_persons(data[, kept, drop = FALSE]))
}

# --- The Martin-Löf test --------------------------------------------------------------------------------------------

martin_lof_case <- function(name, data, model, split) {
  complete <- data[stats::complete.cases(data), , drop = FALSE]
  fit <- fit_model(complete, model)
  vector <- if (is.list(split)) {
    vapply(colnames(complete), function(id) which(vapply(split, function(set) id %in% set, TRUE)), 0)
  } else split
  test <- MLoef(fit, splitcr = vector)
  # The degrees of freedom: the cells of the table of subscores less 1 and each subset's parameters, against the
  # totals less 1 and the parameters of the whole test.
  highest <- apply(complete, 2, max)
  groups <- test$i.groups
  cells <- prod(vapply(groups, function(g) sum(highest[g]) + 1, 0))
  parameters <- sum(vapply(test$subModels, function(m) m$npar, 0))
  df <- (cells - 1 + parameters) - (sum(highest) + fit$npar)
  if (model != "rating-scale") stopifnot(df == test$df)
  list(name = name, model = model, responses = rows_of(data),
       split = if (is.list(split)) lapply(split, I) else split,
       statistic = test$LR, degreesOfFreedom = df, significance = pchisq(test$LR, df, lower.tail = FALSE),
       ermDegreesOfFreedom = test$df, persons = nrow(complete))
}

# --- Data -----------------------------------------------------------------------------------------------------------

# Item ids as measurekit's buildResponseMatrix gives them, so that the sets of the Martin-Löf test and the items left out
# name the same items.
with_ids <- function(data) {
  colnames(data) <- paste0("item", seq_len(ncol(data)))
  data
}
rasch <- with_ids(as.matrix(raschdat1))
sparse_rasch <- rasch
sparse_rasch[matrix(runif(length(sparse_rasch)) < MISSING_SHARE, nrow(sparse_rasch))] <- NA
three_labels <- sample(c("a", "b", "c"), nrow(rasch), replace = TRUE)
two_labels <- sample(c("focal", "reference"), nrow(rasch), replace = TRUE)
partial <- with_ids(as.matrix(pcmdat2))

simulate_rasch_family <- function(persons, thresholds) {
  theta <- rnorm(persons)
  data <- sapply(thresholds, function(tau) {
    vapply(theta, function(t) {
      weights <- exp(c(0, cumsum(t - tau)))
      sample(0:length(tau), 1, prob = weights / sum(weights))
    }, 0)
  })
  colnames(data) <- paste0("item", seq_along(thresholds))
  data
}

# The rating scale model: 500 persons, 6 items of 4 categories, ordered steps.
STEPS <- c(-1, 0, 1)
rating <- simulate_rasch_family(500, lapply(seq(-0.8, 0.8, length.out = 6), function(b) b + STEPS))

# A Rasch test whose first item every person of the upper half answers right: both eRm and measurekit leave it out.
easy <- simulate_rasch_family(300, as.list(c(-5, seq(-1.5, 1.5, length.out = 7))))
upper <- person_split(easy, "median") == "high"
stopifnot(all(easy[upper, 1] == 1), any(easy[!upper, 1] == 0))

# --- Base R: the conditional likelihood of a small sparse test, by every pattern, as in rasch-cml.R ---------------

small_categories <- c(1, 1, 1, 2, 3)
small <- simulate_rasch_family(600, list(-0.6, 0, 0.6, c(-0.6, 0.4), c(-0.8, 0, 0.8)))
small[matrix(runif(length(small)) < 0.05, nrow(small))] <- NA

conditional_loglik <- function(free, data, categories) {
  thresholds <- c(free, -sum(free))
  eta <- lapply(by_item(thresholds, categories), function(tau) c(0, cumsum(tau)))
  total <- 0
  for (p in seq_len(nrow(data))) {
    answered <- which(!is.na(data[p, ]))
    if (length(answered) < 2) next
    x <- data[p, answered]
    r <- sum(x)
    if (r == 0 || r == sum(categories[answered])) next
    patterns <- as.matrix(expand.grid(lapply(categories[answered], function(m) 0:m)))
    patterns <- patterns[rowSums(patterns) == r, , drop = FALSE]
    weight <- function(pattern) exp(-sum(vapply(seq_along(answered), function(j) eta[[answered[j]]][pattern[j] + 1], 0)))
    total <- total + log(weight(x)) - log(sum(apply(patterns, 1, weight)))
  }
  -total
}

base_loglik <- function(data, categories) {
  n <- sum(categories)
  fit <- optim(rep(0, n - 1), conditional_loglik, data = data, categories = categories, method = "BFGS",
               control = list(reltol = 1e-15, maxit = 1000))
  stopifnot(fit$convergence == 0)
  -fit$value
}

base_andersen <- function() {
  labels <- person_split(small, "median")
  # Every category of every item has a response from the persons each group uses.
  for (level in unique(labels)) {
    group <- small[labels == level, ]
    total <- rowSums(group, na.rm = TRUE)
    possible <- drop((!is.na(group)) %*% small_categories)
    used <- group[rowSums(!is.na(group)) >= 2 & total > 0 & total < possible, ]
    stopifnot(all(vapply(seq_along(small_categories), function(i) {
      all(0:small_categories[i] %in% used[, i])
    }, TRUE)))
  }
  full <- base_loglik(small, small_categories)
  groups <- vapply(unique(labels), function(level) base_loglik(small[labels == level, ], small_categories), 0)
  statistic <- 2 * (sum(groups) - full)
  df <- length(unique(labels)) * (sum(small_categories) - 1) - (sum(small_categories) - 1)
  erm <- suppressWarnings(LRtest(PCM(small, sum0 = TRUE)))
  message("Small sparse test: Andersen's statistic is ", statistic, ", where eRm's LRtest gives ", erm$LR)
  stopifnot(abs(erm$LR - statistic) > 0.1)
  list(name = "small, sparse, median", model = "partial-credit", responses = rows_of(small), split = "median",
       statistic = statistic, degreesOfFreedom = df, significance = pchisq(statistic, df, lower.tail = FALSE),
       excluded = I(character(0)), persons = used_persons(small), erm = erm$LR)
}

halves <- list(paste0("item", seq(1, 30, 2)), paste0("item", seq(2, 30, 2)))
thirds <- lapply(1:3, function(g) paste0("item", seq(g, 30, 3)))

write_fixture("fit-rasch-tests", list(
  source = paste("eRm's LRtest, MLoef and Waldtest on raschdat1, complete and sparse, pcmdat2 and simulations of the",
                 "rating scale model and of a Rasch test with an item left out; base R where eRm's value is not the",
                 "statistic: the conditional likelihood of a small sparse test, the degrees of freedom of the rating",
                 "scale model's Martin-Löf test and the location of a partial credit item."),
  andersen = list(
    andersen_case("raschdat1, median", rasch, "rasch", "median"),
    andersen_case("raschdat1, mean", rasch, "rasch", "mean"),
    andersen_case("raschdat1, three labels", rasch, "rasch", three_labels),
    andersen_case("raschdat1, sparse, median", sparse_rasch, "rasch", "median"),
    andersen_case("pcmdat2, median", partial, "partial-credit", "median"),
    andersen_case("rating scale, median", rating, "rating-scale", "median"),
    andersen_case("easy item, median", easy, "rasch", "median"),
    base_andersen()
  ),
  wald = list(
    wald_case("raschdat1, median", rasch, "rasch", "median"),
    wald_case("raschdat1, two labels", rasch, "rasch", two_labels),
    wald_case("raschdat1, sparse, median", sparse_rasch, "rasch", "median"),
    wald_case("pcmdat2, median", partial, "partial-credit", "median"),
    wald_case("rating scale, mean", rating, "rating-scale", "mean"),
    wald_case("easy item, median", easy, "rasch", "median")
  ),
  martinLof = list(
    martin_lof_case("raschdat1, median", rasch, "rasch", "median"),
    martin_lof_case("raschdat1, mean", rasch, "rasch", "mean"),
    martin_lof_case("raschdat1, odd and even", rasch, "rasch", halves),
    martin_lof_case("raschdat1, three sets", rasch, "rasch", thirds),
    martin_lof_case("raschdat1, sparse, median", sparse_rasch, "rasch", "median"),
    martin_lof_case("pcmdat2, median", partial, "partial-credit", "median"),
    martin_lof_case("rating scale, median", rating, "rating-scale", "median")
  )
))

# The tests' hand case of items left out: without item1, person 5 of group b has a total of 0 and item2 is 1 for the
# other persons of b. eRm checks each group once, over all its persons, so it leaves out item1 alone.
cascade <- with_ids(rbind(c(0, 1, 1, 0), c(1, 0, 0, 1), c(0, 0, 1, 0), c(1, 1, 0, 1), c(1, 0, 0, 0), c(1, 1, 1, 0),
                          c(1, 1, 0, 1)))
cascade_groups <- c("a", "a", "a", "a", "b", "b", "b")
cascade_excluded <- unique(unlist(lapply(c("a", "b"), function(level) {
  eRm:::datcheck.LRtest(cascade[cascade_groups == level, ], cascade, "RM")
})))
stopifnot(identical(colnames(cascade)[cascade_excluded], "item1"))
message("eRm leaves out ", colnames(cascade)[cascade_excluded], " of the hand case, where measurekit leaves out item1 and item2")
