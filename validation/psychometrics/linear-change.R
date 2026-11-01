# Linear logistic models of change (LLRA) by conditional maximum likelihood. With two occasions and responses of 0 and 1,
# only the persons who changed on an item inform its effect, and the effects are the coefficients of a logistic
# regression of the direction of the change on the design: glm on the changers, with a column for each effect, is the
# reference, and eRm's LLRA, which takes no sets of items that share an effect (its `itmgrps` is documented as not
# functional), agrees on the cases it takes. For three occasions, and for missing responses, the conditional likelihood is
# written out in base R over the patterns of each person's responses to an item, as the model defines it, and eRm's LLRA
# is the reference where it applies.

source("validation/psychometrics/common.R")
suppressPackageStartupMessages({
  library(eRm)
  library(numDeriv)
})

# --- Simulation -------------------------------------------------------------------------------------------------------
# Each person has a trait on each item, theta ~ N(0, 1), which the model conditions away. The logit of a correct response
# adds, from the second occasion, the trend of the item and the effect of the person's group: effects[[group]][i, t - 1].

simulate_change <- function(persons, items, occasions, groups, trend, effects, seed, missing = 0) {
  set.seed(seed)
  group <- sample(groups, persons, replace = TRUE)
  responses <- lapply(seq_len(occasions), function(t) {
    theta <- matrix(rnorm(persons * items), persons, items)
    shift <- if (t == 1) 0 else outer(rep(1, persons), trend[, t - 1]) +
      do.call(rbind, lapply(group, function(g) effects[[g]][, t - 1]))
    x <- matrix(rbinom(persons * items, 1, plogis(theta + shift)), persons, items)
    colnames(x) <- sprintf("q%02d", seq_len(items))
    x
  })
  # Each response is missing at random with the probability `missing`.
  if (missing > 0) {
    responses <- lapply(responses, function(x) {
      x[matrix(runif(length(x)) < missing, nrow(x))] <- NA
      x
    })
  }
  list(responses = responses, group = group)
}

# --- The conditional likelihood in base R -------------------------------------------------------------------------------
# The effects are those of the item sets, the occasions after the first and the groups, the baseline first. A pair of a
# person and an item that answered at every occasion has the pattern y; its sum s fixes the patterns that it could have
# had, and the probability of y among them is exp(z(y)' eta) over the sum of that for the patterns of sum s, with z(y)
# the sum of the design columns of the occasions at which y is 1.

change_design <- function(sets, occasions, groups) {
  list(sets = sets, occasions = occasions, groups = groups,
       index = function(k, t, g) ((k - 1) * (occasions - 1) + (t - 2)) * length(groups) + g)
}

column <- function(design, k, t, slot) {
  w <- numeric(length(design$sets) * (design$occasions - 1) * length(design$groups))
  if (t >= 2) {
    w[design$index(k, t, 1)] <- 1
    if (slot > 1) w[design$index(k, t, slot)] <- 1
  }
  w
}

conditional_fit <- function(responses, group, baseline, item_sets) {
  occasions <- length(responses)
  persons <- nrow(responses[[1]])
  items <- ncol(responses[[1]])
  groups <- c(baseline, setdiff(unique(group), baseline))
  sets <- unique(item_sets)
  design <- change_design(sets, occasions, groups)
  size <- length(sets) * (occasions - 1) * length(groups)
  patterns <- as.matrix(expand.grid(rep(list(0:1), occasions)))
  # One row for each stratum: the item set, the group, the pattern and how many pairs of a person and an item gave it.
  counts <- list()
  for (p in seq_len(persons)) {
    for (i in seq_len(items)) {
      y <- vapply(responses, function(x) x[p, i], numeric(1))
      if (anyNA(y) || sum(y) == 0 || sum(y) == occasions) next
      key <- paste(match(item_sets[i], sets), match(group[p], groups), paste(y, collapse = ""))
      counts[[key]] <- c(k = match(item_sets[i], sets), g = match(group[p], groups), y = y,
                         n = 1 + (if (is.null(counts[[key]])) 0 else counts[[key]][["n"]]))
    }
  }
  cells <- do.call(rbind, counts)
  loglik <- function(eta) {
    total <- 0
    for (r in seq_len(nrow(cells))) {
      k <- cells[r, "k"]; slot <- cells[r, "g"]
      y <- cells[r, paste0("y", seq_len(occasions))]
      same <- patterns[rowSums(patterns) == sum(y), , drop = FALSE]
      logits <- apply(same, 1, function(pattern) sum(vapply(seq_len(occasions), function(t) pattern[t] * sum(column(design, k, t, slot) * eta), numeric(1))))
      observed <- sum(vapply(seq_len(occasions), function(t) y[t] * sum(column(design, k, t, slot) * eta), numeric(1)))
      total <- total + cells[r, "n"] * (observed - (max(logits) + log(sum(exp(logits - max(logits))))))
    }
    total
  }
  fit <- optim(rep(0, size), function(eta) -loglik(eta), method = "BFGS", control = list(reltol = 1e-15, maxit = 5000))
  # Newton steps from the BFGS result take the gradient to rounding.
  eta <- fit$par
  for (step in 1:5) eta <- eta - solve(hessian(function(e) -loglik(e), eta), grad(function(e) -loglik(e), eta))
  covariance <- solve(hessian(function(e) -loglik(e), eta))
  list(eta = eta, standardErrors = sqrt(diag(covariance)), logLikelihood = loglik(eta), design = design)
}

# The effects of a fit in the order that measurekit lists them: item sets, then occasions from the second, then groups.
effects_of <- function(fit) {
  design <- fit$design
  rows <- list()
  for (k in seq_along(design$sets)) for (t in 2:design$occasions) for (g in seq_along(design$groups)) {
    j <- design$index(k, t, g)
    rows[[length(rows) + 1]] <- list(itemGroup = design$sets[k], occasion = t, group = design$groups[g],
                                     estimate = fit$eta[j], standardError = fit$standardErrors[j])
  }
  rows
}

# --- glm on the changers: two occasions ----------------------------------------------------------------------------------
# The changers are the persons whose responses to an item differ; the response at the second occasion is 1 for a gain. The
# columns are the design columns of the second occasion.

glm_fit <- function(responses, group, baseline, item_sets) {
  groups <- c(baseline, setdiff(unique(group), baseline))
  sets <- unique(item_sets)
  design <- change_design(sets, 2, groups)
  rows <- list()
  for (p in seq_len(nrow(responses[[1]]))) for (i in seq_len(ncol(responses[[1]]))) {
    y <- c(responses[[1]][p, i], responses[[2]][p, i])
    if (!anyNA(y) && y[1] != y[2]) {
      rows[[length(rows) + 1]] <- c(response = y[2], column(design, match(item_sets[i], sets), 2, match(group[p], groups)))
    }
  }
  data <- do.call(rbind, rows)
  fit <- glm(data[, 1] ~ 0 + data[, -1], family = binomial, control = glm.control(epsilon = 1e-14, maxit = 100))
  summary <- summary(fit)$coefficients
  list(eta = unname(coef(fit)), standardErrors = unname(summary[, 2]), logLikelihood = as.numeric(logLik(fit)),
       design = design)
}

# --- eRm's LLRA -------------------------------------------------------------------------------------------------------
# eRm names the trend `trend.<item>.t<occasion>` and a group's effect `<group>.<item>.t<occasion>`, with items I1, I2, ...

erm_fit <- function(responses, group, baseline) {
  occasions <- length(responses)
  items <- ncol(responses[[1]])
  X <- do.call(cbind, responses)
  colnames(X) <- paste0(rep(paste0("I", seq_len(items)), occasions), ".t", rep(seq_len(occasions), each = items))
  fit <- LLRA(X, mpoints = occasions, groups = group, baseline = baseline)
  list(eta = fit$etapar, standardErrors = setNames(fit$se.eta, names(fit$etapar)), logLikelihood = fit$loglik)
}

# eRm's effects in the order of effects_of, one item to a set.
erm_in_order <- function(erm, fit) {
  design <- fit$design
  vapply(effects_of(fit), function(effect) {
    k <- match(effect$itemGroup, design$sets)
    name <- sprintf("%s.I%d.t%d", if (effect$group == design$groups[1]) "trend" else effect$group, k, effect$occasion)
    c(estimate = erm$eta[[name]], standardError = erm$standardErrors[[name]])
  }, numeric(2))
}

# --- The cases ----------------------------------------------------------------------------------------------------------

compare_close <- function(effects, estimates, standard_errors, tolerance, what) {
  stopifnot(
    max(abs(vapply(effects, function(e) e$estimate, 0) - estimates)) < tolerance,
    max(abs(vapply(effects, function(e) e$standardError, 0) / standard_errors - 1)) < tolerance
  )
  message(what, ": largest difference of the estimates ",
          signif(max(abs(vapply(effects, function(e) e$estimate, 0) - estimates)), 3))
}

change_case <- function(name, data, occasions, baseline, item_sets, groups = TRUE, use_glm = occasions == 2, use_erm = TRUE) {
  responses <- data$responses
  group <- if (groups) data$group else rep("all", nrow(responses[[1]]))
  baseline <- if (groups) baseline else "all"
  base <- conditional_fit(responses, group, baseline, item_sets)
  effects <- effects_of(base)
  reference <- base
  source <- "base R"
  if (use_glm) {
    reference <- glm_fit(responses, group, baseline, item_sets)
    source <- "glm"
    stopifnot(max(abs(reference$eta - base$eta)) < 1e-6, abs(reference$logLikelihood - base$logLikelihood) < 1e-8)
    effects <- effects_of(reference)
  }
  if (use_erm) {
    erm <- erm_fit(responses, group, baseline)
    in_order <- erm_in_order(erm, reference)
    compare_close(effects, in_order["estimate", ], in_order["standardError", ], 1e-4, paste(name, "against eRm"))
  }
  list(name = name, occasions = lapply(responses, function(x) unname(x)),
       itemIds = colnames(responses[[1]]),
       groups = if (groups) I(group), baseline = if (groups) baseline,
       itemGroups = if (length(unique(item_sets)) < length(item_sets)) setNames(as.list(item_sets), colnames(responses[[1]])),
       reference = source, effects = effects, logLikelihood = reference$logLikelihood)
}

items8 <- sprintf("q%02d", 1:8)
trend8 <- matrix(c(0.5, 0.2, -0.3, 0.8, -0.6, 0.1, 0.4, -0.1), 8, 1)
treatment8 <- list(ctl = matrix(0, 8, 1), trt = matrix(c(0.7, 0, -0.4, 0.3, 0.5, -0.2, 0, 0.6), 8, 1))
two <- simulate_change(2000, 8, 2, c("ctl", "trt"), trend8, treatment8, 20261029)

three_groups <- simulate_change(1500, 5, 2, c("a", "b", "c"), matrix(c(0.4, -0.3, 0.2, 0.6, 0), 5, 1),
                                list(a = matrix(c(0.2, 0, -0.2, 0.3, 0.1), 5, 1), b = matrix(0, 5, 1),
                                     c = matrix(c(-0.3, 0.4, 0.1, 0, -0.5), 5, 1)), 20261030)

trend4 <- cbind(c(0.5, -0.2, 0.3, 0.1), c(0.9, -0.5, 0.2, 0.4))
treatment4 <- list(ctl = matrix(0, 4, 2), trt = cbind(c(0.4, 0, -0.3, 0.2), c(0.8, 0.3, -0.5, 0.1)))
three <- simulate_change(1500, 4, 3, c("ctl", "trt"), trend4, treatment4, 20261031)
three_missing <- simulate_change(1500, 4, 3, c("ctl", "trt"), trend4, treatment4, 20261031, missing = 0.08)

shared_trend <- matrix(rep(c(0.5, -0.4, 0.3, 0.1), each = 2), 8, 1)
shared_effect <- list(ctl = matrix(0, 8, 1), trt = matrix(rep(c(0.6, -0.2, 0.4, 0), each = 2), 8, 1))
shared <- simulate_change(2000, 8, 2, c("ctl", "trt"), shared_trend, shared_effect, 20261101)

missing_two <- simulate_change(1500, 6, 2, c("ctl", "trt"), matrix(c(0.5, 0.2, -0.3, 0.8, -0.6, 0.1), 6, 1),
                               list(ctl = matrix(0, 6, 1), trt = matrix(c(0.7, 0, -0.4, 0.3, 0.5, -0.2), 6, 1)),
                               20261102, missing = 0.1)

one_group <- simulate_change(800, 5, 2, "ctl", matrix(c(0.4, -0.3, 0.2, 0.6, 0), 5, 1), list(ctl = matrix(0, 5, 1)),
                             20261103)

cases <- list(
  change_case("two occasions, a trend and a treatment effect for each of 8 items", two, 2, "ctl", items8),
  change_case("three groups, the second the baseline", three_groups, 2, "b", sprintf("q%02d", 1:5)),
  change_case("three occasions", three, 3, "ctl", sprintf("q%02d", 1:4)),
  change_case("three occasions with 8% of the responses missing", three_missing, 3, "ctl", sprintf("q%02d", 1:4),
              use_erm = FALSE),
  change_case("four sets of two items that share their effects", shared, 2, "ctl", rep(c("s1", "s2", "s3", "s4"), each = 2),
              use_erm = FALSE),
  change_case("two occasions with 10% of the responses missing", missing_two, 2, "ctl", sprintf("q%02d", 1:6),
              use_erm = FALSE),
  change_case("one group of everyone", one_group, 2, "ctl", sprintf("q%02d", 1:5), groups = FALSE)
)

write_fixture("explanatory-linear-change", list(
  source = paste("Simulated changes (set.seed(20261029) and the five seeds that follow): a trait for each person and item,",
                 "N(0, 1), conditioned away. The effects of two occasions are glm on the persons who changed;",
                 "those of three occasions and of missing responses are the conditional likelihood written out in base R;",
                 "eRm's LLRA agrees to 1e-4 where it applies."),
  cases = cases
))
