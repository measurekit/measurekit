# Summed-score tables, conversion tables and conditional errors of scale scores under the testlet response model.
# References: base R's enumeration of every response pattern of a form, each testlet's effect integrated out of its
# items' pattern likelihood at each node of theta, the patterns' likelihoods then summed by total score; and mirt's
# fscores(method = "EAPsum_2.0"), the Lord-Wingersky algorithm 2.0 of Cai (2015), for the bifactor model whose specific
# slopes are a σ with specific variances 1, the testlet model of each form. Reads irt-testlet.json, of irt.R, for the
# items and testlet SDs that base R estimated on sirt's data.read.

source("validation/psychometrics/common.R")
suppressPackageStartupMessages(library(mirt))

POINTS <- 61 # measurekit's default rule: 61 equally spaced nodes on [-6, 6], weights ∝ φ, for theta and each effect
Z <- seq(-6, 6, length.out = POINTS)
W <- dnorm(Z) / sum(dnorm(Z))
RANGE <- c(-6, 6)

# --- Item probabilities -----------------------------------------------------------------------------------------------
# measurekit's IRT parameters, written out: the logistic models, the graded response model and the GPCM. A matrix of
# one row per theta and one column per category.

category_probabilities <- function(item, theta) {
  p <- item$parameters
  switch(item$model,
    "2pl" = , "3pl" = {
      lower <- if (is.null(p$c)) 0 else p$c
      upper <- if (is.null(p$d)) 1 else p$d
      correct <- lower + (upper - lower) * plogis(p$a * (theta - p$b))
      cbind(1 - correct, correct)
    },
    grm = {
      above <- cbind(1, matrix(sapply(p$b, function(b) plogis(p$a * (theta - b))), length(theta)), 0)
      above[, -ncol(above), drop = FALSE] - above[, -1, drop = FALSE]
    },
    gpcm = {
      steps <- matrix(sapply(p$b, function(b) p$a * (theta - b)), length(theta))
      if (ncol(steps) > 1) steps <- t(apply(steps, 1, cumsum))
      steps <- cbind(0, steps)
      exp(steps - apply(steps, 1, max)) / rowSums(exp(steps - apply(steps, 1, max)))
    }
  )
}
category_count <- function(item) ncol(category_probabilities(item, 0))

# --- Enumeration ------------------------------------------------------------------------------------------------------
# A unit is an item in no testlet or the items of a testlet on the form. Each of its response patterns has a total
# score and a likelihood at each node of theta: an item's category probability, or a testlet's Σ_g w_g Π_j
# P_j(x_j | θ + σ z_g). The form's patterns are all combinations of its units' patterns, and the probability of a total
# score at a node is the sum of the likelihoods of the patterns with that score.

unit_patterns <- function(items, sd, nodes) {
  patterns <- as.matrix(expand.grid(lapply(items, function(item) seq_len(category_count(item)) - 1)))
  pattern_likelihood <- function(theta) {
    probabilities <- lapply(items, function(item) category_probabilities(item, theta))
    Reduce(`*`, lapply(seq_along(items), function(j) t(probabilities[[j]][, patterns[, j] + 1, drop = FALSE])))
  }
  likelihoods <- if (is.na(sd)) {
    pattern_likelihood(nodes)
  } else {
    Reduce(`+`, lapply(seq_along(Z), function(g) W[g] * pattern_likelihood(nodes + sd * Z[g])))
  }
  list(scores = rowSums(patterns), likelihoods = likelihoods)
}

form_units <- function(form) {
  ids <- sapply(form$items, function(item) item$id)
  in_testlet <- ids %in% unlist(lapply(form$testlets, function(testlet) testlet$items))
  c(lapply(form$items[!in_testlet], function(item) list(items = list(item), sd = NA)),
    Filter(function(unit) length(unit$items) > 0, lapply(form$testlets, function(testlet) {
      list(items = form$items[ids %in% testlet$items], sd = testlet$sd)
    })))
}

score_likelihoods_enumerated <- function(form, nodes) {
  combined <- list(scores = 0, likelihoods = matrix(1, 1, length(nodes)))
  for (unit in form_units(form)) {
    patterns <- unit_patterns(unit$items, unit$sd, nodes)
    pairs <- expand.grid(left = seq_along(combined$scores), right = seq_along(patterns$scores))
    combined <- list(scores = combined$scores[pairs$left] + patterns$scores[pairs$right],
                     likelihoods = combined$likelihoods[pairs$left, , drop = FALSE] *
                       patterns$likelihoods[pairs$right, , drop = FALSE])
  }
  likelihoods <- rowsum(combined$likelihoods, combined$scores)
  stopifnot(all(as.numeric(rownames(likelihoods)) == seq_len(nrow(likelihoods)) - 1))
  unname(likelihoods)
}

summed_table <- function(form, prior) {
  nodes <- prior$mean + prior$sd * Z
  likelihoods <- score_likelihoods_enumerated(form, nodes)
  probability <- drop(likelihoods %*% W)
  theta <- drop(likelihoods %*% (W * nodes)) / probability
  sd <- sqrt(sapply(seq_along(theta), function(s) sum(likelihoods[s, ] * W * (nodes - theta[s])^2)) / probability)
  data.frame(score = seq_along(theta) - 1, theta = theta, standardError = sd, probability = probability)
}

# --- mirt's EAPsum_2.0 ------------------------------------------------------------------------------------------------
# The form as a bifactor model: general slope a, specific slope a σ of the item's testlet with variance 1, mirt's
# slope-intercept form d = -a b for the logistic and graded models and d_k = -a (b_1 + ... + b_k) for the GPCM. The
# items are set, not estimated, on data with every category.

mirt_eap_sum <- function(form, data) {
  ids <- sapply(form$items, function(item) item$id)
  specific <- rep(NA, length(ids))
  for (t in seq_along(form$testlets)) specific[ids %in% form$testlets[[t]]$items] <- t
  used <- sort(unique(na.omit(specific)))
  specific <- match(specific, used)
  itemtype <- sapply(form$items, function(item) c("2pl" = "2PL", "3pl" = "3PL", grm = "graded", gpcm = "gpcm")[[item$model]])
  values <- bfactor(data[, ids], specific, itemtype = itemtype, pars = "values")
  set <- function(id, name, value) values$value[values$item == id & values$name == name] <<- value
  for (j in seq_along(ids)) {
    item <- form$items[[j]]
    p <- item$parameters
    set(ids[j], "a1", p$a)
    if (!is.na(specific[j])) set(ids[j], paste0("a", specific[j] + 1), p$a * form$testlets[[used[specific[j]]]]$sd)
    if (item$model %in% c("2pl", "3pl")) set(ids[j], "d", -p$a * p$b)
    if (item$model == "3pl") set(ids[j], "g", p$c)
    if (item$model == "grm") for (k in seq_along(p$b)) set(ids[j], paste0("d", k), -p$a * p$b[k])
    if (item$model == "gpcm") for (k in seq_along(p$b)) set(ids[j], paste0("d", k), -p$a * sum(p$b[1:k]))
  }
  fit <- bfactor(data[, ids], specific, itemtype = itemtype, pars = values, TOL = NaN, verbose = FALSE)
  fscores(fit, method = "EAPsum_2.0", full.scores = FALSE, quadpts = POINTS, theta_lim = c(-6, 6), verbose = FALSE)
}

# --- Forms ------------------------------------------------------------------------------------------------------------
# data.read's twelve items as base R calibrated them under the testlet model; a form of seven of them, whose testlet B
# has one item on the form; eight 2PL items with two testlets and three items in none; and a form of mixed models, a 3PL
# item in no testlet, three graded items in one testlet and two GPCM items in another.

testlet_fixture <- fromJSON("packages/psychometrics/test/fixtures/irt-testlet.json", simplifyVector = FALSE)
read <- Filter(function(case) startsWith(case$name, "data.read"), testlet_fixture$cases)[[1]]
read_items <- lapply(seq_along(read$itemIds), function(j) {
  p <- read$items[[j]]$parameters
  list(id = read$itemIds[[j]], model = "2pl", parameters = list(a = p$a, b = p$b, c = 0, d = 1))
})
read_testlets <- lapply(seq_along(read$testlets), function(t) {
  list(items = unlist(read$testlets[[t]]), sd = read$sd[[t]])
})
read_data <- do.call(rbind, lapply(read$responses, unlist))
colnames(read_data) <- unlist(read$itemIds)

logistic <- function(id, a, b) list(id = id, model = "2pl", parameters = list(a = a, b = b, c = 0, d = 1))
forms <- list(
  list(name = "data.read, 2PL in three testlets", items = read_items, testlets = read_testlets),
  list(name = "seven data.read items, one alone in its testlet",
       items = read_items[match(c("A1", "A2", "A3", "B1", "C1", "C2", "C4"), unlist(read$itemIds))],
       testlets = read_testlets),
  list(name = "eight 2PL items, two testlets",
       items = list(logistic("i1", 1.2, -0.5), logistic("i2", 0.8, 0.4), logistic("i3", 1.5, 1.1),
                    logistic("t1a", 1.1, -1), logistic("t1b", 0.9, 0), logistic("t1c", 1.4, 0.6),
                    logistic("t2a", 1.3, -0.3), logistic("t2b", 0.7, 0.9)),
       testlets = list(list(items = c("t1a", "t1b", "t1c"), sd = 0.8), list(items = c("t2a", "t2b"), sd = 1.4))),
  list(name = "mixed models, 3PL, graded and GPCM",
       items = list(list(id = "m1", model = "3pl", parameters = list(a = 1.4, b = 0.2, c = 0.2, d = 1)),
                    list(id = "g1", model = "grm", parameters = list(a = 1.1, b = I(c(-0.8, 0.5)))),
                    list(id = "g2", model = "grm", parameters = list(a = 0.9, b = I(c(-0.2, 1)))),
                    list(id = "g3", model = "grm", parameters = list(a = 1.6, b = I(c(-1.2, 0.1)))),
                    list(id = "p1", model = "gpcm", parameters = list(a = 0.8, b = I(c(-0.5, 0.3, 1.2)))),
                    list(id = "p2", model = "gpcm", parameters = list(a = 1.2, b = I(c(0, 0.8))))),
       testlets = list(list(items = c("g1", "g2", "g3"), sd = 1), list(items = c("p1", "p2"), sd = 0.6)))
)

# Data with every category of each item, for mirt to set the items on.
category_data <- function(form) {
  set.seed(20261104)
  data <- sapply(form$items, function(item) sample(rep_len(seq_len(category_count(item)) - 1, 300)))
  colnames(data) <- sapply(form$items, function(item) item$id)
  data
}

standard_prior <- list(mean = 0, sd = 1)
shifted_prior <- list(mean = 0.4, sd = 1.3)
tables <- lapply(forms, function(form) {
  standard <- summed_table(form, standard_prior)
  data <- if (startsWith(form$name, "data.read")) read_data else category_data(form)
  reference <- mirt_eap_sum(form, data)
  difference <- max(abs(standard$theta - reference[[2]]), abs(standard$standardError - reference[[3]]))
  message(form$name, ": base R's enumeration and mirt's EAPsum_2.0 differ by ", signif(difference, 3))
  stopifnot(nrow(reference) == nrow(standard), difference < 1e-9)
  list(form = form$name,
       tables = list(list(prior = standard_prior, estimates = standard),
                     list(prior = shifted_prior, estimates = summed_table(form, shifted_prior))))
})

form_fields <- lapply(forms, function(form) {
  list(name = form$name, items = form$items, testlets = form$testlets)
})

write_fixture("irt-testlet-tables", list(
  source = paste("Summed-score tables under the testlet model: base R's enumeration of every response pattern, each",
                 "testlet's effect integrated out on 61 rectangular nodes, the patterns' likelihoods summed by total",
                 "score; mirt 1.48's fscores(method = 'EAPsum_2.0') on the equivalent bifactor model agrees within 1e-9",
                 "under N(0, 1). data.read's items as irt.R's base R maximum estimates them."),
  forms = form_fields,
  tables = tables
))

# --- Conversion tables and conditional errors -------------------------------------------------------------------------
# The theta of each total score is the EAP of the summed score above; the scale score rounds and bounds it; the error
# of the table's scale scores at a theta is their standard deviation under the enumerated P(x | θ).

scales <- list(
  list(slope = 100, intercept = 500, lowest = 200, highest = 800, step = 10),
  list(slope = 100, intercept = 500, lowest = 200, highest = 800, step = 0)
)
scale_field <- function(scale, name, default) if (is.null(scale[[name]])) default else scale[[name]]
reported <- function(x, scale) {
  step <- scale_field(scale, "step", 1)
  rounded <- if (step > 0) floor(x / step + 0.5) * step else x
  pmin(pmax(rounded, scale_field(scale, "lowest", -Inf)), scale_field(scale, "highest", Inf))
}
scale_errors <- function(form, conversion, thetas) {
  likelihoods <- score_likelihoods_enumerated(form, thetas)
  sapply(seq_along(thetas), function(j) {
    p <- likelihoods[, j]
    sqrt(sum(p * (conversion - sum(p * conversion))^2))
  })
}
conversion_case <- function(form, prior, scale) {
  table <- summed_table(form, prior)
  stopifnot(all(is.finite(table$theta)))
  scale_scores <- reported(scale$slope * table$theta + scale$intercept, scale)
  list(form = form$name, prior = prior, scale = scale,
       rows = data.frame(score = table$score, theta = table$theta, standardError = table$standardError,
                         scaleScore = scale_scores, scaleStandardError = scale_errors(form, scale_scores, table$theta)))
}

write_fixture("scaling-testlet-tables", list(
  source = paste("Conversion tables and conditional errors of scale scores under the testlet model: the EAP of the",
                 "summed score and P(x | θ) by base R's enumeration of every response pattern, each testlet's effect",
                 "integrated out on 61 rectangular nodes; the forms of irt-testlet-tables.json."),
  forms = form_fields,
  tables = c(
    unlist(lapply(forms, function(form) lapply(scales, function(scale) conversion_case(form, standard_prior, scale))),
           recursive = FALSE),
    list(conversion_case(forms[[4]], shifted_prior, scales[[1]]))
  ),
  conditionalErrors = lapply(forms, function(form) {
    thetas <- c(-3, -0.5, 0, 1.25, 4)
    conversion <- conversion_case(form, standard_prior, scales[[1]])$rows$scaleScore
    list(form = form$name, conversion = conversion, thetas = thetas, errors = scale_errors(form, conversion, thetas))
  })
))
