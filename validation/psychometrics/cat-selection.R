# Item selection rules of adaptive tests beyond MFI, KL and thOpt (cat-selection.json), on the banks and runs of
# cat-3pl.json and cat-gpcm.json, which cat.R writes: the script reads them and writes expected values only.
#
# - Scores of every eligible item after the first k responses of a run, at catR's estimate there: catR 3.17's MWI
#   (MLWI, MPWI) and KL (KLP) divided by the largest likelihood, times the prior's density for MPWI and KLP, on their
#   grid, as measurekit scales them; GDI (GDI, GDIP); minus EPV (MEPV); minus the distance of bOpt, with the mean
#   threshold for polytomous items, which bOpt refuses.
# - MEI written out in base R: catR's MEI finds each estimate after the next response with uniroot's default tolerance,
#   about 1.2e-4, so the script finds the MAP estimates to 1e-14, takes catR's OIi and Ii at them, and checks catR's MEI
#   within 1e-3.
# - Tests replayed step by step: the starting items, then the item nextItem selects at the MAP estimate, found to 1e-14,
#   with each response drawn once from a true theta with a fixed seed. The script checks that the highest score of the
#   written-out rule, the first on ties, is nextItem's item.
# - Progressive and proportional selection: catR's weights for a planned length, the scores of the progressive rule
#   for uniform draws runif gives (nextItem's own, under the same seed), the proportional rule's probabilities and its
#   exponential race on the same draws; and for a pool of 10 items, the probability that each item is selected, by
#   integrate for the progressive rule.

source("validation/psychometrics/common.R")
suppressPackageStartupMessages(library(catR))

KL_GRID <- c(-4, 4, 33) # catR's default parInt, the grid of measurekit's weighted rules
X <- seq(KL_GRID[1], KL_GRID[2], length = KL_GRID[3])
RANGE <- c(-6, 6) # measurekit's default theta range for MAP
STANDARD <- c(0, 1)
SHIFTED <- c(0.5, 1.3) # a prior N(0.5, 1.3²) for the rules that take one
INTEGRATION <- 1e-13
SEED <- 20261102

read_fixture <- function(name) {
  jsonlite::fromJSON(file.path("packages/psychometrics/test/fixtures", paste0(name, ".json")), simplifyVector = FALSE)
}
fixture3pl <- read_fixture("cat-3pl")
fixtureGpcm <- read_fixture("cat-gpcm")

# catR's banks of the fixtures' items: a, b, c, d for the 3PL; the slope and up to three thresholds for the GPCM.
bank3pl <- t(sapply(fixture3pl$items, function(item) c(a = item$a, b = item$b, c = item$c, d = item$d)))
bankGpcm <- t(sapply(fixtureGpcm$items, function(item) c(item$a, unlist(item$b), rep(NA, 4 - length(item$b)))[1:4]))
BANKS <- list("cat-3pl" = list(bank = bank3pl, model = NULL, fixture = fixture3pl),
              "cat-gpcm" = list(bank = bankGpcm, model = "GPCM", fixture = fixtureGpcm))

categories <- function(bank, model) {
  if (is.null(model)) rep(2, nrow(bank)) else rowSums(!is.na(bank[, -1, drop = FALSE])) + 1
}

# Each category's probability, first derivative and second derivative at theta, items by row.
trace_of <- function(theta, bank, model) {
  pr <- Pi(theta, bank, model = model)
  if (is.null(model)) {
    list(p = cbind(1 - pr$Pi, pr$Pi), d1 = cbind(-pr$dPi, pr$dPi), d2 = cbind(-pr$d2Pi, pr$d2Pi))
  } else {
    list(p = rbind(pr$Pi), d1 = rbind(pr$dPi), d2 = rbind(pr$d2Pi))
  }
}

likelihood <- function(bank, out, x, model) {
  sapply(X, function(theta) {
    if (length(out) == 0) return(1)
    p <- trace_of(theta, bank[out, , drop = FALSE], model)$p
    prod(p[cbind(seq_along(x), x + 1)])
  })
}

# The MAP estimate under a normal prior in RANGE: the largest of the posterior's maxima, each a root of its slope
# found by uniroot to 1e-14 between points of a grid 0.01 apart, or an end of the range.
map_estimate <- function(items, x, model, prior = STANDARD) {
  log_posterior <- function(theta) {
    term <- if (length(x) == 0) 0 else {
      p <- trace_of(theta, items, model)$p
      sum(log(p[cbind(seq_along(x), x + 1)]))
    }
    term + dnorm(theta, prior[1], prior[2], log = TRUE)
  }
  slope <- function(theta) {
    term <- if (length(x) == 0) 0 else {
      tr <- trace_of(theta, items, model)
      index <- cbind(seq_along(x), x + 1)
      sum(tr$d1[index] / tr$p[index])
    }
    term + (prior[1] - theta) / prior[2]^2
  }
  grid <- seq(RANGE[1], RANGE[2], by = 0.01)
  slopes <- sapply(grid, slope)
  candidates <- c(if (slopes[1] <= 0) RANGE[1], if (slopes[length(grid)] >= 0) RANGE[2])
  for (j in which(slopes[-length(grid)] > 0 & slopes[-1] <= 0)) {
    candidates <- c(candidates, if (slopes[j + 1] == 0) grid[j + 1] else
      uniroot(slope, grid[j:(j + 1)], tol = 1e-14)$root)
  }
  candidates[which.max(sapply(candidates, log_posterior))]
}

# MEI written out: Σ_k P_k(θ̂) I(θ̂_k), θ̂_k the MAP estimate after the response k to the item, I observed or Fisher.
mei_score <- function(bank, model, item, out, x, theta, info_type) {
  row <- bank[item, , drop = FALSE]
  count <- categories(row, model)
  probabilities <- trace_of(theta, row, model)$p[1, seq_len(count)]
  information <- sapply(seq_len(count) - 1, function(k) {
    estimate <- map_estimate(bank[c(out, item), , drop = FALSE], c(x, k), model)
    if (info_type == "Fisher") Ii(estimate, row, model = model)$Ii else OIi(estimate, row, k, model = model)
  })
  sum(probabilities * information)
}

difficulty <- function(bank, model) if (is.null(model)) bank[, "b"] else rowMeans(bank[, -1, drop = FALSE], na.rm = TRUE)

# The score of each item by a rule after the items `out` with responses `x`, at `theta`; NA for administered items.
rule_scores <- function(bank, model, rule, out, x, theta, prior = STANDARD, info_type = "observed") {
  given <- bank[out, , drop = FALSE]
  answers <- if (length(x) == 0) NULL else x
  lik <- likelihood(bank, out, x, model)
  posterior <- lik * dnorm(X, prior[1], prior[2])
  scores <- sapply(seq_len(nrow(bank)), function(i) {
    switch(rule,
      MLWI = MWI(bank, i, answers, given, model = model, lower = KL_GRID[1], upper = KL_GRID[2], nqp = KL_GRID[3],
                 type = "MLWI") / max(lik),
      MPWI = MWI(bank, i, answers, given, model = model, lower = KL_GRID[1], upper = KL_GRID[2], nqp = KL_GRID[3],
                 type = "MPWI", priorPar = prior) / max(posterior),
      KLP = KL(bank, i, answers, given, model = model, theta = theta, type = "KLP", priorPar = prior, X = X,
               lik = lik, nqp = KL_GRID[3]) / max(posterior),
      GDI = GDI(bank, i, answers, given, model = model, type = "GDI", X = X, lik = lik, nqp = KL_GRID[3]),
      GDIP = GDI(bank, i, answers, given, model = model, type = "GDIP", priorPar = prior, X = X, lik = lik,
                 nqp = KL_GRID[3]),
      MEPV = -EPV(bank, i, answers, theta, given, model = model, priorPar = prior, parInt = KL_GRID),
      MEI = mei_score(bank, model, i, out, x, theta, info_type),
      bOpt = -abs(difficulty(bank, model)[i] - theta)
    )
  })
  if (rule == "MEI") {
    catr <- sapply(seq_len(nrow(bank)), function(i) {
      MEI(bank, i, answers, theta, given, model = model, method = "BM", priorPar = prior, range = RANGE,
          parInt = KL_GRID, infoType = info_type)
    })
    stopifnot(max(abs(catr - scores) / pmax(1, abs(scores))) < 1e-3)
  }
  scores[out] <- NA
  scores
}

# --- Scores ---------------------------------------------------------------------------------------------------------

item_index <- function(fixture, id) which(sapply(fixture$items, `[[`, "id") == id)

score_case <- function(bank_name, run_name, step, rule, prior = NULL, info_type = NULL) {
  source <- BANKS[[bank_name]]
  run <- Find(function(r) r$name == run_name, source$fixture$runs)
  steps <- run$steps[seq_len(step)]
  out <- vapply(steps, function(s) item_index(source$fixture, s$nextItem), integer(1))
  x <- vapply(steps, function(s) s$response, numeric(1))
  theta <- run$steps[[step + 1]]$theta
  options <- c(if (!is.null(prior)) list(prior = list(mean = prior[1], sd = prior[2])),
               if (!is.null(info_type)) list(information = tolower(info_type)))
  scores <- rule_scores(source$bank, source$model, rule, out, x, theta,
                        if (is.null(prior)) STANDARD else prior, if (is.null(info_type)) "observed" else info_type)
  list(name = paste0(rule, if (!is.null(prior)) " with a prior N(0.5, 1.3)",
                     if (!is.null(info_type)) paste0(", ", info_type, " information"), ": ", bank_name, ", ", run_name,
                     ", after ", step, " items"),
       fixture = bank_name, run = run_name, step = step, rule = rule,
       options = if (length(options) == 0) setNames(list(), character(0)) else options, theta = theta, scores = scores)
}

DICHOTOMOUS_RULES <- c("MLWI", "MPWI", "KLP", "GDI", "GDIP", "MEPV", "MEI", "bOpt")
POLYTOMOUS_RULES <- c("MLWI", "MPWI", "KLP", "MEPV", "MEI", "bOpt")
scores <- c(
  unlist(lapply(c(0, 1, 6), function(step) lapply(DICHOTOMOUS_RULES, function(rule) {
    score_case("cat-3pl", "MFI with MAP", step, rule)
  })), recursive = FALSE),
  lapply(DICHOTOMOUS_RULES, function(rule) score_case("cat-3pl", "KL with EAP until mixed, then WLE", 10, rule)),
  lapply(c("MPWI", "KLP", "GDIP", "MEPV"), function(rule) score_case("cat-3pl", "MFI with MAP", 6, rule, SHIFTED)),
  list(score_case("cat-3pl", "MFI with MAP", 6, "MEI", info_type = "Fisher")),
  unlist(lapply(c(0, 3, 9), function(step) lapply(POLYTOMOUS_RULES, function(rule) {
    score_case("cat-gpcm", "MFI with MAP", step, rule)
  })), recursive = FALSE),
  lapply(c("MPWI", "KLP", "MEPV"), function(rule) score_case("cat-gpcm", "MFI with MAP", 9, rule, SHIFTED)),
  list(score_case("cat-gpcm", "MFI with MAP", 9, "MEI", info_type = "Fisher"))
)
names(scores) <- sapply(scores, `[[`, "name")

# --- Replays with starting items -------------------------------------------------------------------------------------

LENGTH <- 12

replay <- function(bank_name, rule, start, true_theta, seed, prior = STANDARD) {
  source <- BANKS[[bank_name]]
  bank <- source$bank
  model <- source$model
  count <- categories(bank, model)
  set.seed(seed)
  draws <- runif(LENGTH)
  out <- integer(0)
  x <- numeric(0)
  for (k in seq_len(LENGTH)) {
    item <- if (k <= length(start)) start[k] else {
      theta <- map_estimate(bank[out, , drop = FALSE], x, model)
      written <- rule_scores(bank, model, rule, out, x, theta, prior)
      chosen <- which.max(written)
      if (rule != "bOpt" || is.null(model)) {
        selection <- nextItem(bank, model = model, theta = theta, out = out, x = x, criterion = rule, method = "BM",
                              priorPar = prior, range = RANGE, parInt = KL_GRID, random.seed = seed + k)
        stopifnot(selection$item == chosen)
      }
      chosen
    }
    p <- trace_of(true_theta, bank[item, , drop = FALSE], model)$p[1, seq_len(count[item])]
    response <- sum(draws[k] > cumsum(p)[-count[item]])
    out <- c(out, item)
    x <- c(x, response)
  }
  ids <- sapply(BANKS[[bank_name]]$fixture$items, `[[`, "id")
  list(name = paste0(rule, ": ", bank_name, ", from ", length(start), " starting items"), fixture = bank_name,
       rule = rule, trueTheta = true_theta, starting = ids[start], items = ids[out], responses = x)
}

replays <- c(
  lapply(seq_along(DICHOTOMOUS_RULES), function(r) {
    replay("cat-3pl", DICHOTOMOUS_RULES[r], c(17, 42, 88), c(-0.8, 0.4, 1.3)[(r %% 3) + 1], SEED + 10 * r)
  }),
  lapply(seq_along(POLYTOMOUS_RULES), function(r) {
    replay("cat-gpcm", POLYTOMOUS_RULES[r], c(5, 22), c(-0.6, 0.9)[(r %% 2) + 1], SEED + 100 + 10 * r)
  })
)
names(replays) <- sapply(replays, `[[`, "name")

# --- Progressive and proportional selection -------------------------------------------------------------------------

# catR's weights for the k-th selection of a test of planned length Q: w of progressive, and Q w the exponent of
# proportional.
progressive_weight <- function(k, length, acceleration) {
  if (k == 0) 0 else sum((1:k)^acceleration) / sum((1:(length - 1))^acceleration)
}

random_case <- function(bank_name, run_name, step, length, acceleration, seed) {
  source <- BANKS[[bank_name]]
  bank <- source$bank
  run <- Find(function(r) r$name == run_name, source$fixture$runs)
  steps <- run$steps[seq_len(step)]
  out <- vapply(steps, function(s) item_index(source$fixture, s$nextItem), integer(1))
  x <- vapply(steps, function(s) s$response, numeric(1))
  theta <- run$steps[[step + 1]]$theta
  info <- Ii(theta, bank, model = source$model)$Ii
  eligible <- setdiff(seq_len(nrow(bank)), out)
  largest <- max(info[eligible])
  w <- progressive_weight(step, length, acceleration)
  set.seed(seed)
  u <- runif(nrow(bank))
  progressive <- info * w + u * largest * (1 - w)
  set.seed(seed)
  selection <- nextItem(bank, model = source$model, theta = theta, out = if (length(out) > 0) out,
                        x = if (length(x) > 0) x, criterion = "progressive", rule = "length", thr = length,
                        AP = acceleration, random.seed = seed)
  stopifnot(selection$item == eligible[which.max(progressive[eligible])])
  exponent <- length * w
  weights <- info^exponent
  probabilities <- weights / sum(weights[eligible])
  race <- exponent * log(info) - log(-log(1 - u))
  progressive[out] <- NA
  probabilities[out] <- NA
  race[out] <- NA
  list(name = paste0(bank_name, ", ", run_name, ", after ", step, " items, length ", length, ", acceleration ",
                     acceleration),
       fixture = bank_name, run = run_name, step = step, length = length, acceleration = acceleration, theta = theta,
       weight = w, uniforms = u[eligible], progressive = progressive, proportional = probabilities, race = race)
}

random_cases <- list(
  random_case("cat-3pl", "MFI with MAP", 0, 12, 1, SEED + 1),
  random_case("cat-3pl", "MFI with MAP", 4, 12, 1, SEED + 2),
  random_case("cat-3pl", "MFI with MAP", 9, 12, 2, SEED + 3),
  random_case("cat-gpcm", "MFI with MAP", 6, 15, 0.5, SEED + 4)
)
names(random_cases) <- sapply(random_cases, `[[`, "name")

# The probability that each of the first 10 items of the 3PL bank is selected, after items 1 and 2 answered 1 and 0,
# at theta 0.3, for a planned length of 5: by integrate over the uniform score of each item for the progressive rule.
pool <- bank3pl[1:10, ]
draw_theta <- 0.3
draw_info <- Ii(draw_theta, pool)$Ii
draw_eligible <- 3:10
draw_largest <- max(draw_info[draw_eligible])
draw_weight <- progressive_weight(2, 5, 1)
lower <- draw_weight * draw_info[draw_eligible]
upper <- lower + (1 - draw_weight) * draw_largest
progressive_probabilities <- sapply(seq_along(draw_eligible), function(j) {
  cdf <- function(s, i) pmin(pmax((s - lower[i]) / (upper[i] - lower[i]), 0), 1)
  density <- function(s) {
    others <- setdiff(seq_along(draw_eligible), j)
    Reduce(`*`, lapply(others, function(i) cdf(s, i)), 1 / (upper[j] - lower[j]))
  }
  breaks <- sort(unique(c(lower[j], upper[j], lower[lower > lower[j] & lower < upper[j]],
                          upper[upper > lower[j] & upper < upper[j]])))
  sum(sapply(seq_len(length(breaks) - 1), function(b) {
    integrate(density, breaks[b], breaks[b + 1], rel.tol = INTEGRATION, abs.tol = 0)$value
  }))
})
stopifnot(abs(sum(progressive_probabilities) - 1) < 1e-10)
draw_weights <- draw_info[draw_eligible]^(5 * draw_weight)

write_fixture("cat-selection", list(
  source = paste("catR 3.17 on the banks and runs of cat-3pl.json and cat-gpcm.json: MWI, KL, GDI, EPV and nextItem;",
                 "MEI and the MAP estimates written out in base R to 1e-14; integrate for the progressive",
                 "probabilities."),
  scores = scores,
  replays = replays,
  random = random_cases,
  draws = list(items = I(draw_eligible - 1), administered = I(c(0, 1)), responses = I(c(1, 0)), theta = draw_theta,
               length = 5, weight = draw_weight, progressive = progressive_probabilities,
               proportional = draw_weights / sum(draw_weights))
))
