# Adaptive testing fixtures from catR: the next item by maximum Fisher information, Kullback-Leibler, thOpt and, within
# aStratified's strata, bOpt (nextItem); the estimate and its standard error after each response (thetaEst, semTheta);
# content balancing (nextItem with cbControl) and stopping decisions (checkStopRule). Banks are catR's generated banks;
# each run's responses are drawn once from a true theta with a fixed seed and are then a fixed sequence.

source("validation/psychometrics/common.R")
suppressPackageStartupMessages(library(catR))

SEED <- 20261029
RANGE <- c(-6, 6) # the package's default theta range for ML, WLE and MAP
# the package's EAP integrates over 61 equally spaced points on [-6, 6] with weights proportional to the prior; catR's
# trapezoid rule on the same points halves the two end weights, where the density is 6e-9.
EAP_GRID <- c(-6, 6, 61)
KL_GRID <- c(-4, 4, 33) # catR's default parInt for KL, which the package uses
LENGTH <- 15

item_id <- function(i) sprintf("item%03d", i)

# the package's parameters of catR's 4PL and GPCM bank rows; catR uses D = 1 by default, as the package does.
logistic_items <- function(bank) {
  lapply(seq_len(nrow(bank)), function(i) {
    list(id = item_id(i), a = bank[i, "a"], b = bank[i, "b"], c = bank[i, "c"], d = bank[i, "d"])
  })
}
gpcm_items <- function(bank) {
  lapply(seq_len(nrow(bank)), function(i) {
    thresholds <- bank[i, -1]
    list(id = item_id(i), a = bank[i, 1], b = I(unname(thresholds[!is.na(thresholds)])))
  })
}

categories <- function(bank, model) {
  if (is.null(model)) rep(2, nrow(bank)) else rowSums(!is.na(bank[, -1, drop = FALSE])) + 1
}

probabilities <- function(theta, bank, model) {
  p <- Pi(theta, bank, model = model)$Pi
  if (is.null(model)) cbind(1 - p, p) else p
}

# The likelihood of responses x to the items `out` at each theta of X, as catR's KL computes it.
likelihood <- function(X, bank, out, x, model) {
  sapply(X, function(theta) {
    if (length(out) == 0) return(1)
    p <- probabilities(theta, bank[out, , drop = FALSE], model)
    prod(p[cbind(seq_along(x), x + 1)])
  })
}

estimate <- function(bank, out, x, method, model) {
  it <- bank[out, , drop = FALSE]
  theta <- if (method == "EAP") {
    thetaEst(it, x, model = model, method = "EAP", parInt = EAP_GRID)
  } else {
    thetaEst(it, x, model = model, method = method, range = RANGE)
  }
  list(method = method, theta = theta,
       standardError = semTheta(theta, it, x, model = model, method = method, parInt = EAP_GRID))
}

# catR's thOpt: the theta at which a 4PL item's information is largest, from nextItem's source.
information_peak <- function(bank) {
  a <- bank[, "a"]
  b <- bank[, "b"]
  c <- bank[, "c"]
  d <- bank[, "d"]
  u <- -3 / 4 + (c + d - 2 * c * d) / 2
  v <- (c + d - 1) / 4
  xstar <- 2 * sqrt(-u / 3) * cos(acos(-v * sqrt(-27 / u^3) / 2) / 3 + 4 * pi / 3) + 1 / 2
  b + log((xstar - c) / (d - xstar)) / a
}

# Scores of the eligible items, NA for the administered ones: MFI's information and KL's integral, the latter divided
# by the largest likelihood on the grid as the package scales it.
selection_scores <- function(bank, model, criterion, theta, out, x) {
  scores <- if (criterion == "MFI") {
    Ii(theta, bank, model = model)$Ii
  } else {
    X <- seq(KL_GRID[1], KL_GRID[2], length = KL_GRID[3])
    lik <- likelihood(X, bank, out, x, model)
    it <- bank[out, , drop = FALSE]
    sapply(seq_len(nrow(bank)), function(i) {
      KL(bank, i, x, it, model = model, theta = theta, nqp = KL_GRID[3], X = X, lik = lik)
    }) / max(lik)
  }
  scores[out] <- NA
  scores
}

# An adaptive test of LENGTH items: before each item, catR's estimate from the responses so far with `methods[1]`
# until they are mixed and `methods[2]` after, and catR's next item at that estimate; then a response drawn from the
# item's probabilities at `true_theta`. The first item is selected at theta = 0, the prior mean, without an estimate:
# its method and standard error are NA. `excluded(k)` lists further items not eligible for item k. nextItem ends with
# set.seed(NULL), so each run seeds its draws and catR's random choices among ties of content areas.
run <- function(name, seed, bank, model, criterion, methods, true_theta, control = NULL, with_scores = TRUE,
                excluded = function(k) integer(0)) {
  count <- categories(bank, model)
  set.seed(seed)
  draws <- runif(LENGTH)
  out <- integer(0)
  x <- numeric(0)
  current <- list(method = NA, theta = 0, standardError = NA)
  steps <- lapply(seq_len(LENGTH), function(k) {
    if (k > 1) {
      method <- if (any(x > 0) && any(x < count[out] - 1)) methods[2] else methods[1]
      current <<- estimate(bank, out, x, method, model)
    }
    # catR takes NULL, not an empty vector, for no items.
    unavailable <- c(out, excluded(k))
    selection <- nextItem(bank, model = model, theta = current$theta, out = if (length(unavailable) > 0) unavailable,
                          x = if (k > 1) x, criterion = criterion, parInt = KL_GRID, random.seed = seed + k,
                          cbControl = control$cbControl, cbGroup = control$groups)
    item <- selection$item
    p <- probabilities(true_theta, bank[item, , drop = FALSE], model)
    response <- sum(draws[k] > cumsum(p[!is.na(p)])[-count[item]])
    step <- c(current, list(nextItem = item_id(item), response = response))
    if (with_scores) step$scores <- selection_scores(bank, model, criterion, current$theta, out, x)
    if (!is.null(control)) step$administeredProportions <- selection$prior.prop
    out <<- c(out, item)
    x <<- c(x, response)
    step
  })
  method <- if (any(x > 0) && any(x < count[out] - 1)) methods[2] else methods[1]
  list(name = name, selection = criterion, estimators = methods, trueTheta = true_theta, steps = steps,
       final = estimate(bank, out, x, method, model), administered = out)
}

bank3pl <- as.matrix(genDichoMatrix(items = 100, model = "3PL", seed = SEED))
runs3pl <- list(
  run("MFI with MAP", SEED + 100, bank3pl, NULL, "MFI", c("BM", "BM"), 0.8),
  run("MFI with EAP until mixed, then ML", SEED + 200, bank3pl, NULL, "MFI", c("EAP", "ML"), -1.2),
  run("KL with EAP until mixed, then WLE", SEED + 300, bank3pl, NULL, "KL", c("EAP", "WL"), 0.3),
  run("thOpt with MAP", SEED + 400, bank3pl, NULL, "thOpt", c("BM", "BM"), 1.5, with_scores = FALSE)
)

# Stopping decisions along the MFI-with-MAP run: catR's checkStopRule with the remaining items, as randomCAT calls it.
stopping_run <- runs3pl[[1]]
STOP_RULES <- list(
  list(rule = "length", threshold = 10),
  list(rule = "precision", threshold = 0.5),
  list(rule = "minInfo", threshold = 0.3)
)
stopping <- lapply(seq_len(LENGTH), function(k) {
  out <- stopping_run$administered[seq_len(k)]
  current <- if (k < LENGTH) stopping_run$steps[[k + 1]] else stopping_run$final
  decide <- function(rules) {
    checkStopRule(th = current$theta, se = current$standardError, N = k, it = bank3pl[-out, , drop = FALSE],
                  stop = list(rule = sapply(rules, `[[`, "rule"), thr = sapply(rules, `[[`, "threshold")))$decision
  }
  list(administered = k, theta = current$theta, standardError = current$standardError,
       decisions = sapply(STOP_RULES, function(rule) decide(list(rule))), any = decide(STOP_RULES))
})

write_fixture("cat-3pl", list(
  source = sprintf("catR's genDichoMatrix(items = 100, model = \"3PL\", seed = %d)", SEED),
  items = logistic_items(bank3pl),
  peaks = information_peak(bank3pl),
  runs = lapply(runs3pl, function(r) r[names(r) != "administered"]),
  stoppingRules = STOP_RULES,
  stopping = stopping
))

# Chang and Ying's a-stratification of a 2PL bank: catR's aStratified strata by a, PER_STRATUM items from each in
# order of increasing a, then the last; within the stratum, catR's bOpt, the difficulty closest to the estimate.
STRATA <- 4
PER_STRATUM <- 3
bank2pl <- as.matrix(genDichoMatrix(items = 100, model = "2PL", seed = SEED))
strata <- aStratified(bank2pl, STRATA)
scheduled <- function(k) paste0("Group", min(ceiling(k / PER_STRATUM), STRATA))
stratified_run <- run("a-stratified with MAP", SEED + 800, bank2pl, NULL, "bOpt", c("BM", "BM"), 0.2,
                      with_scores = FALSE, excluded = function(k) which(strata != scheduled(k)))
write_fixture("cat-2pl", list(
  source = sprintf("catR's genDichoMatrix(items = 100, model = \"2PL\", seed = %d)", SEED),
  items = logistic_items(bank2pl),
  strata = as.integer(sub("Group", "", strata)),
  perStratum = PER_STRATUM,
  run = stratified_run[names(stratified_run) != "administered"]
))

bank_gpcm <- as.matrix(genPolyMatrix(items = 40, nrCat = 4, model = "GPCM", seed = SEED))
runs_gpcm <- list(
  run("MFI with MAP", SEED + 500, bank_gpcm, "GPCM", "MFI", c("BM", "BM"), -0.4),
  run("KL with EAP until mixed, then ML", SEED + 600, bank_gpcm, "GPCM", "KL", c("EAP", "ML"), 1.1)
)
write_fixture("cat-gpcm", list(
  source = sprintf("catR's genPolyMatrix(items = 40, nrCat = 4, model = \"GPCM\", seed = %d)", SEED),
  items = gpcm_items(bank_gpcm),
  runs = lapply(runs_gpcm, function(r) r[names(r) != "administered"])
))

AREAS <- c("algebra", "geometry", "statistics")
TARGETS <- c(0.5, 0.3, 0.2)
content_bank <- genDichoMatrix(items = 60, model = "3PL", seed = SEED,
                               cbControl = list(names = AREAS, props = TARGETS))
bank_content <- as.matrix(content_bank[, c("a", "b", "c", "d")])
content_run <- run("MFI with MAP and content balancing", SEED + 700, bank_content, NULL, "MFI", c("BM", "BM"), 0.5,
                   control = list(cbControl = list(names = AREAS, props = TARGETS), groups = content_bank$Group),
                   with_scores = FALSE)
items_content <- logistic_items(bank_content)
for (i in seq_along(items_content)) items_content[[i]]$contentArea <- as.character(content_bank$Group[i])
write_fixture("cat-content", list(
  source = sprintf("catR's genDichoMatrix(items = 60, model = \"3PL\", seed = %d, cbControl) with three areas", SEED),
  items = items_content,
  targets = as.list(setNames(TARGETS, AREAS)),
  run = content_run[names(content_run) != "administered"]
))
