# Classification stopping rules (cat-stopping.json), along the four catR runs of cat-3pl.json, which the test reads:
# after each of the 15 responses, from catR's estimate and its standard error at that point. Expected values only, by
# case name.
#
# The confidence-interval rule: catR's checkStopRule with rule "classification" for each cut, at alpha = 1 - level; a
# test of several cuts stops when every cut's rule does, and its category is the number of cuts at or below the lower
# end of the interval. catR stops when any one of several rules does, so each cut is its own call.
#
# The sequential probability ratio test: mirtCAT's SPRT written out in base R. At each cut, the log-likelihood of the
# responses at its upper bound less that at its lower bound, from catR's Pi, against ln(β / (1 - α)) and
# ln((1 - β) / α): below the first decides below the cut, above the second above it. mirtCAT 1.14 compares the vector of
# several cuts with `||`, which R 4.3 and later refuse ("'length = 2' in coercion to 'logical(1)'"), so its SPRT runs one
# cut alone. The script checks the write-up against mirtCAT on one cut: on three simulated persons the write-up decides
# at the item where mirtCAT stops, and as mirtCAT does.

source("validation/psychometrics/common.R")
suppressPackageStartupMessages({
  library(catR)
  library(mirtCAT)
})

fixture <- fromJSON("packages/psychometrics/test/fixtures/cat-3pl.json", simplifyVector = FALSE)
bank <- do.call(rbind, lapply(fixture$items, function(item) c(a = item$a, b = item$b, c = item$c, d = item$d)))
item_index <- setNames(seq_along(fixture$items), vapply(fixture$items, `[[`, "", "id"))

# The estimate after k responses of a run, and the items and responses so far.
after <- function(run, k) {
  current <- if (k < length(run$steps)) run$steps[[k + 1]] else run$final
  steps <- run$steps[seq_len(k)]
  list(
    theta = current$theta, standardError = current$standardError,
    items = unname(item_index[vapply(steps, `[[`, "", "nextItem")]),
    responses = vapply(steps, function(step) as.numeric(step$response), 0)
  )
}

interval_decisions <- function(cuts, level) {
  lapply(fixture$runs, function(run) {
    list(run = run$name, decisions = lapply(seq_along(run$steps), function(k) {
      point <- after(run, k)
      cleared <- vapply(cuts, function(cut) {
        checkStopRule(th = point$theta, se = point$standardError, N = k,
                      stop = list(rule = "classification", thr = cut, alpha = 1 - level))$decision
      }, TRUE)
      lower <- point$theta - qnorm(1 - (1 - level) / 2) * point$standardError
      list(administered = k, stops = all(cleared), category = if (all(cleared)) sum(cuts <= lower) else NULL)
    }))
  })
}

log_likelihood <- function(theta, items, responses) {
  p <- Pi(theta, bank[items, , drop = FALSE])$Pi
  sum(log(ifelse(responses == 1, p, 1 - p)))
}

# Each cut's decision: 1 above, -1 below, 0 undecided.
sprt_decisions_at <- function(cuts, alpha, beta, items, responses) {
  bounds <- c(log(beta / (1 - alpha)), log((1 - beta) / alpha))
  vapply(cuts, function(cut) {
    ratio <- log_likelihood(cut$upper, items, responses) - log_likelihood(cut$lower, items, responses)
    if (ratio > bounds[2]) 1 else if (ratio < bounds[1]) -1 else 0
  }, 0)
}

sprt_decisions <- function(cuts, alpha, beta) {
  lapply(fixture$runs, function(run) {
    list(run = run$name, decisions = lapply(seq_along(run$steps), function(k) {
      point <- after(run, k)
      decided <- sprt_decisions_at(cuts, alpha, beta, point$items, point$responses)
      list(administered = k, stops = all(decided != 0), category = if (all(decided != 0)) sum(decided == 1) else NULL)
    }))
  })
}

# The write-up against mirtCAT's own SPRT on one cut.
model <- generate.mirt_object(data.frame(a1 = bank[, "a"], d = -bank[, "a"] * bank[, "b"], g = bank[, "c"]), "3PL")
check_cut <- list(list(lower = -0.1, upper = 0.4))
for (truth in c(1.2, -0.9, 0.6)) {
  set.seed(20261102 + round(100 * truth))
  pattern <- generate_pattern(model, Theta = matrix(truth))
  result <- mirtCAT(mo = model, local_pattern = pattern, criteria = "MI", method = "MAP",
                    design = list(sprt_lower = -0.1, sprt_upper = 0.4, max_items = 60))
  items <- result$items_answered
  decided <- vapply(seq_along(items), function(k) {
    sprt_decisions_at(check_cut, 0.05, 0.05, items[1:k], result$scored_responses[items[1:k]])
  }, 0)
  first <- which(decided != 0)[1]
  if (result$classification == "no decision") {
    stopifnot(is.na(first), length(items) == 60)
  } else {
    stopifnot(first == length(items), (decided[first] == 1) == (result$classification == "above cutoff"))
  }
}

interval_case <- function(name, cuts, level) {
  list(name = name, cuts = as.list(cuts), level = level, runs = interval_decisions(cuts, level))
}

sprt_case <- function(name, bounds, alpha, beta) {
  cuts <- lapply(bounds, function(b) list(lower = b[1], upper = b[2]))
  list(name = name, cuts = cuts, alpha = alpha, beta = beta, runs = sprt_decisions(cuts, alpha, beta))
}

write_fixture("cat-stopping", list(
  source = paste(
    "The runs of cat-3pl.json: catR 3.17's checkStopRule(rule = \"classification\") for each cut; mirtCAT 1.14's SPRT",
    "written out in base R with catR's Pi, checked against mirtCAT on one cut."
  ),
  interval = list(
    interval_case("one cut at 0", 0, 0.95),
    interval_case("one cut at 0.5, level 0.9", 0.5, 0.9),
    interval_case("cuts at -0.5 and 2.5, level 0.9", c(-0.5, 2.5), 0.9),
    interval_case("three cuts, level 0.8", c(-2, 0.3, 3), 0.8)
  ),
  sprt = list(
    sprt_case("one cut", list(c(-0.6, 0.2)), 0.05, 0.05),
    sprt_case("one cut, alpha 0.1 and beta 0.2", list(c(0.3, 0.9)), 0.1, 0.2),
    sprt_case("two cuts", list(c(-2, -1), c(0, 1)), 0.05, 0.05),
    sprt_case("two cuts, alpha and beta 0.1", list(c(-1.8, -0.6), c(1, 2.2)), 0.1, 0.1)
  )
))
