# Writes rater-scoring.json: the routing curve, the windows of drift, the mappings to the human scale and the replay
# shifts of the evaluation of automated scores, on the essays of rater-automated-scoring.json.
#
# References, base R written out: each statistic of a routing point and of a window on its subset of essays (Cohen's
# kappa, unweighted and with quadratic weights on the categories' values, from the table of the pair; cor; the SMD on the
# pooled SD; RMSE); the isotonic regression by R's isoreg of the human scores on the machine scores, the essays with one
# machine score sorted by their human score from the highest so that isoreg pools each tie, checked against
# Iso::pava of the mean human score of each machine score weighted by the essays; the mapped scores by approx with
# rule = 2; the linear mapping by mean and sd; the replay's shift by mean and sd of the differences.
#
# Data: ELLIPSE's 8,859 essays, the machine scores rounded and trimmed to 1 to 5 against the first human score, with a
# confidence of the scorer drawn from how near its unrounded score lies to a point (set.seed(20261104), two decimals);
# PERSUADE 2.0's 1,818 essays scored over ten weeks at seeded times, in windows of a week; the first 200 of ELLIPSE's
# unrounded machine scores locked, and replayed with planted shifts of 0, 0.04 and 0.06 SD and centered noise. The
# linear mapping is of the first 500 essays, from exp(machine / 2), as the unrounded machine scores already have the
# human scores' mean and SD.

source("validation/psychometrics/common.R")
suppressPackageStartupMessages(library(Iso))

SEED <- 20261104
WEEK_MS <- 7 * 24 * 3600 * 1000
REPLAY_SUBJECTS <- 200
REPLAY_SHIFTS <- c(0, 0.04, 0.06)
MAX_SHIFT_SD <- 0.05
LINEAR_SUBJECTS <- 500

scoring <- fromJSON("packages/psychometrics/test/fixtures/rater-automated-scoring.json")$cases
to_scale <- function(scores, scale) pmin(scale[2], pmax(scale[1], floor(scores + 0.5)))

# Cohen's kappa of two lists of categories, with agreement weights on the categories' values.
cohen_kappa <- function(x, y, quadratic) {
  levels <- min(x, y):max(x, y)
  if (length(levels) == 1) return(NA)
  observed <- table(factor(x, levels), factor(y, levels)) / length(x)
  expected <- outer(rowSums(observed), colSums(observed))
  weights <- if (quadratic) 1 - outer(levels, levels, "-")^2 / (length(levels) - 1)^2 else diag(length(levels))
  p_o <- sum(weights * observed)
  p_e <- sum(weights * expected)
  if (p_e == 1) NA else (p_o - p_e) / (1 - p_e)
}

agreement <- function(machine, human) {
  n <- length(machine)
  pooled <- if (n > 1) sqrt((var(machine) + var(human)) / 2) else NA
  list(
    subjects = n,
    exactAgreement = mean(machine == human),
    adjacentAgreement = mean(abs(machine - human) <= 1),
    kappa = cohen_kappa(machine, human, FALSE),
    quadraticKappa = cohen_kappa(machine, human, TRUE),
    correlation = if (n > 1 && sd(machine) > 0 && sd(human) > 0) cor(machine, human) else NA,
    smd = if (!is.na(pooled) && pooled > 0) (mean(machine) - mean(human)) / pooled else NA,
    rmse = sqrt(mean((machine - human)^2))
  )
}

# --- Routing curve -------------------------------------------------------------------------------------------------------
set.seed(SEED)
ellipse <- scoring$ELLIPSE
machine <- to_scale(ellipse$machine, ellipse$scale)
human <- ellipse$first
nearness <- 0.5 - abs(ellipse$machine - round(ellipse$machine))
confidence <- round(pmin(1, pmax(0, nearness + rnorm(length(machine), 0, 0.15))), 2)
thresholds <- sort(unique(confidence), decreasing = TRUE)
routing <- lapply(thresholds, function(threshold) {
  covered <- confidence >= threshold
  c(list(minConfidence = threshold, coverage = mean(covered)), agreement(machine[covered], human[covered]))
})

# --- Windows of drift ----------------------------------------------------------------------------------------------------
persuade <- scoring$`PERSUADE 2.0`
persuade_machine <- to_scale(persuade$machine, persuade$scale)
start <- 1.7e12
time <- start + round(runif(length(persuade_machine), 0, 10 * WEEK_MS))
index <- floor((time - min(time)) / WEEK_MS)
windows <- lapply(sort(unique(index)), function(k) {
  members <- index == k
  list(start = min(time) + k * WEEK_MS, end = min(time) + (k + 1) * WEEK_MS,
       agreement = agreement(persuade_machine[members], persuade$human[members]))
})

# --- Mappings ------------------------------------------------------------------------------------------------------------
x <- ellipse$machine
y <- human
order_ <- order(x, -y)
fit <- isoreg(x[order_], y[order_])
knot_x <- sort(unique(x))
knot_y <- sapply(knot_x, function(value) unique(fit$yf[x[order_] == value]))
stopifnot(is.numeric(knot_y), length(knot_y) == length(knot_x))
means <- tapply(y, x, mean)
counts <- tapply(y, x, length)
stopifnot(max(abs(pava(as.numeric(means), as.numeric(counts)) - knot_y)) < 1e-12)
# The unrounded machine scores already have the human mean and SD, so the linear mapping takes a curved function of them.
linear_x <- exp(x[seq_len(LINEAR_SUBJECTS)] / 2)
linear_y <- y[seq_len(LINEAR_SUBJECTS)]
at <- c(min(x) - 1, quantile(x, c(0.1, 0.33, 0.5, 0.77, 0.9), names = FALSE), max(x) + 1, knot_x[c(2, 10)])
mapping <- list(
  knots = list(machine = knot_x, human = knot_y),
  at = at,
  mapped = approx(knot_x, knot_y, xout = at, rule = 2, ties = "ordered")$y,
  linear = list(machine = linear_x, human = linear_y, slope = sd(linear_y) / sd(linear_x),
                intercept = mean(linear_y) - sd(linear_y) / sd(linear_x) * mean(linear_x))
)

# --- Replays -------------------------------------------------------------------------------------------------------------
locked <- ellipse$machine[seq_len(REPLAY_SUBJECTS)]
noise <- rnorm(REPLAY_SUBJECTS, 0, 0.3)
noise <- noise - mean(noise)
replays <- lapply(REPLAY_SHIFTS, function(shift) {
  replayed <- locked + shift * sd(locked) + noise
  differences <- replayed - locked
  standardized <- mean(differences) / sd(locked)
  list(shift = shift, locked = locked, replayed = replayed, meanShift = mean(differences), lockedSd = sd(locked),
       standardizedShift = standardized, standardError = sd(differences) / sqrt(REPLAY_SUBJECTS) / sd(locked),
       correlation = cor(locked, replayed), passed = abs(standardized) <= MAX_SHIFT_SD)
})

write_fixture("rater-scoring", list(
  source = paste(
    "Base R 4.6.1 on the essays of rater-automated-scoring.json: kappa from the table of each pair with weights on the",
    "categories' values, cor, the pooled SMD and RMSE; isoreg, checked against Iso 0.0-21's pava, and approx(rule = 2);",
    "set.seed(20261104) for the confidences, the times and the replays' noise."
  ),
  routing = list(confidence = confidence, points = routing),
  drift = list(time = time, width = WEEK_MS, windows = windows),
  mapping = mapping,
  replay = list(maxShiftSd = MAX_SHIFT_SD, cases = replays)
))
