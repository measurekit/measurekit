# The LKT features that logistic.R does not cover, as LKT 1.7.0 computes them on its public sample, and the time-window
# counts of DAS3H, which no package computes, written out in base R. The sample has no clock, so each learner's events get
# a seeded time: the practice time before the event, as LKT's own, plus gaps of nothing, a minute, an hour, a day or three
# days that the learner spends away. Every feature is a function of one learner's events at one skill, so LKT's columns
# for the skill of each row are the expected values.

source("validation/knowledge-tracing/common.R")
suppressPackageStartupMessages({
  library(LKT)
  library(data.table)
})

set.seed(20261029)
data(samplelkt)
lkt <- as.data.frame(samplelkt)
lkt <- lkt[lkt$Outcome %in% c("CORRECT", "INCORRECT"), ]
lkt$Outcome <- as.character(lkt$Outcome)
lkt$Anon.Student.Id <- as.character(lkt$Anon.Student.Id)
lkt$CF..ansbin. <- as.integer(lkt$Outcome == "CORRECT")
# A duration is above 0 in measurekit; the sample has events of 0 seconds.
lkt$Duration..sec. <- pmax(lkt$Duration..sec., 1)
lkt$CF..reltime. <- LKT:::practiceTime(lkt)
rows <- nrow(lkt)
away <- ifelse(runif(rows) < 1 / 3, 0, sample(c(60, 3600, 86400, 3 * 86400), rows, replace = TRUE))
first_of_learner <- !duplicated(lkt$Anon.Student.Id)
away[first_of_learner] <- 0
lkt$CF..Time. <- lkt$CF..reltime. + ave(away, lkt$Anon.Student.Id, FUN = cumsum)
lkt <- lkt[order(lkt$Anon.Student.Id, lkt$CF..Time.), ]
lkt <- data.table::setDT(computeSpacingPredictors(lkt, "KC..Default."))
lkt$cor <- LKT:::countOutcome(lkt, lkt$index, "CORRECT")
lkt$icor <- LKT:::countOutcome(lkt, lkt$index, "INCORRECT")

DAY <- 86400
# measurekit's feature, LKT's name and its parameters par1 to par4. LKT's `logit` has g = 0.1 + 30 par1, and the dash
# features a scale of par1 days.
spec <- function(name, feature, lkt_feature, par1 = 0, par2 = 0, par3 = 0, par4 = 0) {
  list(name = name, feature = feature, lkt = lkt_feature, par = c(par1, par2, par3, par4))
}
specs <- list(
  spec("proportion", "proportion", "prop"),
  spec("netSuccesses", "netSuccesses", "linecomp"),
  spec("powerOpportunities", list(powerOpportunities = 0.6), "powafm", 0.6),
  spec("logit", list(logit = 2.5), "logit", (2.5 - 0.1) / 30),
  spec("decayedProportionFailed", list(decayedProportionFailed = 0.9), "propdec2", 0.9),
  spec("recencySuccess", list(recencySuccess = 0.5), "recencysuc", 0.5),
  spec("recencyFailure", list(recencyFailure = 0.5), "recencyfail", 0.5),
  spec("base", list(base = 0.5), "base", 0.5),
  spec("baseSuccess", list(baseSuccess = 0.4), "basesuc", 0.4),
  spec("baseFailure", list(baseFailure = 0.3), "basefail", 0.3),
  spec("basePractice", list(basePractice = list(decay = 0.5, scale = 0.1)), "base2", 0.5, 0.1),
  spec("basePracticeSuccess", list(basePracticeSuccess = list(decay = 0.4, scale = 0.2)), "base2suc", 0.4, 0.2),
  spec("basePracticeFailure", list(basePracticeFailure = list(decay = 0.3, scale = 0.05)), "base2fail", 0.3, 0.05),
  spec("dash", list(dash = 2 * DAY), "dashafm", 2),
  spec("dashSuccess", list(dashSuccess = 0.5 * DAY), "dashsuc", 0.5),
  spec("ppe", list(ppe = list(power = 0.2, decay = 0.3, spacing = 0.4, weight = 0.6)), "ppe", 0.2, 0.3, 0.4, 0.6)
)

compute <- function(s) {
  column <- LKT:::computefeatures(lkt, s$lkt, s$par[1], s$par[2], lkt$index, lkt$index, s$par[3], s$par[4], 0,
                                  "KC..Default.")
  as.numeric(column)
}
lkt_columns <- lapply(specs, compute)
# The seeded times give every event an age above 0 and no ties, where LKT returns infinity and NaN, so that measurekit's
# values at those edges are checked by the tests alone.
stopifnot(all(vapply(lkt_columns, function(column) all(is.finite(column)), TRUE)))

# --- The windows of DAS3H, in base R -------------------------------------------------------------------------------------
# ln(1 + the attempts, successes or failures of the learner at the skill whose time is within w of the event's, before it).
window_count <- function(w, pick) {
  vapply(seq_len(rows), function(i) {
    earlier <- which(lkt$Anon.Student.Id[seq_len(i - 1)] == lkt$Anon.Student.Id[i] &
                       lkt$KC..Default.[seq_len(i - 1)] == lkt$KC..Default.[i] & lkt$CF..Time.[i] - lkt$CF..Time.[seq_len(i - 1)] <= w)
    log(1 + sum(pick(lkt$CF..ansbin.[earlier])))
  }, 0)
}
windows <- list(
  list(name = "windowOpportunities", feature = list(windowOpportunities = DAY), values = window_count(DAY, function(y) rep(1, length(y)))),
  list(name = "windowSuccesses", feature = list(windowSuccesses = 3600), values = window_count(3600, function(y) y)),
  list(name = "windowFailures", feature = list(windowFailures = 3 * DAY), values = window_count(3 * DAY, function(y) 1 - y))
)

events <- events_of(lkt$Anon.Student.Id, lkt$KC..Default., as.list(lkt$KC..Default.), lkt$CF..Time., lkt$CF..ansbin.)
events <- Map(function(event, duration) c(event, list(duration = duration)), events, lkt$Duration..sec.)

write_fixture("logistic-lkt-features", list(
  source = paste("LKT 1.7.0 computefeatures on samplelkt (1,941 events of 17 learners) with seeded times: practice time",
                 "plus gaps away (set.seed(20261029)); the windows of DAS3H in base R."),
  events = events,
  features = c(
    lapply(specs, function(s) list(name = s$name, feature = s$feature, lkt = s$lkt, values = compute(s))),
    lapply(windows, function(w) list(name = w$name, feature = w$feature, lkt = NULL, values = w$values))
  )
))
