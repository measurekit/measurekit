# First attempts, by DataShop's convention: the first try of a learner at an item, with a first try that took hints
# counted as incorrect. Base R on a seeded log, with and without the attempt number the host records.

source("validation/knowledge-tracing/common.R")

set.seed(20261031)
learner_count <- 40
events <- do.call(rbind, lapply(seq_len(learner_count), function(l) {
  data.frame(learner = sprintf("learner%02d", l), item = sprintf("item%d", sample(8, 24, replace = TRUE)),
             time = cumsum(rexp(24, 1 / 60)) + l, stringsAsFactors = FALSE)
}))
events <- events[order(events$time), ]
rownames(events) <- NULL
events$outcome <- ifelse(runif(nrow(events)) < 0.15, NA, rbinom(nrow(events), 1, 0.6))
events$hints <- ifelse(is.na(events$outcome), sample(0:2, nrow(events), replace = TRUE, prob = c(0.2, 0.6, 0.2)),
                       sample(0:2, nrow(events), replace = TRUE, prob = c(0.7, 0.2, 0.1)))
# The host's attempt number counts every earlier event of the learner at the item, and starts at 1.
events$attemptNumber <- ave(seq_len(nrow(events)), events$learner, events$item, FUN = seq_along)

# The first attempts, as the positions of the events kept, with their outcomes after recoding hinted ones.
first_attempts <- function(numbered, hinted) {
  observed <- !is.na(events$outcome)
  first <- if (numbered) events$attemptNumber == 1 else !duplicated(ifelse(observed, paste(events$learner, events$item), NA)) & observed
  kept <- which(!observed | first)
  outcome <- events$outcome[kept]
  hinted_first <- !is.na(outcome) & events$hints[kept] > 0
  if (hinted == "incorrect") outcome[hinted_first] <- 0
  if (hinted == "null") outcome[hinted_first] <- NA
  list(positions = kept - 1, outcomes = outcome)
}

event_list <- function(numbered) lapply(seq_len(nrow(events)), function(i) {
  event <- list(learner = events$learner[i], item = events$item[i], skills = I(character(0)), time = events$time[i],
                outcome = if (is.na(events$outcome[i])) NULL else events$outcome[i], hints = events$hints[i])
  if (numbered) event$attemptNumber <- events$attemptNumber[i]
  event
})

write_fixture("data-first-attempts", list(
  source = "set.seed(20261031): 40 learners, 24 events each at 8 items, 15% without an outcome; base R.",
  numbered = event_list(TRUE),
  unnumbered = event_list(FALSE),
  expected = lapply(setNames(c(TRUE, FALSE), c("numbered", "unnumbered")), function(numbered) {
    lapply(setNames(c("incorrect", "null", "keep"), c("incorrect", "null", "keep")),
           function(hinted) first_attempts(numbered, hinted))
  })
))
