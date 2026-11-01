# Exposure features of the logistic model: the events without an outcome at a skill, as instructional factors analysis
# counts them. Features are written out in base R from their definitions, and the coefficients are glm's.

source("validation/knowledge-tracing/common.R")

DECAY <- 0.7
EXPONENT <- 0.5
SKILLS <- c("s1", "s2")

set.seed(20261030)
learner_count <- 60
true_beta <- c(s1 = -0.2, s2 = 0.3)
true_exposure <- c(s1 = 0.25, s2 = 0.1)
true_decayed <- c(s1 = 0.3, s2 = 0.2)
true_recency <- c(s1 = 0.5, s2 = -0.4)

events <- do.call(rbind, lapply(seq_len(learner_count), function(l) {
  data.frame(learner = sprintf("learner%02d", l), skill = sample(SKILLS, 36, replace = TRUE),
             lesson = runif(36) < 0.3, time = cumsum(rexp(36, 1 / 60)) + l, stringsAsFactors = FALSE)
}))
events <- events[order(events$time), ]
rownames(events) <- NULL

# The features of each event at its skill, before it, for the learner's earlier lessons at that skill.
state <- new.env()
features <- data.frame(exposures = numeric(nrow(events)), decayedExposures = 0, recencyExposure = 0)
events$outcome <- NA_integer_
for (i in seq_len(nrow(events))) {
  key <- paste(events$learner[i], events$skill[i], sep = "\r")
  h <- state[[key]] %||% list(count = 0, decayed = 0, last = NA)
  elapsed <- if (is.na(h$last)) 0 else events$time[i] - h$last
  features[i, ] <- c(h$count, h$decayed, if (elapsed == 0) 0 else elapsed^-EXPONENT)
  if (events$lesson[i]) {
    state[[key]] <- list(count = h$count + 1, decayed = DECAY * h$decayed + 1, last = events$time[i])
  } else {
    k <- events$skill[i]
    theta <- true_beta[[k]] + true_exposure[[k]] * features$exposures[i] + true_decayed[[k]] * features$decayedExposures[i] +
      true_recency[[k]] * features$recencyExposure[i]
    events$outcome[i] <- rbinom(1, 1, plogis(theta))
  }
}

# One column of each feature for each skill, and glm on the events with an outcome.
observed <- which(!events$lesson)
columns <- do.call(cbind, lapply(SKILLS, function(k) {
  at <- as.numeric(events$skill[observed] == k)
  m <- cbind(at, at * features$exposures[observed], at * features$decayedExposures[observed],
             at * features$recencyExposure[observed])
  colnames(m) <- paste0(c("intercept", "exposures", "decayedExposures", "recencyExposure"), ":", k)
  m
}))
fit <- glm(events$outcome[observed] ~ 0 + columns, family = binomial, control = glm.control(epsilon = 1e-14, maxit = 100))
score <- crossprod(columns, events$outcome[observed] - fitted(fit))
stopifnot(fit$converged, max(abs(score)) < 1e-9, !anyNA(coef(fit)))
b <- setNames(coef(fit), colnames(columns))

write_fixture("logistic-exposures", list(
  source = "set.seed(20261030): 60 learners, 36 events each at one of 2 skills, 30% lessons; glm(binomial).",
  decay = DECAY,
  exponent = EXPONENT,
  events = lapply(seq_len(nrow(events)), function(i) list(
    learner = events$learner[i], item = paste0("i-", events$skill[i]), skills = I(events$skill[i]),
    time = events$time[i], outcome = if (events$lesson[i]) NULL else events$outcome[i])),
  features = list(exposures = features$exposures, decayedExposures = features$decayedExposures,
                  recencyExposure = features$recencyExposure),
  skills = setNames(lapply(SKILLS, function(k) list(
    intercept = unname(b[[paste0("intercept:", k)]]),
    coefficients = list(exposures = unname(b[[paste0("exposures:", k)]]),
                        "decayedExposures(0.7)" = unname(b[[paste0("decayedExposures:", k)]]),
                        "recencyExposure(0.5)" = unname(b[[paste0("recencyExposure:", k)]])))), SKILLS),
  logLikelihood = as.numeric(logLik(fit))
))
