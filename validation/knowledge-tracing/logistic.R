# Logistic model fixtures: LKT's features on its public sample, and fits of the additive factors model, performance
# factors analysis and LKT features by glm, by lme4's Laplace approximation (glmer, nAGQ = 1) and with a ridge by
# LiblineaR, which LKT fits with.

source("validation/knowledge-tracing/common.R")
suppressPackageStartupMessages({
  library(LKT)
  library(lme4)
  library(LiblineaR)
})

DECAYS <- list(proportion = 0.9, logit = 0.8, recency = 0.5, counts = 0.7)
# The equivalent of LiblineaR's cost C: minimizing 1/2 |w|² + C × negative log-likelihood is the ridge λ = 1/C.
COST <- 512

# Features of each learner's history with each skill, before each event, written out from LKT's definitions. `skills`
# is a list of skill vectors; the events are in time order for each learner.
history_features <- function(learner, skills, time, outcome) {
  n <- length(learner)
  rows <- list()
  state <- new.env()
  for (i in seq_len(n)) {
    for (skill in skills[[i]]) {
      key <- paste(learner[i], skill, sep = "\r")
      h <- state[[key]] %||% list(s = 0, f = 0, last = NA, ds = c(0, 0, 0), df = c(0, 0, 0))
      attempts <- h$s + h$f
      d <- c(DECAYS$proportion, DECAYS$logit, DECAYS$counts)
      prior <- d^attempts
      elapsed <- if (is.na(h$last)) 0 else time[i] - h$last
      rows[[length(rows) + 1]] <- data.frame(
        event = i, skill = skill,
        opportunities = attempts, successes = h$s, failures = h$f,
        logOpportunities = log(1 + attempts), logSuccesses = log(1 + h$s), logFailures = log(1 + h$f),
        decayedProportion = (prior[1] + h$ds[1]) / (prior[1] + h$ds[1] + prior[1] + h$df[1]),
        decayedLogit = log((prior[2] + h$ds[2]) / (prior[2] + h$df[2])),
        recency = if (elapsed == 0) 0 else elapsed^-DECAYS$recency,
        decayedOpportunities = h$ds[3] + h$df[3], decayedSuccesses = h$ds[3], decayedFailures = h$df[3]
      )
      y <- outcome[i]
      state[[key]] <- list(s = h$s + y, f = h$f + 1 - y, last = time[i], ds = d * h$ds + y, df = d * h$df + 1 - y)
    }
  }
  do.call(rbind, rows)
}

# --- LKT's sample: features as LKT computes them, and ridge fits as LiblineaR makes them ---------------------------

data(samplelkt)
lkt <- as.data.frame(samplelkt)
lkt <- lkt[lkt$Outcome %in% c("CORRECT", "INCORRECT"), ]
lkt$Outcome <- as.character(lkt$Outcome)
lkt$Anon.Student.Id <- as.character(lkt$Anon.Student.Id)
lkt$CF..ansbin. <- as.integer(lkt$Outcome == "CORRECT")
lkt$CF..reltime. <- LKT:::practiceTime(lkt)
lkt$CF..Time. <- lkt$CF..reltime.
lkt <- lkt[order(lkt$Anon.Student.Id, lkt$CF..Time.), ]
lkt <- data.table::setDT(computeSpacingPredictors(lkt, "KC..Default."))
features <- c("lineafm", "linesuc", "linefail", "logafm", "logsuc", "logfail", "propdec", "logitdec", "recency",
              "expdecafm", "expdecsuc", "expdecfail")
model <- LKT(data = lkt, interc = FALSE, components = rep("KC..Default.", length(features)), features = features,
             fixedpars = c(DECAYS$proportion, DECAYS$logit, DECAYS$recency, DECAYS$counts, DECAYS$counts, DECAYS$counts),
             verbose = FALSE)
computed <- model$newdata
column <- function(feature) computed[[paste0(feature, "KC..Default.")]]
written <- history_features(lkt$Anon.Student.Id, as.list(lkt$KC..Default.), lkt$CF..Time., lkt$CF..ansbin.)
names <- c(lineafm = "opportunities", linesuc = "successes", linefail = "failures", logafm = "logOpportunities",
           logsuc = "logSuccesses", logfail = "logFailures", propdec = "decayedProportion", logitdec = "decayedLogit",
           recency = "recency", expdecafm = "decayedOpportunities", expdecsuc = "decayedSuccesses",
           expdecfail = "decayedFailures")
for (feature in features) {
  stopifnot(isTRUE(all.equal(column(feature), written[[names[[feature]]]], tolerance = 1e-12)))
}

# The ridge fits in the package's parametrization, every learner and skill with its own column: LiblineaR's L2-regularized
# logistic regression (type 0) without a bias term, to a tight tolerance.
# LiblineaR stops on a gradient norm relative to the start's, which leaves weakly penalized directions about 1e-5 short
# of the minimum; Newton steps on the same objective, -log-likelihood + |w|² / (2C), finish it.
liblinear_gaps <- c()
liblinear <- function(design, y) {
  fit <- LiblineaR(design, y, type = 0, cost = COST, epsilon = 1e-12, bias = -1)
  weights <- fit$W[1, seq_len(ncol(design))]
  if (fit$ClassNames[1] == 0) weights <- -weights
  w <- weights
  for (iteration in 1:20) {
    p <- plogis(drop(design %*% w))
    gradient <- drop(crossprod(design, p - y)) + w / COST
    if (max(abs(gradient)) < 1e-12) break
    w <- w - solve(crossprod(design, design * (p * (1 - p))) + diag(ncol(design)) / COST, gradient)
  }
  stopifnot(max(abs(gradient)) < 1e-10)
  liblinear_gaps <<- c(liblinear_gaps, max(abs(w - weights)))
  setNames(w, colnames(design))
}
skills_lkt <- factor(lkt$KC..Default.)
learners_lkt <- factor(lkt$Anon.Student.Id)
ridge_design <- function(columns, learners = TRUE) {
  cbind(if (learners) model.matrix(~ 0 + learners_lkt) else NULL, model.matrix(~ 0 + skills_lkt), columns)
}
skill_columns <- function(values, name) {
  m <- model.matrix(~ 0 + skills_lkt) * values
  colnames(m) <- paste0(name, ":", levels(skills_lkt))
  m
}
by_skill <- function(weights, names) {
  setNames(lapply(levels(skills_lkt), function(k) {
    list(intercept = unname(weights[[paste0("skills_lkt", k)]]),
         coefficients = setNames(lapply(names, function(f) unname(weights[[paste0(f, ":", k)]])), names))
  }), levels(skills_lkt))
}
by_learner <- function(weights) {
  setNames(lapply(levels(learners_lkt), function(l) unname(weights[[paste0("learners_lkt", l)]])), levels(learners_lkt))
}
afm_design <- ridge_design(skill_columns(written$opportunities, "opportunities"))
afm_ridge <- liblinear(afm_design, lkt$CF..ansbin.)
pfa_ridge <- liblinear(ridge_design(cbind(skill_columns(written$successes, "successes"),
                                          skill_columns(written$failures, "failures")), learners = FALSE),
                       lkt$CF..ansbin.)

# LKT's own fit of AFM: its formula drops the first learner's column and LiblineaR stops at epsilon = 1e-4.
lkt_afm <- LKT(data = lkt, interc = FALSE, components = c("Anon.Student.Id", "KC..Default.", "KC..Default."),
               features = c("intercept", "intercept", "lineafm$"), verbose = FALSE)
equivalent <- plogis(drop(afm_design %*% afm_ridge))
# LKT's own design, every column but the first learner's, fitted to a tight tolerance: the parametrization's share.
lkt_design <- as.matrix(lkt_afm$predictors)
tight <- LiblineaR(lkt_design, lkt$CF..ansbin., type = 0, cost = COST, epsilon = 1e-12, bias = 0)
tight_prediction <- predict(tight, lkt_design, proba = TRUE)$probabilities[, "1"]
log_likelihood <- function(p) sum(log(ifelse(lkt$CF..ansbin. == 1, p, 1 - p)))
lkt_difference <- list(
  columns = c(lkt = ncol(lkt_design), ours = ncol(afm_design)),
  liblinearGap = max(liblinear_gaps),
  maxPrediction = max(abs(lkt_afm$prediction - equivalent)),
  maxPredictionTight = max(abs(tight_prediction - equivalent)),
  logLikelihood = lkt_afm$loglike,
  tightLogLikelihood = log_likelihood(tight_prediction),
  equivalentLogLikelihood = log_likelihood(equivalent)
)

write_fixture("logistic-lkt", list(
  source = "LKT's samplelkt: features by LKT 1.7.0; ridge fits by LiblineaR(type = 0, cost = 512), finished by Newton.",
  decays = DECAYS,
  ridge = 1 / COST,
  events = events_of(lkt$Anon.Student.Id, lkt$KC..Default., as.list(lkt$KC..Default.), lkt$CF..Time., lkt$CF..ansbin.),
  features = setNames(lapply(features, function(feature) column(feature)), names[features]),
  afm = list(skills = by_skill(afm_ridge, "opportunities"), learners = by_learner(afm_ridge)),
  pfa = list(skills = by_skill(pfa_ridge, c("successes", "failures"))),
  lkt = lkt_difference
))

# --- A simulated log with items of one and two skills: glm and glmer -------------------------------------------------

set.seed(20261029)
SKILLS <- paste0("s", 1:5)
ITEMS <- c(setNames(as.list(rep(SKILLS, each = 2)), paste0("i", 1:10)),
           list(i11 = c("s1", "s2"), i12 = c("s2", "s3"), i13 = c("s3", "s4"), i14 = c("s4", "s5")))
learner_count <- 80
true_theta <- rnorm(learner_count, 0, 0.7)
true_beta <- setNames(rnorm(5, -0.3, 0.4), SKILLS)
true_gamma <- setNames(runif(5, 0.03, 0.12), SKILLS)
simulated <- do.call(rbind, lapply(seq_len(learner_count), function(l) {
  data.frame(learner = sprintf("learner%02d", l), item = sample(names(ITEMS), 40, replace = TRUE),
             time = cumsum(rexp(40, 1 / 60)) + l)
}))
simulated <- simulated[order(simulated$time), ]
simulated_skills <- unname(ITEMS[simulated$item])
practiced <- new.env()
simulated$outcome <- sapply(seq_len(nrow(simulated)), function(i) {
  skills <- simulated_skills[[i]]
  key <- paste(simulated$learner[i], skills)
  counts <- sapply(key, function(k) practiced[[k]] %||% 0)
  for (k in key) practiced[[k]] <- (practiced[[k]] %||% 0) + 1
  theta <- true_theta[as.integer(sub("learner", "", simulated$learner[i]))]
  rbinom(1, 1, plogis(theta + sum(true_beta[skills] + true_gamma[skills] * counts)))
})

# One row per event: the sum over its skills of a skill's indicator, or of its indicator times a feature.
design_of <- function(events, skills, values) {
  rows <- history_features(events$learner, skills, events$time, events$outcome)
  indicator <- function(column) {
    m <- matrix(0, nrow(events), length(SKILLS), dimnames = list(NULL, SKILLS))
    for (r in seq_len(nrow(rows))) m[rows$event[r], rows$skill[r]] <- m[rows$event[r], rows$skill[r]] + column[r]
    m
  }
  c(list(intercept = indicator(rep(1, nrow(rows)))), lapply(setNames(values, values), function(v) indicator(rows[[v]])))
}
glm_fit <- function(events, skills, values, learners) {
  x <- design_of(events, skills, values)
  learner <- factor(events$learner)
  columns <- do.call(cbind, lapply(names(x), function(n) { m <- x[[n]]; colnames(m) <- paste0(n, ":", SKILLS); m }))
  design <- if (learners) cbind(model.matrix(~ 0 + learner), columns) else columns
  # With every event of one skill, the learners' and the skills' intercepts are collinear: the last skill's is left out,
  # then the intercepts are moved so that the learners' mean is 0, as the package reports them; no prediction changes.
  flat <- learners && all(lengths(skills) == 1)
  if (flat) design <- design[, colnames(design) != paste0("intercept:", SKILLS[length(SKILLS)])]
  fit <- glm(events$outcome ~ 0 + design, family = binomial, control = glm.control(epsilon = 1e-14, maxit = 100))
  score <- crossprod(design, events$outcome - fitted(fit))
  stopifnot(fit$converged, max(abs(score)) < 1e-9)
  b <- setNames(coef(fit), colnames(design))
  stopifnot(!anyNA(b))
  theta <- if (learners) b[paste0("learner", levels(learner))] else NULL
  intercepts <- setNames(sapply(paste0("intercept:", SKILLS), function(name) if (name %in% names(b)) b[[name]] else 0),
                         paste0("intercept:", SKILLS))
  if (flat) {
    shift <- mean(theta)
    theta <- theta - shift
    intercepts <- intercepts + shift
  }
  list(
    skills = setNames(lapply(SKILLS, function(k) list(
      intercept = unname(intercepts[[paste0("intercept:", k)]]),
      coefficients = setNames(lapply(values, function(v) unname(b[[paste0(v, ":", k)]])), labels_of(values))
    )), SKILLS),
    learners = if (learners) setNames(as.list(unname(theta)), levels(learner)) else NULL,
    logLikelihood = as.numeric(logLik(fit))
  )
}
labels_of <- function(values) {
  sapply(values, function(v) switch(v, decayedLogit = sprintf("decayedLogit(%s)", DECAYS$logit),
                                    recency = sprintf("recency(%s)", DECAYS$recency), v))
}
# The events of one skill, from the learners with both outcomes among them: under fixed proficiency, a learner whose
# outcomes are all equal has no finite estimate.
mixed <- function(events) {
  both <- tapply(events$outcome, events$learner, function(y) length(unique(y)) == 2)
  events$learner %in% names(both)[both]
}
stopifnot(all(mixed(simulated)))
one_skill <- which(lengths(simulated_skills) == 1)
single <- one_skill[mixed(simulated[one_skill, ])]
single_events <- simulated[single, ]

# The Laplace approximation of a random intercept model: glmer's estimates, then the maximum of the approximation in base
# R, with each learner's conditional mode by Newton's method to convergence; lme4 solves its inner problem to about 1e-7
# only. At glmer's estimates and at the maximum, the base R approximation equals lme4's deviance function.
glmer_fit <- function(events, skills, values) {
  x <- design_of(events, skills, values)
  design <- do.call(cbind, lapply(names(x), function(n) { m <- x[[n]]; colnames(m) <- paste0(n, ":", SKILLS); m }))
  learner <- factor(events$learner)
  y <- events$outcome
  fit <- suppressWarnings(glmer(y ~ 0 + design + (1 | learner), family = binomial, nAGQ = 1))
  deviance <- glmer(y ~ 0 + design + (1 | learner), family = binomial, nAGQ = 1, devFunOnly = TRUE)
  modes_at <- function(sd, beta) {
    eta0 <- drop(design %*% beta)
    modes <- setNames(rep(0, nlevels(learner)), levels(learner))
    for (iteration in 1:100) {
      p <- plogis(eta0 + modes[learner])
      step <- (tapply(y - p, learner, sum) - modes / sd^2) / (tapply(p * (1 - p), learner, sum) + 1 / sd^2)
      modes <- modes + step
      if (max(abs(step)) < 1e-15) break
    }
    modes
  }
  laplace <- function(par) {
    sd <- exp(par[1])
    beta <- par[-1]
    modes <- modes_at(sd, beta)
    p <- plogis(drop(design %*% beta) + modes[learner])
    h <- tapply(p * (1 - p), learner, sum)
    sum(dbinom(y, 1, p, log = TRUE)) - sum(modes^2) / (2 * sd^2) - sum(log1p(sd^2 * h)) / 2
  }
  start <- c(log(getME(fit, "theta")), fixef(fit))
  stopifnot(abs(laplace(start) - as.numeric(logLik(fit))) < 1e-8)
  # Central differences of the exact approximation for the gradient: the maximum is flat, and derivative-free steps
  # stop short of it.
  gradient <- function(par) {
    sapply(seq_along(par), function(j) {
      step <- 1e-5 * max(1, abs(par[j]))
      up <- par
      down <- par
      up[j] <- up[j] + step
      down[j] <- down[j] - step
      (laplace(down) - laplace(up)) / (2 * step)
    })
  }
  optimum <- nlminb(start, function(par) -laplace(par), gradient,
                    control = list(rel.tol = 1e-15, x.tol = 1e-15, eval.max = 1e5, iter.max = 1e5))
  # Newton steps on the differenced gradient, which nlminb's stopping rules leave short of the maximum.
  par <- optimum$par
  for (iteration in 1:10) {
    g <- gradient(par)
    if (max(abs(g)) < 1e-7) break
    par <- par - solve(optimHess(par, function(x) -laplace(x), gradient), g)
  }
  stopifnot(max(abs(gradient(par))) < 1e-6)
  optimum <- list(par = par, objective = -laplace(par))
  sd <- exp(optimum$par[1])
  beta <- optimum$par[-1]
  stopifnot(abs(-deviance(c(sd, beta)) / 2 + optimum$objective) < 1e-8)
  b <- setNames(beta, colnames(design))
  list(
    skills = setNames(lapply(SKILLS, function(k) list(
      intercept = unname(b[[paste0("intercept:", k)]]),
      coefficients = setNames(lapply(values, function(v) unname(b[[paste0(v, ":", k)]])), labels_of(values))
    )), SKILLS),
    learners = as.list(modes_at(sd, beta)),
    learnerSd = sd,
    logLikelihood = -optimum$objective,
    glmer = list(sd = unname(getME(fit, "theta")), logLikelihood = as.numeric(logLik(fit)),
                 intercept = unname(fixef(fit)[["designintercept:s1"]]))
  )
}

simulated_events <- function(events, skills) events_of(events$learner, events$item, skills, events$time, events$outcome)
write_fixture("logistic-glm", list(
  source = "set.seed(20261029): 80 learners, 40 events each, items of one or two of 5 skills; glm and lme4.",
  decays = DECAYS,
  events = simulated_events(simulated, simulated_skills),
  # The events of one skill, from the learners with both outcomes among them, as 0-based positions in `events`.
  singleSkill = single - 1,
  fits = list(
    list(name = "AFM, fixed proficiency", features = list("opportunities"), proficiency = "fixed",
         expected = glm_fit(simulated, simulated_skills, "opportunities", TRUE)),
    list(name = "AFM, fixed proficiency, one skill per event", features = list("opportunities"), proficiency = "fixed",
         singleSkill = TRUE,
         expected = glm_fit(single_events, simulated_skills[single], "opportunities", TRUE)),
    list(name = "PFA", features = list("successes", "failures"), proficiency = "none",
         expected = glm_fit(simulated, simulated_skills, c("successes", "failures"), FALSE)),
    list(name = "LKT features, fixed proficiency", features = list(list(decayedLogit = DECAYS$logit),
                                                                     list(recency = DECAYS$recency)),
         proficiency = "fixed",
         expected = glm_fit(simulated, simulated_skills, c("decayedLogit", "recency"), TRUE)),
    list(name = "AFM, random intercept", features = list("opportunities"), proficiency = "random",
         expected = glmer_fit(simulated, simulated_skills, "opportunities")),
    list(name = "PFA, random intercept", features = list("successes", "failures"), proficiency = "random",
         expected = glmer_fit(simulated, simulated_skills, c("successes", "failures")))
  )
))
