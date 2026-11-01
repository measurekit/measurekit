# Logistic models with terms beyond the skill: item intercepts, coefficients shared across skills, terms over items and the
# learner, covariates, and standard errors. Fits by glm (with the learners as sum-contrast columns where fixed proficiency
# is not identified apart from the intercepts), by the Laplace maximum that lme4's deviance defines, and with a ridge by
# LiblineaR, which Gervet et al.'s Best-LR is fitted with. Standard errors are those of summary(glm), and for a random
# intercept the inverse Schur complement that vcov(glmer, use.hessian = FALSE) is.

source("validation/knowledge-tracing/common.R")
suppressPackageStartupMessages({
  library(lme4)
  library(LiblineaR)
})

# --- A seeded log: 60 learners, 36 events each, items of one skill and items of two --------------------------------

set.seed(20261101)
SKILLS <- paste0("s", 1:6)
ITEMS <- c(setNames(as.list(rep(SKILLS, each = 3)), paste0("i", sprintf("%02d", 1:18))),
           list(i19 = c("s1", "s2"), i20 = c("s2", "s3"), i21 = c("s3", "s4"), i22 = c("s4", "s5"), i23 = c("s5", "s6"),
                i24 = c("s6", "s1")))
learner_count <- 60
true_theta <- rnorm(learner_count, 0, 0.7)
true_item <- setNames(rnorm(length(ITEMS), 0, 0.5), names(ITEMS))
true_skill <- setNames(rnorm(6, -0.2, 0.3), SKILLS)
events <- do.call(rbind, lapply(seq_len(learner_count), function(l) {
  data.frame(learner = sprintf("learner%02d", l), item = sample(names(ITEMS), 36, replace = TRUE),
             time = cumsum(rexp(36, 1 / 60)) + l, x = round(rnorm(36), 3))
}))
events <- events[order(events$time), ]
event_skills <- unname(ITEMS[events$item])
counts <- new.env()
events$outcome <- sapply(seq_len(nrow(events)), function(i) {
  skills <- event_skills[[i]]
  key <- paste(events$learner[i], c(skills, "item", events$item[i]), sep = "\r")
  opportunities <- sapply(key, function(k) counts[[k]] %||% 0)
  for (k in key) counts[[k]] <- (counts[[k]] %||% 0) + 1
  theta <- true_theta[as.integer(sub("learner", "", events$learner[i]))]
  eta <- theta + true_item[[events$item[i]]] + sum(true_skill[skills]) + 0.08 * sum(log1p(opportunities[seq_along(skills)])) +
    0.3 * events$x[i]
  rbinom(1, 1, plogis(eta))
})
row.names(events) <- NULL

# --- The history of each event, written out from the definitions ----------------------------------------------------

# For each event the successes and failures before it, of the learner at each of its keys: `keys` lists the keys of each event.
history <- function(keys) {
  n <- nrow(events)
  state <- new.env()
  successes <- vector("list", n)
  failures <- vector("list", n)
  for (i in seq_len(n)) {
    names <- paste(events$learner[i], keys[[i]], sep = "\r")
    before <- lapply(names, function(k) state[[k]] %||% c(0, 0))
    successes[[i]] <- sapply(before, `[`, 1)
    failures[[i]] <- sapply(before, `[`, 2)
    for (j in seq_along(names)) state[[names[j]]] <- before[[j]] + c(events$outcome[i], 1 - events$outcome[i])
  }
  list(successes = successes, failures = failures)
}
histories <- list(
  skill = history(event_skills),
  item = history(as.list(events$item)),
  learner = history(rep(list("all"), nrow(events)))
)
feature_value <- function(feature, s, f) {
  switch(feature, logSuccesses = log1p(s), logFailures = log1p(f), opportunities = s + f, successes = s, failures = f,
         stop("unknown feature"))
}

# --- The design of a model's options, one column per coefficient ----------------------------------------------------

term <- function(feature, over = "skill", coefficient = if (over == "learner") "shared" else "each") {
  list(feature = feature, over = over, coefficient = coefficient)
}
label_of <- function(t) if (t$coefficient == "each") t$feature else paste0(t$over, ":", t$feature)

# The columns: names say what each is for, so that estimates come back in the shape of measurekit's parameters.
design_of <- function(options, learners = "none") {
  n <- nrow(events)
  items <- names(ITEMS)
  columns <- list()
  indicator <- function(level, ids) {
    m <- matrix(0, n, length(ids), dimnames = list(NULL, paste0(level, ":", ids)))
    for (i in seq_len(n)) m[i, paste0(level, ":", if (level == "skill") event_skills[[i]] else events$item[i])] <- 1
    m
  }
  if ("skill" %in% options$intercepts) columns$skillIntercept <- indicator("skill", SKILLS)
  if ("item" %in% options$intercepts) columns$itemIntercept <- indicator("item", items)
  shared <- list()
  for (t in options$features) {
    h <- histories[[t$over]]
    values <- mapply(feature_value, t$feature, h$successes, h$failures, SIMPLIFY = FALSE, USE.NAMES = FALSE)
    if (t$coefficient == "shared") {
      shared[[label_of(t)]] <- sapply(values, sum)
    } else {
      ids <- if (t$over == "skill") SKILLS else items
      m <- matrix(0, n, length(ids), dimnames = list(NULL, paste0(t$over, "-each:", t$feature, ":", ids)))
      for (i in seq_len(n)) {
        at <- if (t$over == "skill") event_skills[[i]] else events$item[i]
        m[i, paste0(t$over, "-each:", t$feature, ":", at)] <- values[[i]]
      }
      columns[[paste0(t$over, t$feature)]] <- m
    }
  }
  for (label in names(shared)) columns[[label]] <- matrix(shared[[label]], n, 1, dimnames = list(NULL, paste0("shared:", label)))
  for (name in options$covariates) {
    columns[[name]] <- matrix(events[[name]], n, 1, dimnames = list(NULL, paste0("shared:covariate:", name)))
  }
  learner <- factor(events$learner)
  if (learners == "dummy") columns$learner <- model.matrix(~ 0 + learner, data.frame(learner = learner))
  if (learners == "sum") {
    contrasts(learner) <- contr.sum(nlevels(learner))
    columns$learner <- model.matrix(~ learner)[, -1, drop = FALSE]
  }
  do.call(cbind, columns)
}

# The coefficients, in measurekit's parameters: skills, items, shared and learners, from named estimates.
parameters_of <- function(values, options, learner_theta = NULL) {
  at <- function(name) if (name %in% names(values)) unname(values[[name]]) else 0
  block <- function(level, ids, intercept) {
    labels <- as.character(unlist(lapply(Filter(function(t) t$over == level && t$coefficient == "each", options$features), `[[`, "feature")))
    setNames(lapply(ids, function(id) list(
      intercept = if (intercept) at(paste0(level, ":", id)) else 0,
      coefficients = setNames(lapply(labels, function(feature) at(paste0(level, "-each:", feature, ":", id))), labels)
    )), ids)
  }
  items_used <- "item" %in% options$intercepts || any(sapply(options$features, function(t) t$over == "item" && t$coefficient == "each"))
  shared_names <- grep("^shared:", names(values), value = TRUE)
  c(
    list(skills = block("skill", SKILLS, "skill" %in% options$intercepts)),
    if (items_used) list(items = block("item", names(ITEMS), "item" %in% options$intercepts)),
    if (length(shared_names) > 0) list(shared = setNames(lapply(shared_names, function(n) unname(values[[n]])), sub("^shared:", "", shared_names))),
    if (!is.null(learner_theta)) list(learners = as.list(learner_theta))
  )
}

# The options in the form logistic() takes: terms as objects, intercepts, covariates and the proficiency.
options_json <- function(options, proficiency, ridge = NULL) {
  c(list(features = lapply(options$features, function(t) lapply(t, identity)), intercepts = I(options$intercepts),
         proficiency = proficiency),
    if (length(options$covariates) > 0) list(covariates = I(options$covariates)),
    if (!is.null(ridge)) list(ridge = ridge))
}

# --- Fits by glm ----------------------------------------------------------------------------------------------------

# glm at a tight tolerance, with standard errors from summary(glm). With fixed proficiency the learners are columns of
# sum contrasts when the intercepts leave a flat direction (every event has the same number of intercepts), so that their
# mean is 0 as measurekit centers them; the last learner's value and error follow from the others.
glm_case <- function(name, options, proficiency) {
  flat <- proficiency == "fixed" && (all(lengths(event_skills) == 1) || identical(options$intercepts, "item"))
  design <- design_of(options, if (proficiency == "none") "none" else if (flat) "sum" else "dummy")
  fit <- glm(events$outcome ~ 0 + design, family = binomial, control = glm.control(epsilon = 1e-14, maxit = 100))
  stopifnot(fit$converged, !anyNA(coef(fit)), max(abs(crossprod(design, events$outcome - fitted(fit)))) < 1e-9)
  b <- setNames(coef(fit), colnames(design))
  v <- vcov(fit)
  dimnames(v) <- list(colnames(design), colnames(design))
  estimates <- b
  errors <- setNames(sqrt(diag(v)), colnames(design))
  theta <- NULL
  theta_errors <- NULL
  if (proficiency != "none") {
    levels <- sort(unique(events$learner))
    named <- grep("^learner", names(b), value = TRUE)
    if (flat) {
      free <- b[named]
      weights <- rep(1, length(free))
      theta <- c(free, -sum(free))
      theta_errors <- c(sqrt(diag(v)[named]), sqrt(drop(weights %*% v[named, named] %*% weights)))
    } else {
      theta <- b[named]
      theta_errors <- sqrt(diag(v)[named])
    }
    names(theta) <- names(theta_errors) <- levels
    estimates <- b[!names(b) %in% named]
    errors <- errors[!names(errors) %in% named]
  }
  list(
    name = name, options = options_json(options, proficiency),
    expected = c(parameters_of(estimates, options, theta), list(logLikelihood = as.numeric(logLik(fit)))),
    standardErrors = parameters_of(errors, options, theta_errors)
  )
}

# --- The Laplace maximum for a random intercept, and the inverse Schur complement ---------------------------------

# The maximum of the Laplace approximation of lme4's deviance over the fixed effects and the standard deviation, in
# base R, and the standard errors of the fixed effects as vcov(glmer, use.hessian = FALSE) has them: the inverse of the
# Schur complement of the learners' block of the penalized Hessian.
random_case <- function(name, options) {
  design <- design_of(options)
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
  schur_errors <- function(sd, beta) {
    modes <- modes_at(sd, beta)
    p <- plogis(drop(design %*% beta) + modes[learner])
    w <- as.vector(p * (1 - p))
    weighted <- design * w
    b <- rowsum(weighted, learner)
    a <- c(tapply(w, learner, sum)) + 1 / sd^2
    sqrt(diag(solve(crossprod(design, weighted) - crossprod(b, b / a[rownames(b)]))))
  }
  start <- c(log(getME(fit, "theta")), fixef(fit))
  stopifnot(abs(laplace(start) - as.numeric(logLik(fit))) < 1e-8)
  # lme4's own standard errors at its own estimates, which the written-out Schur complement must reproduce.
  lme4_errors <- sqrt(diag(as.matrix(vcov(fit, use.hessian = FALSE))))
  mine <- schur_errors(exp(start[1]), start[-1])
  stopifnot(max(abs(lme4_errors / mine - 1)) < 1e-4)
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
  par <- optimum$par
  for (iteration in 1:10) {
    g <- gradient(par)
    if (max(abs(g)) < 1e-7) break
    par <- par - solve(optimHess(par, function(x) -laplace(x), gradient), g)
  }
  stopifnot(max(abs(gradient(par))) < 1e-6)
  sd <- exp(par[1])
  beta <- setNames(par[-1], colnames(design))
  stopifnot(abs(-deviance(c(sd, beta)) / 2 - laplace(par)) < 1e-8)
  errors <- setNames(schur_errors(sd, beta), colnames(design))
  list(
    name = name, options = options_json(options, "random"),
    expected = c(parameters_of(beta, options, modes_at(sd, beta)), list(learnerSd = sd, logLikelihood = laplace(par))),
    standardErrors = parameters_of(errors, options)
  )
}

# --- Ridge fits by LiblineaR ----------------------------------------------------------------------------------------

# LiblineaR's L2-regularized logistic regression (type 0) without a bias term at cost 1 / ridge, to a tight tolerance,
# finished by Newton steps on the same objective as in logistic.R: the minimum of -log-likelihood + ridge / 2 |w|².
ridge_case <- function(name, options, proficiency, ridge) {
  design <- design_of(options, if (proficiency == "none") "none" else "dummy")
  y <- events$outcome
  fit <- LiblineaR(design, y, type = 0, cost = 1 / ridge, epsilon = 1e-12, bias = -1)
  w <- fit$W[1, seq_len(ncol(design))]
  if (fit$ClassNames[1] == 0) w <- -w
  liblinear_gap <- NULL
  start <- w
  for (iteration in 1:30) {
    p <- plogis(drop(design %*% w))
    gradient <- drop(crossprod(design, p - y)) + ridge * w
    if (max(abs(gradient)) < 1e-12) break
    w <- w - solve(crossprod(design, design * (p * (1 - p))) + ridge * diag(ncol(design)), gradient)
  }
  stopifnot(max(abs(gradient)) < 1e-10)
  b <- setNames(w, colnames(design))
  named <- grep("^learner", names(b), value = TRUE)
  theta <- if (proficiency == "none") NULL else setNames(b[named], sort(unique(events$learner)))
  p <- plogis(drop(design %*% w))
  list(
    name = name, options = options_json(options, proficiency, ridge),
    expected = c(parameters_of(b[!names(b) %in% named], options, theta), list(logLikelihood = sum(dbinom(y, 1, p, log = TRUE)))),
    liblinearGap = max(abs(start - w))
  )
}

models <- list(
  items = list(intercepts = "item", features = list(), covariates = character()),
  sharedAcrossSkills = list(intercepts = "skill", features = list(term("logSuccesses", coefficient = "shared"),
                                                                    term("logFailures", coefficient = "shared")),
                            covariates = character()),
  itemAndLearner = list(intercepts = "skill",
                        features = list(term("opportunities", over = "item", coefficient = "shared"),
                                        term("logSuccesses", over = "learner"), term("logFailures", over = "learner")),
                        covariates = character()),
  covariate = list(intercepts = "skill", features = list(term("opportunities")), covariates = "x"),
  randomItems = list(intercepts = "item", features = list(), covariates = "x"),
  bestLr = list(intercepts = c("skill", "item"),
                features = list(term("logSuccesses"), term("logFailures"), term("logSuccesses", over = "learner"),
                                term("logFailures", over = "learner")),
                covariates = character()),
  itemSlopes = list(intercepts = "skill", features = list(term("opportunities", over = "item")), covariates = character())
)
one_skill <- all(lengths(event_skills) == 1)
stopifnot(!one_skill)

# Every learner and every item has both outcomes among their events, so that no estimate runs off to infinity.
stopifnot(all(tapply(events$outcome, events$learner, function(y) length(unique(y)) == 2)),
          all(tapply(events$outcome, events$item, function(y) length(unique(y)) == 2)))

# The events of one skill each, with the learners who have both outcomes among them: fixed proficiency has a flat direction.
single <- which(lengths(event_skills) == 1)
single_events <- events[single, ]
stopifnot(all(tapply(single_events$outcome, single_events$learner, function(y) length(unique(y)) == 2)))

# glm for the one-skill-per-event case runs on its own events: the design functions read the global events, so swap them.
all_events <- events
all_skills <- event_skills
all_histories <- histories
with_events <- function(subset, skills, run) {
  events <<- subset
  event_skills <<- skills
  histories <<- list(
    skill = history(skills),
    item = history(as.list(subset$item)),
    learner = history(rep(list("all"), nrow(subset)))
  )
  on.exit({
    events <<- all_events
    event_skills <<- all_skills
    histories <<- all_histories
  })
  run()
}
single_case <- with_events(single_events, all_skills[single], function() {
  glm_case("skill intercepts of events with one skill, fixed proficiency",
           list(intercepts = "skill", features = list(), covariates = character()), "fixed")
})

fits <- list(
  glm_case("item intercepts", models$items, "none"),
  glm_case("item intercepts, fixed proficiency", models$items, "fixed"),
  glm_case("coefficients shared across skills", models$sharedAcrossSkills, "none"),
  glm_case("shared terms over items and the learner", models$itemAndLearner, "none"),
  glm_case("a covariate with skill intercepts and a slope, fixed proficiency", models$covariate, "fixed"),
  random_case("item intercepts and a covariate, random proficiency", models$randomItems),
  ridge_case("Best-LR: skill and item intercepts, fixed proficiency, ridge", models$bestLr, "fixed", 1),
  ridge_case("a coefficient for each item, ridge", models$itemSlopes, "none", 0.5)
)
fits[[length(fits) + 1]] <- c(single_case, list(subset = single - 1))

write_fixture("logistic-terms", list(
  source = "set.seed(20261101): 60 learners, 36 events each at 24 items of 6 skills, with a covariate; glm, lme4 and LiblineaR.",
  events = lapply(seq_len(nrow(all_events)), function(i) {
    list(learner = all_events$learner[i], item = all_events$item[i], skills = I(all_skills[[i]]), time = all_events$time[i],
         outcome = all_events$outcome[i], covariates = list(x = all_events$x[i]))
  }),
  fits = fits
))
