# Item response theory fixtures: calibration, ability estimation and information for the logistic and polytomous
# models, mixed-format calibration, and joint maximum likelihood. References: mirt (calibration, fscores, iteminfo,
# testinfo), ltm (Gauss-Hermite calibration), TAM (tam.jml), and base R written out below: direct maximization of the
# marginal posterior (Bock-Lieberman), the delta method for polytomous standard errors, Warm's weighted likelihood
# estimate, and the JML estimating equations. catR and eRm cross-check.

source("validation/psychometrics/common.R")
suppressPackageStartupMessages({
  library(mirt)
  library(ltm)
  library(catR)
  library(TAM)
  library(eRm)
})

POINTS <- 61 # mirt's default quadrature: 61 equally spaced nodes on [-6, 6]
TOL <- 1e-10 # mirt's EM tolerance, tight so that its estimates are the optimum
RANGE <- c(-6, 6) # measurekit's default theta range for ML, WLE and MAP
INFORMATION_THETA <- seq(-6, 6, by = 0.5)
LOGISTIC <- c("1pl", "2pl", "3pl", "4pl")
# Models whose Jeffreys prior gives Warm's estimator, so that mirt's WLE is a reference: the exponential family, and the
# GRM, whose categories have Σ P'³ / P² = Σ P' P'' / P, so that ½ d ln I / dθ is Warm's J / (2 I).
WARM_MODELS <- c("1pl", "2pl", "pcm", "gpcm", "rsm", "nrm", "grm")

fit_mirt <- function(data, model, itemtype, se_type, ...) {
  fit <- mirt(data, model, itemtype = itemtype, SE = TRUE, SE.type = se_type, TOL = TOL, quadpts = POINTS,
              verbose = FALSE, technical = list(NCYCLES = 1e5), ...)
  stopifnot(extract.mirt(fit, "converged"))
  fit
}

# --- measurekit's parameters of mirt's items ----------------------------------------------------------------------
# Logistic items: coef(IRTpars = TRUE), which applies the delta method to mirt's slope-intercept and logit scale.
# Polytomous items: measurekit's parameters computed from mirt's slope-intercept values, and their standard errors by
# the delta method from vcov(), with the Jacobian by central differences. mirt 1.48's coef(IRTpars = TRUE,
# printSE = TRUE) is not used for them: its GPCM b1 standard error disagrees with the delta method on its own
# covariance matrix.

item_values <- function(values, item) {
  rows <- values[values$item == item, ]
  setNames(rows$value, rows$name)
}

# measurekit's parameters of a polytomous item from its named mirt values.
polytomous_parameters <- function(model, v) {
  numbered <- function(prefix) v[grep(paste0("^", prefix, "[0-9]+$"), names(v))]
  switch(model,
    grm = list(a = v[["a1"]], b = unname(-numbered("d") / v[["a1"]])),
    gpcm = , pcm = list(a = v[["a1"]], b = unname(-diff(numbered("d")) / v[["a1"]])),
    nrm = {
      a <- unname(v[["a1"]] * numbered("ak"))
      d <- unname(numbered("d"))
      list(a = a - mean(a), c = d - mean(d))
    },
    rsm = {
      # mirt's rsm: P(k) ∝ exp(a1 (k θ - b1 - ... - bk) + k c), thresholds b_j - c / a1.
      thresholds <- unname(numbered("b") - v[["c"]] / v[["a1"]])
      list(a = v[["a1"]], b = mean(thresholds), steps = thresholds - mean(thresholds))
    }
  )
}

# measurekit's estimated parameters of a polytomous item, named as its vector entries.
estimated_parameters <- function(model, p) {
  k <- length(p$a) # categories of an NRM item
  switch(model,
    grm = , gpcm = , pcm = setNames(c(p$a, p$b), c("a", paste0("b", seq_along(p$b)))),
    nrm = setNames(c(p$a[-k], p$c[-k]), c(paste0("a", seq_len(k - 1)), paste0("c", seq_len(k - 1)))),
    rsm = {
      free <- head(p$steps, -1)
      setNames(c(p$a, p$b, free), c("a", "b", if (length(free) > 0) paste0("step", seq_along(free))))
    }
  )
}

# Standard errors of the estimated parameters of the polytomous items, by item (NULL for logistic items), by the delta
# method from mirt's covariance matrix of its own parameters.
delta_errors <- function(fit, items, models) {
  covariance <- vcov(fit)
  values <- mod2values(fit)
  # A column of a constrained parameter names all its parameter numbers: "a1.1.6.11.16".
  parnums <- lapply(strsplit(colnames(covariance), ".", fixed = TRUE), function(x) as.integer(x[-1]))
  base <- sapply(parnums, function(p) values$value[values$parnum == p[1]])
  estimates <- function(x) {
    for (j in seq_along(x)) values$value[values$parnum %in% parnums[[j]]] <- x[j]
    lapply(seq_along(items), function(i) {
      if (models[i] %in% LOGISTIC) return(NULL)
      estimated_parameters(models[i], polytomous_parameters(models[i], item_values(values, items[i])))
    })
  }
  jacobian <- sapply(seq_along(base), function(j) {
    h <- 1e-6 * max(1, abs(base[j]))
    up <- base
    down <- base
    up[j] <- up[j] + h
    down[j] <- down[j] - h
    (unlist(estimates(up)) - unlist(estimates(down))) / (2 * h)
  })
  errors <- sqrt(diag(jacobian %*% covariance %*% t(jacobian)))
  centers <- estimates(base)
  ends <- cumsum(lengths(centers))
  lapply(seq_along(centers), function(i) {
    if (length(centers[[i]]) == 0) return(NULL)
    setNames(errors[(ends[i] - length(centers[[i]]) + 1):ends[i]], names(centers[[i]]))
  })
}

# measurekit's items of a mirt fit: parameters and the standard errors of the estimated ones, by item. Fixed items have
# no standard errors.
psychometrics_items <- function(fit, models) {
  items <- colnames(extract.mirt(fit, "data"))
  values <- mod2values(fit)
  coefficients <- coef(fit, IRTpars = TRUE, printSE = TRUE)
  polytomous <- if (any(!models %in% LOGISTIC)) delta_errors(fit, items, models)
  lapply(seq_along(items), function(i) {
    estimated <- any(values$est[values$item == items[i]])
    if (models[i] %in% LOGISTIC) {
      x <- coefficients[[i]]
      se <- if ("SE" %in% rownames(x)) x["SE", c("a", "b", "g", "u")] else rep(NA, 4)
      names(se) <- c("a", "b", "c", "d")
      parameters <- list(a = x["par", "a"], b = x["par", "b"], c = x["par", "g"], d = x["par", "u"])
      errors <- as.list(se[!is.na(se)])
    } else {
      parameters <- polytomous_parameters(models[i], item_values(values, items[i]))
      parameters <- lapply(parameters, function(x) if (length(x) > 1) I(x) else x)
      errors <- as.list(polytomous[[i]])
    }
    if (estimated) list(parameters = parameters, standardErrors = errors) else list(parameters = parameters)
  })
}

mirt_calibration <- function(fit, models) {
  list(models = I(models), items = psychometrics_items(fit, models), logLikelihood = logLik(fit),
       aic = extract.mirt(fit, "AIC"), bic = extract.mirt(fit, "BIC"))
}

parameter_matrix <- function(items) t(sapply(items, function(item) unlist(item$parameters[c("a", "b", "c", "d")])))

# --- Base R marginal likelihood -------------------------------------------------------------------------------
# The marginal log-likelihood of logistic items with parameters (a, b, c, d) by rows, mirt's rectangular quadrature,
# and its gradient by Fisher's identity: the posterior expectation of the complete-data gradient.

rectangular <- function(points) {
  nodes <- seq(-6, 6, length.out = points)
  list(nodes = nodes, weights = dnorm(nodes) / sum(dnorm(nodes)))
}

marginal <- function(x, pars, rule = rectangular(POINTS)) {
  x <- as.matrix(x)
  correct <- ifelse(is.na(x), 0, x)
  wrong <- ifelse(is.na(x), 0, 1 - x)
  z <- outer(rule$nodes, pars[, 1]) - matrix(pars[, 1] * pars[, 2], length(rule$nodes), nrow(pars), byrow = TRUE)
  logistic <- plogis(z)
  lower <- matrix(pars[, 3], nrow(z), ncol(z), byrow = TRUE)
  range <- matrix(pars[, 4] - pars[, 3], nrow(z), ncol(z), byrow = TRUE)
  p <- lower + range * logistic
  log_joint <- correct %*% t(log(p)) + wrong %*% t(log(1 - p)) + matrix(log(rule$weights), nrow(x), nrow(z), byrow = TRUE)
  largest <- apply(log_joint, 1, max)
  posterior <- exp(log_joint - largest)
  totals <- rowSums(posterior)
  posterior <- posterior / totals
  expected_correct <- t(posterior) %*% correct
  expected_wrong <- t(posterior) %*% wrong
  residual <- expected_correct / p - expected_wrong / (1 - p)
  slope <- range * logistic * (1 - logistic)
  theta <- matrix(rule$nodes, nrow(z), ncol(z))
  gradient <- cbind(
    colSums(residual * slope * (theta - matrix(pars[, 2], nrow(z), ncol(z), byrow = TRUE))),
    colSums(residual * -slope * matrix(pars[, 1], nrow(z), ncol(z), byrow = TRUE)),
    colSums(residual * (1 - logistic)),
    colSums(residual * logistic)
  )
  list(logLikelihood = sum(largest + log(totals)), gradient = gradient)
}

# Maximizes the marginal log-likelihood plus the log prior of c over the free columns of `start` (a, b and c), with
# c on the logit scale for the optimizer, by BFGS from `start`. Returns the estimates, the log-likelihood and the
# standard errors from the Hessian of the log posterior on the parameters' own scale (optimHess).
posterior_mode <- function(x, start, log_prior, prior_gradient) {
  k <- nrow(start)
  to_pars <- function(theta) {
    m <- matrix(theta, k, 3)
    cbind(m[, 1:2], plogis(m[, 3]), 1)
  }
  objective <- function(theta) {
    pars <- to_pars(theta)
    -(marginal(x, pars)$logLikelihood + sum(log_prior(pars[, 3])))
  }
  gradient <- function(theta) {
    pars <- to_pars(theta)
    g <- marginal(x, pars)$gradient
    g[, 3] <- (g[, 3] + prior_gradient(pars[, 3])) * pars[, 3] * (1 - pars[, 3])
    -as.vector(g[, 1:3])
  }
  start_theta <- c(start[, 1], start[, 2], qlogis(start[, 3]))
  result <- optim(start_theta, objective, gradient, method = "BFGS", control = list(reltol = 1e-16, maxit = 10000))
  natural_objective <- function(v) {
    p <- cbind(matrix(v, k, 3), 1)
    -(marginal(x, p)$logLikelihood + sum(log_prior(p[, 3])))
  }
  natural_gradient <- function(v) {
    p <- cbind(matrix(v, k, 3), 1)
    g <- marginal(x, p)$gradient
    g[, 3] <- g[, 3] + prior_gradient(p[, 3])
    -as.vector(g[, 1:3])
  }
  # BFGS stops on the relative change of the objective; Newton steps take the gradient to its rounding level.
  estimate <- as.vector(to_pars(result$par)[, 1:3])
  for (step in 1:4) {
    estimate <- estimate - solve(optimHess(estimate, natural_objective, natural_gradient), natural_gradient(estimate))
  }
  stopifnot(max(abs(natural_gradient(estimate))) < 1e-8)
  pars <- cbind(matrix(estimate, k, 3), 1)
  covariance <- solve(optimHess(estimate, natural_objective, natural_gradient))
  se <- matrix(sqrt(diag(covariance)), k, 3)
  list(
    items = lapply(seq_len(k), function(i) {
      list(parameters = list(a = pars[i, 1], b = pars[i, 2], c = pars[i, 3], d = 1),
           standardErrors = list(a = se[i, 1], b = se[i, 2], c = se[i, 3]))
    }),
    logLikelihood = marginal(x, pars)$logLikelihood,
    logPosterior = -natural_objective(estimate)
  )
}

# --- Abilities ----------------------------------------------------------------------------------------------------

response_patterns <- function(item_count, categories = 2) {
  patterns <- as.matrix(expand.grid(rep(list(seq_len(categories) - 1), item_count)))
  dimnames(patterns) <- NULL
  patterns
}

# Category probabilities P and their first and second theta derivatives P1 and P2, theta by category, of an item with
# measurekit's parameters, written out for each model.
item_trace <- function(model, p, theta) {
  if (model %in% LOGISTIC) {
    l <- plogis(p$a * (theta - p$b))
    range <- p$d - p$c
    first <- range * p$a * l * (1 - l)
    second <- first * p$a * (1 - 2 * l)
    return(list(P = cbind(1 - p$c - range * l, p$c + range * l), P1 = cbind(-first, first), P2 = cbind(-second, second)))
  }
  if (model == "grm") {
    k <- length(p$b) + 1
    s <- matrix(plogis(p$a * outer(theta, p$b, "-")), length(theta))
    cumulative <- cbind(1, s, 0)
    first <- cbind(0, p$a * s * (1 - s), 0)
    second <- cbind(0, p$a^2 * s * (1 - s) * (1 - 2 * s), 0)
    difference <- function(x) x[, 1:k, drop = FALSE] - x[, 2:(k + 1), drop = FALSE]
    return(list(P = difference(cumulative), P1 = difference(first), P2 = difference(second)))
  }
  # Divide-by-total models: P_k ∝ exp(z_k), z_k linear in theta with slope s_k.
  if (model == "nrm") {
    slopes <- p$a
    z <- outer(theta, p$a) + matrix(p$c, length(theta), length(p$c), byrow = TRUE)
  } else {
    thresholds <- if (model == "rsm") p$b + p$steps else p$b
    slopes <- p$a * (0:length(thresholds))
    z <- outer(theta, slopes) - matrix(p$a * cumsum(c(0, thresholds)), length(theta), length(slopes), byrow = TRUE)
  }
  P <- exp(z - apply(z, 1, max))
  P <- P / rowSums(P)
  deviation <- outer(-drop(P %*% slopes), slopes, "+")
  variance <- rowSums(P * deviation^2)
  list(P = P, P1 = P * deviation, P2 = P * (deviation^2 - variance))
}

# Information of the answered items at theta, from mirt's iteminfo.
answered_information <- function(fit, pattern, theta) {
  sum(sapply(which(!is.na(pattern)), function(i) iteminfo(extract.item(fit, i), theta)))
}

# Warm's weighted likelihood estimate: the maximum in RANGE of the weighted log-likelihood, whose derivative is the
# estimating function d log L / dθ + J / (2I), J = Σ P'P'' / P over the answered items and their categories. The
# candidates are the roots where the function turns from positive to negative and the ends it points to; the weighted
# log-likelihood at each, relative to the lower end, is the integral of the estimating function. Returns the estimate
# and the number of roots.
warm <- function(models, parameters, pattern) {
  answered <- which(!is.na(pattern))
  estimating <- Vectorize(function(theta) {
    slope <- 0
    j <- 0
    information <- 0
    for (i in answered) {
      trace <- item_trace(models[i], parameters[[i]], theta)
      response <- pattern[i] + 1
      slope <- slope + trace$P1[1, response] / trace$P[1, response]
      j <- j + sum(trace$P1 * trace$P2 / trace$P)
      information <- information + sum(trace$P1^2 / trace$P)
    }
    slope + j / (2 * information)
  })
  grid <- seq(RANGE[1], RANGE[2], by = 0.25)
  values <- estimating(grid)
  crossings <- which(values[-length(values)] > 0 & values[-1] <= 0)
  roots <- sapply(crossings, function(j) uniroot(estimating, grid[j + 0:1], tol = 1e-13)$root)
  candidates <- c(if (values[1] <= 0) RANGE[1], roots, if (values[length(values)] > 0) RANGE[2])
  heights <- sapply(candidates, function(t) integrate(estimating, RANGE[1], t, rel.tol = 1e-12)$value)
  list(theta = candidates[which.max(heights)], roots = length(roots))
}

# Scores response patterns with mirt's fscores and base R. ML estimates that mirt finds infinite are the end of RANGE
# on that side. Standard errors of ML and WLE are 1 / √I and of MAP 1 / √(I + 1), from the answered items'
# information; EAP's is the posterior SD.
abilities <- function(fit, models, patterns, items = psychometrics_items(fit, models)) {
  parameters <- lapply(items, `[[`, "parameters")
  # Column 1 holds the estimates and column 2 their standard errors, whatever the factor's name.
  scores <- function(method, ...) fscores(fit, method = method, response.pattern = patterns, quadpts = POINTS, ...)
  eap <- scores("EAP")
  map <- scores("MAP")
  ml <- suppressWarnings(scores("ML"))
  rows <- seq_len(nrow(patterns))
  ml_theta <- pmin(pmax(ml[, 1], RANGE[1]), RANGE[2])
  wle <- lapply(rows, function(r) warm(models, parameters, patterns[r, ]))
  wle_theta <- sapply(wle, `[[`, "theta")
  if (all(models %in% LOGISTIC)) {
    # catR finds one root with uniroot's default tolerance, so it checks only patterns with a single root.
    single <- sapply(wle, `[[`, "roots") == 1
    pars <- parameter_matrix(items)
    catr <- sapply(rows[single], function(r) thetaEst(pars, patterns[r, ], method = "WL", range = RANGE))
    stopifnot(max(abs(catr - wle_theta[single])) < 1e-3)
  }
  if (all(models %in% WARM_MODELS)) {
    mirt_wle <- scores("WLE")[, 1]
    stopifnot(max(abs(mirt_wle - wle_theta)) < 1e-4)
  }
  estimate <- function(theta, se) lapply(rows, function(r) list(theta = theta[r], standardError = se[r]))
  information <- function(theta) sapply(rows, function(r) answered_information(fit, patterns[r, ], theta[r]))
  list(
    models = I(models),
    parameters = parameters,
    patterns = patterns,
    posteriorMean = estimate(eap[, 1], eap[, 2]),
    posteriorMode = estimate(map[, 1], 1 / sqrt(information(map[, 1]) + 1)),
    maximumLikelihood = estimate(ml_theta, 1 / sqrt(information(ml_theta))),
    weightedLikelihood = estimate(wle_theta, 1 / sqrt(information(wle_theta)))
  )
}

information <- function(fit, models, items = psychometrics_items(fit, models)) {
  list(
    models = I(models),
    parameters = lapply(items, `[[`, "parameters"),
    theta = INFORMATION_THETA,
    items = lapply(seq_along(models), function(i) iteminfo(extract.item(fit, i), INFORMATION_THETA)),
    test = testinfo(fit, INFORMATION_THETA)
  )
}

# --- LSAT7 ---------------------------------------------------------------------------------------------------------

lsat7 <- expand.table(LSAT7)
patterns <- rbind(
  response_patterns(5),
  c(1, NA, 0, NA, 1),
  c(NA, 0, 1, 1, NA),
  c(NA, NA, NA, NA, 1),
  c(0, NA, NA, NA, NA)
)
models_2pl <- rep("2pl", 5)

two_pl <- fit_mirt(lsat7, 1, "2PL", "Richardson")
oakes <- fit_mirt(lsat7, 1, "2PL", "Oakes")
standard_errors <- function(fit) unlist(lapply(psychometrics_items(fit, models_2pl), `[[`, "standardErrors"))
stopifnot(max(abs(standard_errors(oakes) - standard_errors(two_pl))) < 1e-5)
# The base R likelihood and its gradient reproduce mirt's.
two_pl_pars <- parameter_matrix(psychometrics_items(two_pl, models_2pl))
stopifnot(abs(marginal(lsat7, two_pl_pars)$logLikelihood - logLik(two_pl)) < 1e-8)
stopifnot(max(abs(marginal(lsat7, two_pl_pars)$gradient[, 1:2])) < 1e-4)

gauss_hermite <- ltm(lsat7 ~ z1, control = list(GHk = 21, iter.em = 1000, method = "BFGS"))
# summary() gives the IRT parameters with standard errors by the delta method.
ltm_coefficients <- summary(gauss_hermite)$coefficients
ltm_items <- lapply(seq_len(5), function(i) {
  b <- ltm_coefficients[i, ]
  a <- ltm_coefficients[i + 5, ]
  list(parameters = list(a = a[["value"]], b = b[["value"]], c = 0, d = 1),
       standardErrors = list(a = a[["std.err"]], b = b[["std.err"]]))
})

anchors <- list(Item.1 = list(a = 1, b = -2, c = 0, d = 1), Item.2 = list(a = 1.2, b = -0.7, c = 0, d = 1))
values <- mirt(lsat7, 1, itemtype = "2PL", pars = "values")
for (id in names(anchors)) {
  rows <- values$item == id
  values$value[rows & values$name == "a1"] <- anchors[[id]]$a
  values$value[rows & values$name == "d"] <- -anchors[[id]]$a * anchors[[id]]$b
  values$est[rows] <- FALSE
}
anchored <- fit_mirt(lsat7, 1, "2PL", "Richardson", pars = values)

beta_prior <- list(type = "beta", alpha = 5, beta = 17)
three_pl_start <- mirt(lsat7, mirt.model("F = 1-5\nPRIOR = (1-5, g, expbeta, 5, 17)"), itemtype = "3PL",
                       TOL = TOL, quadpts = POINTS, verbose = FALSE)
models_3pl <- rep("3pl", 5)
three_pl <- posterior_mode(
  lsat7,
  parameter_matrix(psychometrics_items(three_pl_start, models_3pl))[, 1:3],
  function(c) dbeta(c, beta_prior$alpha, beta_prior$beta, log = TRUE),
  function(c) (beta_prior$alpha - 1) / c - (beta_prior$beta - 1) / (1 - c)
)
# mirt's EM with an expbeta prior stops elsewhere: its prior gradient omits the logit's chain rule (mirt 1.48).
message("3PL log posterior: base R ", three_pl$logPosterior, ", mirt ",
        logLik(three_pl_start) + extract.mirt(three_pl_start, "logPrior"))

write_fixture("irt-lsat7", list(
  source = "mirt::LSAT7, expanded: 1000 persons, 5 dichotomous items, complete.",
  itemIds = colnames(lsat7),
  responses = unname(as.matrix(lsat7)),
  calibrations = list(
    c(list(name = "2PL, mirt", quadrature = list(method = "rectangular", points = POINTS)),
      mirt_calibration(two_pl, models_2pl)),
    list(name = "2PL, ltm with 21 Gauss-Hermite points", quadrature = list(method = "gauss-hermite", points = 21),
         models = I(models_2pl), items = ltm_items, logLikelihood = as.numeric(logLik(gauss_hermite))),
    c(list(name = "2PL with items 1 and 2 fixed, mirt", quadrature = list(method = "rectangular", points = POINTS),
           fixed = anchors),
      mirt_calibration(anchored, models_2pl)),
    list(name = "3PL with a Beta(5, 17) prior on c, base R posterior mode",
         quadrature = list(method = "rectangular", points = POINTS), priors = list(c = beta_prior),
         models = I(models_3pl), items = three_pl$items, logLikelihood = three_pl$logLikelihood)
  ),
  abilities = list(abilities(two_pl, models_2pl, patterns), abilities(three_pl_start, models_3pl, patterns)),
  information = list(information(two_pl, models_2pl), information(three_pl_start, models_3pl))
))

# --- Simulated 1PL with missing responses --------------------------------------------------------------------------

set.seed(20261029)
persons <- 800
item_count <- 12
a <- 1.3
b <- seq(-2, 2, length.out = item_count)
theta <- rnorm(persons)
simulated <- matrix(rbinom(persons * item_count, 1, plogis(a * outer(theta, b, "-"))), persons, item_count)
simulated[matrix(runif(persons * item_count) < 0.1, persons, item_count)] <- NA
colnames(simulated) <- sprintf("q%02d", seq_len(item_count))
equal_slopes <- mirt.model(sprintf("F = 1-%d\nCONSTRAIN = (1-%d, a1)", item_count, item_count))
simulated_1pl <- simulated
equal_slopes_1pl <- equal_slopes
models_1pl <- rep("1pl", item_count)
one_pl <- fit_mirt(as.data.frame(simulated), equal_slopes, "2PL", "Richardson")
rasch <- mirt(as.data.frame(simulated), 1, itemtype = "Rasch", TOL = TOL, quadpts = POINTS, verbose = FALSE)
# The 1PL is the Rasch model with the latent variance freed. The log-likelihoods differ only by quadrature: mirt keeps
# its nodes on [-6, 6] when it frees the variance.
stopifnot(abs(logLik(rasch) - logLik(one_pl)) < 1e-3)

write_fixture("irt-simulated-1pl", list(
  source = sprintf("Simulated 1PL (set.seed(20261029)): %d persons, %d items, a = %g, about 10%% missing.",
                   persons, item_count, a),
  itemIds = colnames(simulated),
  responses = unname(simulated),
  generating = list(a = a, b = b),
  calibrations = list(
    c(list(name = "1PL, mirt 2PL with equal slopes", quadrature = list(method = "rectangular", points = POINTS)),
      mirt_calibration(one_pl, models_1pl))
  ),
  abilities = list(abilities(one_pl, models_1pl, simulated[1:40, ]))
))

# --- Simulated 4PL with priors -------------------------------------------------------------------------------------

set.seed(20261029)
persons <- 2000
item_count <- 12
a <- round(runif(item_count, 1.2, 2.2), 2)
b <- round(seq(-2, 2, length.out = item_count), 2)
c <- round(runif(item_count, 0.1, 0.25), 2)
d <- round(runif(item_count, 0.85, 0.97), 2)
theta <- rnorm(persons)
probability <- sapply(seq_len(item_count), function(i) c[i] + (d[i] - c[i]) * plogis(a[i] * (theta - b[i])))
simulated <- matrix(rbinom(persons * item_count, 1, probability), persons, item_count)
colnames(simulated) <- sprintf("q%02d", seq_len(item_count))
# Normal priors on logit(g) and logit(u), which mirt differentiates correctly; as priors on c and d their log density
# is -((logit(x) - mean) / sd)^2 / 2.
lower_prior <- list(type = "logit-normal", mean = qlogis(0.15), sd = 0.25)
upper_prior <- list(type = "logit-normal", mean = qlogis(0.9), sd = 0.25)
models_4pl <- rep("4pl", item_count)
four_pl <- fit_mirt(as.data.frame(simulated), mirt.model(sprintf(
  "F = 1-%d\nPRIOR = (1-%d, g, norm, %.17g, %.17g), (1-%d, u, norm, %.17g, %.17g)",
  item_count, item_count, lower_prior$mean, lower_prior$sd, item_count, upper_prior$mean, upper_prior$sd
)), "4PL", "Oakes")

write_fixture("irt-simulated-4pl", list(
  source = sprintf("Simulated 4PL (set.seed(20261029)): %d persons, %d items, complete.", persons, item_count),
  itemIds = colnames(simulated),
  responses = unname(simulated),
  generating = list(a = a, b = b, c = c, d = d),
  calibrations = list(
    c(list(name = "4PL with normal priors on logit(c) and logit(d), mirt",
           quadrature = list(method = "rectangular", points = POINTS),
           priors = list(c = lower_prior, d = upper_prior)),
      mirt_calibration(four_pl, models_4pl))
  ),
  abilities = list(abilities(four_pl, models_4pl, simulated[1:40, ])),
  information = list(information(four_pl, models_4pl))
))

# --- Science: polytomous models ------------------------------------------------------------------------------------

science <- mirt::Science - 1 # categories 0 to 3
science_patterns <- rbind(
  response_patterns(4, 4),
  c(NA, 2, 3, NA),
  c(0, NA, NA, NA),
  c(NA, NA, 3, 3),
  c(1, NA, 2, 0)
)
science_models <- function(model) rep(model, ncol(science))

# The RSM on measurekit's scale: mirt's rsm with the slopes freed and equal, and the latent variance fixed at 1.
rsm_values <- mirt(science, 1, itemtype = "rsm", pars = "values")
rsm_values$est[rsm_values$name == "a1"] <- TRUE
rsm_values$est[rsm_values$name == "COV_11"] <- FALSE
rsm_values$value[rsm_values$name == "COV_11"] <- 1
polytomous_fits <- list(
  grm = fit_mirt(science, 1, "graded", "Richardson"),
  gpcm = fit_mirt(science, 1, "gpcm", "Richardson"),
  pcm = fit_mirt(science, mirt.model("F = 1-4\nCONSTRAIN = (1-4, a1)"), "gpcm", "Richardson"),
  rsm = fit_mirt(science, 1, "rsm", "Richardson", pars = rsm_values,
                 constrain = list(rsm_values$parnum[rsm_values$name == "a1"])),
  nrm = fit_mirt(science, 1, "nominal", "Richardson")
)
# The PCM and the RSM with a freed latent variance instead of a common slope fit the same up to quadrature.
stopifnot(abs(logLik(mirt(science, 1, itemtype = "Rasch", TOL = TOL, verbose = FALSE)) -
  logLik(polytomous_fits$pcm)) < 1e-3)
stopifnot(abs(logLik(mirt(science, 1, itemtype = "rsm", TOL = TOL, verbose = FALSE)) -
  logLik(polytomous_fits$rsm)) < 1e-3)
# measurekit's parameters from mirt's values reproduce coef(IRTpars = TRUE) where mirt has the IRT form.
for (model in c("grm", "gpcm", "pcm", "nrm")) {
  fit <- polytomous_fits[[model]]
  mirt_irt <- coef(fit, IRTpars = TRUE, simplify = TRUE)$items
  ours <- t(sapply(psychometrics_items(fit, science_models(model)), function(item) unlist(item$parameters)))
  stopifnot(max(abs(mirt_irt - ours)) < 1e-12)
}
# mirt's GPCM IRTpars standard error of b1 differs from the delta method on mirt's own covariance matrix.
message("Comfort GPCM SE of b1: coef(IRTpars = TRUE) ",
        coef(polytomous_fits$gpcm, IRTpars = TRUE, printSE = TRUE)$Comfort["SE", "b1"], ", delta method ",
        psychometrics_items(polytomous_fits$gpcm, science_models("gpcm"))[[1]]$standardErrors$b1)

labels <- c(grm = "GRM", gpcm = "GPCM", pcm = "PCM, mirt GPCM with equal slopes",
           rsm = "RSM, mirt rsm with equal free slopes and unit variance", nrm = "NRM")
write_fixture("irt-science", list(
  source = "mirt::Science, recoded 0 to 3: 392 persons, 4 items with 4 categories, complete.",
  itemIds = colnames(science),
  responses = unname(as.matrix(science)),
  calibrations = lapply(names(polytomous_fits), function(model) {
    c(list(name = paste0(labels[[model]], ", mirt"), quadrature = list(method = "rectangular", points = POINTS)),
      mirt_calibration(polytomous_fits[[model]], science_models(model)))
  }),
  abilities = lapply(names(polytomous_fits), function(model) {
    abilities(polytomous_fits[[model]], science_models(model), science_patterns)
  }),
  information = lapply(names(polytomous_fits), function(model) {
    information(polytomous_fits[[model]], science_models(model))
  })
))

# --- Simulated mixed format: 3PL and GPCM with anchors -------------------------------------------------------------

set.seed(20261029)
persons <- 3000
mc <- 9
a_mc <- round(runif(mc, 1, 2), 2)
b_mc <- round(seq(-1.5, 1.5, length.out = mc), 2)
c_mc <- round(runif(mc, 0.12, 0.25), 2)
a_cr <- c(1.1, 0.8, 1.4)
b_cr <- list(c(-1, 0.2, 1.1), c(-0.4, 0.7), c(-1.4, -0.3, 0.9))
theta <- rnorm(persons)
multiple_choice <- sapply(seq_len(mc), function(i) {
  rbinom(persons, 1, c_mc[i] + (1 - c_mc[i]) * plogis(a_mc[i] * (theta - b_mc[i])))
})
constructed <- sapply(seq_along(a_cr), function(i) {
  z <- t(apply(cbind(0, a_cr[i] * outer(theta, b_cr[[i]], "-")), 1, cumsum))
  p <- exp(z) / rowSums(exp(z))
  apply(p, 1, function(row) sample.int(length(row), 1, prob = row) - 1)
})
mixed <- as.data.frame(cbind(multiple_choice, constructed))
colnames(mixed) <- sprintf("q%02d", seq_len(ncol(mixed)))
mixed_models <- c(rep("3pl", mc), rep("gpcm", length(a_cr)))
mixed_prior <- list(type = "logit-normal", mean = qlogis(0.2), sd = 0.5)
mixed_model <- mirt.model(sprintf("F = 1-%d\nPRIOR = (1-%d, g, norm, %.17g, %.17g)", ncol(mixed), mc,
                                  mixed_prior$mean, mixed_prior$sd))
itemtypes <- c(rep("3PL", mc), rep("gpcm", length(a_cr)))
mixed_anchors <- list(q01 = list(a = a_mc[1], b = b_mc[1], c = c_mc[1], d = 1),
                      q10 = list(a = a_cr[1], b = I(b_cr[[1]])))
values <- mirt(mixed, mixed_model, itemtype = itemtypes, pars = "values")
set_value <- function(values, item, name, value) {
  values$value[values$item == item & values$name == name] <- value
  values
}
values <- set_value(values, "q01", "a1", a_mc[1])
values <- set_value(values, "q01", "d", -a_mc[1] * b_mc[1])
values <- set_value(values, "q01", "g", c_mc[1])
intercepts <- -a_cr[1] * cumsum(b_cr[[1]])
values <- set_value(values, "q10", "a1", a_cr[1])
for (k in seq_along(intercepts)) values <- set_value(values, "q10", paste0("d", k), intercepts[k])
values$est[values$item %in% names(mixed_anchors)] <- FALSE
mixed_fit <- fit_mirt(mixed, mixed_model, itemtypes, "Oakes", pars = values)

write_fixture("irt-simulated-mixed", list(
  source = sprintf("Simulated mixed format (set.seed(20261029)): %d persons, %d 3PL and %d GPCM items, complete.",
                   persons, mc, length(a_cr)),
  itemIds = colnames(mixed),
  responses = unname(as.matrix(mixed)),
  generating = list(a = c(a_mc, a_cr), b = c(as.list(b_mc), lapply(b_cr, I)), c = c_mc),
  calibrations = list(
    c(list(name = "3PL and GPCM with q01 and q10 fixed and a normal prior on logit(c), mirt",
           quadrature = list(method = "rectangular", points = POINTS), fixed = mixed_anchors,
           priors = list(c = mixed_prior)),
      mirt_calibration(mixed_fit, mixed_models))
  ),
  abilities = list(abilities(mixed_fit, mixed_models, as.matrix(mixed[1:40, ]))),
  information = list(information(mixed_fit, mixed_models))
))

# --- Joint maximum likelihood --------------------------------------------------------------------------------------
# TAM's JML (tam.jml): the Rasch model for dichotomous items and the partial credit model for the others; a total
# score of 0 or of the maximum moves 0.3 points inward; each iteration solves the persons' score equations, centers
# theta on the persons and updates the items; the thresholds are then multiplied by (I - 1) / I. The adjusted scores
# make the person and item equations inconsistent by their total, so the estimates are the fixed point of these
# steps: the item equations hold, and the person equations up to the centering shift. Base R iterates the same steps
# from TAM's estimates to a change below 1e-12 and checks TAM against that solution; it also gives the standard errors.

pcm_probabilities <- function(thresholds, theta) {
  z <- cumsum(c(0, theta - thresholds))
  p <- exp(z - max(z))
  p / sum(p)
}

# The threshold information matrix Σ (P(X ≥ max(j, m)) - P(X ≥ j) P(X ≥ m)) of an item over the given thetas, and
# the expected numbers of responses in categories j and above.
threshold_information <- function(thresholds, theta) {
  tails <- matrix(t(sapply(theta, function(t) rev(cumsum(rev(pcm_probabilities(thresholds, t))))[-1])),
                  length(theta))
  expected <- colSums(tails)
  j <- seq_along(thresholds)
  list(information = outer(j, j, function(x, y) expected[pmax(x, y)]) - crossprod(tails), expected = expected)
}

jml_solution <- function(resp, thresholds, theta, adjustment = 0.3) {
  categories <- apply(resp, 2, max, na.rm = TRUE) + 1
  answered <- !is.na(resp)
  score <- rowSums(resp, na.rm = TRUE)
  maximum <- drop(answered %*% (categories - 1))
  target <- ifelse(score == 0, adjustment, ifelse(score == maximum, maximum - adjustment, score))
  for (iteration in 1:10000) {
    old <- c(theta, unlist(thresholds))
    theta <- sapply(seq_len(nrow(resp)), function(p) {
      t <- theta[p]
      for (step in 1:100) {
        moments <- sapply(which(answered[p, ]), function(i) {
          probabilities <- pcm_probabilities(thresholds[[i]], t)
          k <- seq_along(probabilities) - 1
          c(sum(k * probabilities), sum(k^2 * probabilities) - sum(k * probabilities)^2)
        })
        change <- (target[p] - sum(moments[1, ])) / sum(moments[2, ])
        t <- t + change
        if (abs(change) < 1e-14) break
      }
      t
    })
    theta <- theta - mean(theta)
    thresholds <- lapply(seq_along(thresholds), function(i) {
      x <- resp[answered[, i], i]
      observed <- sapply(seq_len(categories[i] - 1), function(j) sum(x >= j))
      d <- thresholds[[i]]
      for (step in 1:100) {
        moments <- threshold_information(d, theta[answered[, i]])
        change <- solve(moments$information, moments$expected - observed)
        d <- d + change
        if (max(abs(change)) < 1e-14) break
      }
      d
    })
    if (max(abs(c(theta, unlist(thresholds)) - old)) < 1e-12) break
  }
  stopifnot(iteration < 10000)
  log_likelihood <- sum(sapply(seq_len(nrow(resp)), function(p) {
    sum(sapply(which(answered[p, ]), function(i) log(pcm_probabilities(thresholds[[i]], theta[p])[resp[p, i] + 1])))
  }))
  errors <- lapply(seq_along(thresholds), function(i) {
    sqrt(diag(solve(threshold_information(thresholds[[i]], theta[answered[, i]])$information)))
  })
  person_errors <- sapply(seq_len(nrow(resp)), function(p) {
    1 / sqrt(sum(sapply(which(answered[p, ]), function(i) {
      probabilities <- pcm_probabilities(thresholds[[i]], theta[p])
      k <- seq_along(probabilities) - 1
      sum(k^2 * probabilities) - sum(k * probabilities)^2
    })))
  })
  list(thresholds = thresholds, errors = errors, theta = theta, personErrors = person_errors,
       logLikelihood = log_likelihood, target = target)
}

set.seed(20261029)
persons <- 500
dichotomous <- 10
steps <- list(c(-1, 0.8), c(-0.3, 0.6), c(-1.5, 0, 1.2), c(-0.8, 0.2, 1.6))
theta <- rnorm(persons, 0, 1.3)
b <- seq(-2, 2, length.out = dichotomous)
rasch_responses <- sapply(b, function(difficulty) rbinom(persons, 1, plogis(theta - difficulty)))
pcm_responses <- sapply(steps, function(thresholds) {
  z <- t(apply(cbind(0, outer(theta, thresholds, "-")), 1, cumsum))
  p <- exp(z) / rowSums(exp(z))
  apply(p, 1, function(row) sample.int(length(row), 1, prob = row) - 1)
})
jml_data <- cbind(rasch_responses, pcm_responses)
jml_data[matrix(runif(length(jml_data)) < 0.05, nrow(jml_data))] <- NA
colnames(jml_data) <- sprintf("q%02d", seq_len(ncol(jml_data)))
item_count <- ncol(jml_data)
correction <- (item_count - 1) / item_count

tam <- NULL
invisible(capture.output(
  tam <- tam.jml(jml_data, control = list(conv = 1e-10, convM = 1e-12, maxiter = 10000, Msteps = 50), verbose = FALSE)
))
categories <- apply(jml_data, 2, max, na.rm = TRUE) + 1
ends <- cumsum(categories - 1)
by_item <- function(x) lapply(seq_len(item_count), function(i) x[(ends[i] - categories[i] + 2):ends[i]])
solution <- jml_solution(jml_data, by_item(tam$xsi / correction), tam$theta)
# The solution uses TAM's adjusted total scores, and TAM's estimates and log-likelihood are close to it.
stopifnot(max(abs(solution$target - tam$PersonScores)) < 1e-12)
tam_difference <- max(abs(unlist(solution$thresholds) * correction - tam$xsi), abs(solution$theta - tam$theta))
message("JML: largest difference between TAM and the solution of its steps ", tam_difference)
stopifnot(tam_difference < 1e-7, abs(solution$logLikelihood + tam$deviance / 2) < 1e-6)
# For dichotomous items, the standard error is TAM's errorP, one over the root of the item's information, times the
# correction, which TAM does not apply to it. For polytomous items TAM's errorP uses only the diagonal of the
# information matrix.
tam_errors <- by_item(tam$errorP)
stopifnot(max(abs(unlist(solution$errors[categories == 2]) / unlist(tam_errors[categories == 2]) - 1)) < 1e-6)
# eRm estimates the same models by conditional maximum likelihood, which needs no correction; with missing responses
# its polytomous estimates are unreliable, so it is compared on the complete cases. Both sets of thresholds are
# centered, as the two methods fix the origin differently.
complete <- jml_data[complete.cases(jml_data), ]
invisible(capture.output(complete_tam <- tam.jml(complete, verbose = FALSE)))
cml <- thresholds(PCM(complete))$threshtable[[1]][, -1, drop = FALSE]
cml_thresholds <- unlist(lapply(seq_len(item_count), function(i) cml[i, seq_len(categories[i] - 1)]))
centered <- function(x) x - mean(x)
cml_difference <- max(abs(centered(cml_thresholds) - centered(complete_tam$xsi)))
message("JML: largest difference between the corrected thresholds and eRm's CML thresholds on the complete cases ",
        cml_difference)
stopifnot(cml_difference < 0.2)

write_fixture("irt-simulated-pcm", list(
  source = sprintf("Simulated Rasch and PCM (set.seed(20261029)): %d persons, %d dichotomous and %d polytomous items, about 5%% missing.",
                   persons, dichotomous, length(steps)),
  itemIds = colnames(jml_data),
  responses = unname(jml_data),
  generating = list(b = b, steps = lapply(steps, I)),
  jointMaximumLikelihood = list(
    name = "TAM tam.jml, iterated by base R to a change below 1e-12",
    items = lapply(seq_len(item_count), function(i) {
      b <- solution$thresholds[[i]] * correction
      se <- solution$errors[[i]] * correction
      if (categories[i] == 2) {
        list(model = "1pl", parameters = list(a = 1, b = b, c = 0, d = 1), standardErrors = list(b = se))
      } else {
        list(model = "pcm", parameters = list(a = 1, b = I(b)),
             standardErrors = as.list(setNames(se, paste0("b", seq_along(se)))))
      }
    }),
    persons = lapply(seq_len(persons), function(p) list(theta = solution$theta[p],
                                                        standardError = solution$personErrors[p])),
    logLikelihood = solution$logLikelihood
  )
))

# --- Distributions of theta: free normal distributions, groups, online calibration, histograms -------------------
# Base R maximizes the marginal log-likelihood of 2PL items over groups of persons, each group's theta on mirt's 61
# nodes with weights proportional to the normal density or a histogram: Newton's method on the gradient by Fisher's
# identity, with the Hessian by central differences of that gradient, which also gives the standard errors. mirt
# (optimizer = "NR", SE.type = "Oakes") cross-checks: its M-step maximizes Σ n_q ln φ(θ_q) without normalizing the
# weights over the nodes, so its fixed point is not the maximum of the log-likelihood it reports.

NODES <- seq(-6, 6, length.out = POINTS)

normal_weights <- function(mean, sd) {
  w <- dnorm(NODES, mean, sd)
  w / sum(w)
}

# The E-step: each person's posterior over the nodes, the log-likelihood, and the traces P(correct) at the nodes.
posteriors <- function(x, a, d, log_weights) {
  traces <- plogis(outer(NODES, a) + matrix(d, POINTS, length(a), byrow = TRUE))
  log_joint <- log_weights
  for (i in seq_along(a)) {
    answered <- !is.na(x[, i])
    log_joint[answered, ] <- log_joint[answered, ] + outer(x[answered, i], log(traces[, i])) +
      outer(1 - x[answered, i], log(1 - traces[, i]))
  }
  largest <- apply(log_joint, 1, max)
  posterior <- exp(log_joint - largest)
  totals <- rowSums(posterior)
  list(posterior = posterior / totals, logLikelihood = sum(largest + log(totals)), traces = traces)
}

# A calibration of 2PL items: `fixed` items keep (a, d); each group has "standard-normal", "normal" or a histogram.
# Parameters: the free items' a, then their d, then each normal group's mean and sd.
mml_problem <- function(x, group, fixed, a, d, distributions, histograms = list()) {
  free <- setdiff(seq_len(ncol(x)), fixed)
  groups <- unique(group)
  normal <- groups[distributions[groups] == "normal"]
  unpack <- function(p) {
    a[free] <- p[seq_along(free)]
    d[free] <- p[length(free) + seq_along(free)]
    latent <- matrix(p[-seq_len(2 * length(free))], ncol = 2, byrow = TRUE)
    list(a = a, d = d, latent = latent)
  }
  log_weights <- function(u) {
    by_group <- sapply(groups, function(g) {
      k <- match(g, normal)
      if (!is.na(k)) return(log(normal_weights(u$latent[k, 1], u$latent[k, 2])))
      if (distributions[[g]] == "empirical-histogram") return(log(histograms[[g]]))
      log(normal_weights(0, 1))
    })
    t(by_group)[match(group, groups), , drop = FALSE]
  }
  gradient <- function(p) {
    u <- unpack(p)
    e <- posteriors(x, u$a, u$d, log_weights(u))
    item_gradient <- sapply(free, function(i) {
      answered <- !is.na(x[, i])
      residual <- colSums(e$posterior[answered, , drop = FALSE] * (x[answered, i] - matrix(e$traces[, i], sum(answered), POINTS, byrow = TRUE)))
      c(sum(residual * NODES), sum(residual))
    })
    latent_gradient <- unlist(lapply(seq_along(normal), function(k) {
      counts <- colSums(e$posterior[group == normal[k], , drop = FALSE])
      mean <- u$latent[k, 1]
      sd <- u$latent[k, 2]
      excess <- counts - sum(counts) * normal_weights(mean, sd)
      c(sum(excess * (NODES - mean)) / sd^2, sum(excess * (NODES - mean)^2) / sd^3)
    }))
    c(item_gradient[1, ], item_gradient[2, ], latent_gradient)
  }
  list(unpack = unpack, gradient = gradient, logLikelihood = function(p) {
    u <- unpack(p)
    posteriors(x, u$a, u$d, log_weights(u))$logLikelihood
  }, free = free, normal = normal)
}

hessian <- function(gradient, p) {
  h <- sapply(seq_along(p), function(j) {
    step <- 1e-5 * max(1, abs(p[j]))
    up <- p
    down <- p
    up[j] <- up[j] + step
    down[j] <- down[j] - step
    (gradient(up) - gradient(down)) / (2 * step)
  })
  (h + t(h)) / 2
}

# Newton's method from `start` to a gradient below 1e-7, then the standard errors from the same Hessian, with those of
# b = -d / a by the delta method.
maximize <- function(problem, start) {
  p <- start
  for (step in 1:50) {
    g <- problem$gradient(p)
    if (max(abs(g)) < 1e-7) break
    p <- p - solve(hessian(problem$gradient, p), g)
  }
  stopifnot(max(abs(problem$gradient(p))) < 1e-7)
  covariance <- solve(-hessian(problem$gradient, p))
  k <- length(problem$free)
  u <- problem$unpack(p)
  items <- lapply(seq_along(u$a), function(i) {
    j <- match(i, problem$free)
    parameters <- list(a = u$a[i], b = -u$d[i] / u$a[i], c = 0, d = 1)
    if (is.na(j)) return(list(parameters = parameters))
    jacobian <- c(u$d[i] / u$a[i]^2, -1 / u$a[i]) # db / d(a, d)
    block <- covariance[c(j, k + j), c(j, k + j)]
    list(parameters = parameters, standardErrors = list(a = sqrt(block[1, 1]), b = sqrt(drop(t(jacobian) %*% block %*% jacobian))))
  })
  latent_errors <- sqrt(diag(covariance))[-seq_len(2 * k)]
  groups <- lapply(seq_along(problem$normal), function(k2) {
    list(group = problem$normal[k2], mean = u$latent[k2, 1], sd = u$latent[k2, 2],
         standardErrors = list(mean = latent_errors[2 * k2 - 1], sd = latent_errors[2 * k2]))
  })
  list(items = items, groups = groups, logLikelihood = problem$logLikelihood(p))
}

set_anchors <- function(values, anchors) {
  for (id in names(anchors)) {
    rows <- values$item == id
    values$value[rows & values$name == "a1"] <- anchors[[id]]$a
    values$value[rows & values$name == "d"] <- -anchors[[id]]$a * anchors[[id]]$b
    values$est[rows] <- FALSE
  }
  values
}

# The start of Newton's method from a mirt fit: the free items' a and d, then each normal group's mean and sd.
mirt_start <- function(fits, free, normal_groups) {
  coefficients <- coef(fits[[1]], simplify = TRUE)$items
  latent <- unlist(lapply(normal_groups, function(g) {
    gp <- coef(fits[[g]], simplify = TRUE)
    c(gp$means[1], sqrt(gp$cov[1]))
  }))
  c(coefficients[free, "a1"], coefficients[free, "d"], latent)
}

latent_case <- function(name, x, group, anchors, distributions, fits) {
  fixed <- match(names(anchors), colnames(x))
  a <- rep(1, ncol(x))
  d <- rep(0, ncol(x))
  a[fixed] <- vapply(anchors, `[[`, numeric(1), "a")
  d[fixed] <- -a[fixed] * vapply(anchors, `[[`, numeric(1), "b")
  problem <- mml_problem(x, group, fixed, a, d, distributions)
  reference <- maximize(problem, mirt_start(fits, problem$free, problem$normal))
  # mirt's estimates, a cross-check: the same items and means within 5e-3.
  estimates <- function(r) c(sapply(r$items, function(i) unlist(i$parameters[c("a", "b")])), sapply(r$groups, `[[`, "mean"))
  mirt_items <- coef(fits[[1]], IRTpars = TRUE, simplify = TRUE)$items
  mirt_means <- sapply(problem$normal, function(g) coef(fits[[g]], simplify = TRUE)$means[1])
  difference <- max(abs(estimates(reference) - c(t(mirt_items[, c("a", "b")]), mirt_means)))
  message(name, ": largest difference between the maximum and mirt's estimates ", difference)
  stopifnot(difference < 5e-3)
  free_count <- 2 * length(problem$free) + 2 * length(problem$normal)
  list(name = name, itemIds = colnames(x), responses = unname(x), groups = if (length(unique(group)) > 1) group,
       fixed = if (length(anchors) > 0) anchors else setNames(list(), character(0)), distributions = as.list(distributions), items = reference$items, groupDistributions = reference$groups,
       logLikelihood = reference$logLikelihood,
       aic = -2 * reference$logLikelihood + 2 * free_count,
       bic = -2 * reference$logLikelihood + free_count * log(sum(rowSums(!is.na(x)) > 0)))
}

fit_nr <- function(estimate, ...) {
  fit <- estimate(..., TOL = TOL, quadpts = POINTS, optimizer = "NR", verbose = FALSE, technical = list(NCYCLES = 1e5))
  stopifnot(extract.mirt(fit, "converged"))
  fit
}

# Anchors fix the scale: 4 of 12 2PL items at their generating values, theta ~ N(0.5, 1.2^2), 5% missing.
set.seed(20261029)
persons <- 2000
item_count <- 12
a <- round(runif(item_count, 0.8, 2), 2)
b <- round(seq(-2, 2, length.out = item_count), 2)
theta <- rnorm(persons, 0.5, 1.2)
anchored_data <- sapply(seq_len(item_count), function(i) rbinom(persons, 1, plogis(a[i] * (theta - b[i]))))
anchored_data[matrix(runif(length(anchored_data)) < 0.05, persons)] <- NA
colnames(anchored_data) <- sprintf("q%02d", seq_len(item_count))
anchored_anchors <- setNames(lapply(1:4, function(i) list(a = a[i], b = b[i], c = 0, d = 1)), colnames(anchored_data)[1:4])
values <- set_anchors(mirt(anchored_data, 1, itemtype = "2PL", pars = "values"), anchored_anchors)
values$est[values$name %in% c("MEAN_1", "COV_11")] <- TRUE
anchored_fit <- fit_nr(mirt, anchored_data, 1, itemtype = "2PL", pars = values)
anchored <- latent_case("2PL with 4 fixed items and a normal distribution, base R maximum", anchored_data,
                        rep("all", persons), anchored_anchors, c(all = "normal"), list(all = anchored_fit))

# Two groups on one scale: 15 2PL items, the reference group N(0, 1), the focal group N(-0.5, 0.8^2), all items free.
set.seed(20261029)
item_count <- 15
a <- round(runif(item_count, 0.8, 2), 2)
b <- round(seq(-2, 1.5, length.out = item_count), 2)
group <- rep(c("reference", "focal"), each = 1000)
theta <- ifelse(group == "reference", rnorm(2000), rnorm(2000, -0.5, 0.8))
grouped_data <- sapply(seq_len(item_count), function(i) rbinom(2000, 1, plogis(a[i] * (theta - b[i]))))
colnames(grouped_data) <- sprintf("q%02d", seq_len(item_count))
grouped_fit <- fit_nr(multipleGroup, grouped_data, 1, group = factor(group, c("reference", "focal")), itemtype = "2PL",
                      invariance = c(colnames(grouped_data), "free_means", "free_var"))
grouped_fits <- list(reference = extract.group(grouped_fit, "reference"), focal = extract.group(grouped_fit, "focal"))
grouped <- latent_case("2PL in two groups, the focal group's normal distribution free, base R maximum", grouped_data,
                       group, list(), c(reference = "standard-normal", focal = "normal"), grouped_fits)

# Online calibration: adaptive tests of 15 items from 40 operational 2PL items, fixed, by maximum information at the
# EAP estimate, each with 2 of 6 pretest items at random; theta ~ N(0.3, 1.1^2).
set.seed(20261029)
persons <- 2000
operational <- 40
pretest <- 6
pool_a <- round(runif(operational + pretest, 0.8, 2), 2)
pool_b <- round(rnorm(operational + pretest, 0, 1.2), 2)
theta <- rnorm(persons, 0.3, 1.1)
online_data <- matrix(NA_real_, persons, operational + pretest)
log_posterior <- matrix(log(normal_weights(0, 1)), persons, POINTS, byrow = TRUE)
for (step in 1:15) {
  posterior <- exp(log_posterior - apply(log_posterior, 1, max))
  estimate <- drop(posterior %*% NODES) / rowSums(posterior)
  p <- plogis(outer(estimate, pool_a[1:operational]) - matrix(pool_a[1:operational] * pool_b[1:operational], persons, operational, byrow = TRUE))
  information <- p * (1 - p) * matrix(pool_a[1:operational]^2, persons, operational, byrow = TRUE)
  information[!is.na(online_data[, 1:operational])] <- -Inf
  chosen <- max.col(information, ties.method = "first")
  response <- rbinom(persons, 1, plogis(pool_a[chosen] * (theta - pool_b[chosen])))
  online_data[cbind(seq_len(persons), chosen)] <- response
  traces <- plogis(outer(pool_a[chosen], NODES) - pool_a[chosen] * pool_b[chosen])
  log_posterior <- log_posterior + response * log(traces) + (1 - response) * log(1 - traces)
}
for (person in seq_len(persons)) {
  seeded <- operational + sample.int(pretest, 2)
  online_data[person, seeded] <- rbinom(2, 1, plogis(pool_a[seeded] * (theta[person] - pool_b[seeded])))
}
colnames(online_data) <- c(sprintf("op%02d", seq_len(operational)), sprintf("pre%d", seq_len(pretest)))
# Operational items no test selected have no responses, which mirt does not take.
administered <- colSums(!is.na(online_data)) > 0
online_data <- online_data[, administered]
pool_a <- pool_a[administered]
pool_b <- pool_b[administered]
operational <- sum(administered) - pretest
online_anchors <- setNames(lapply(1:operational, function(i) list(a = pool_a[i], b = pool_b[i], c = 0, d = 1)),
                           colnames(online_data)[1:operational])
values <- set_anchors(mirt(online_data, 1, itemtype = "2PL", pars = "values"), online_anchors)
values$est[values$name %in% c("MEAN_1", "COV_11")] <- TRUE
online_fit <- fit_nr(mirt, online_data, 1, itemtype = "2PL", pars = values)
online <- latent_case(sprintf("Online calibration: %d fixed operational and %d pretest 2PL items, a normal distribution, base R maximum", operational, pretest),
                      online_data, rep("all", persons), online_anchors, c(all = "normal"), list(all = online_fit))

# A histogram on fixed items: base R's EM from the standard normal weights, each cycle's weights the expected counts
# over their sum, until no weight changes by more than 1e-4, measurekit's rule. The weights creep toward a few
# spikes, the maximum, so a histogram is where EM stops.
histogram_tolerance <- 1e-4
histogram_items <- online_anchors
weights <- normal_weights(0, 1)
for (cycle in 1:1e5) {
  e <- posteriors(online_data[, 1:operational], pool_a[1:operational], -pool_a[1:operational] * pool_b[1:operational],
                  matrix(log(weights), persons, POINTS, byrow = TRUE))
  updated <- colSums(e$posterior) / persons
  change <- max(abs(updated - weights))
  weights <- updated
  if (change <= histogram_tolerance) break
}
stopifnot(change <= histogram_tolerance)
histogram_logLikelihood <- posteriors(online_data[, 1:operational], pool_a[1:operational],
                                      -pool_a[1:operational] * pool_b[1:operational],
                                      matrix(log(weights), persons, POINTS, byrow = TRUE))$logLikelihood

write_fixture("irt-latent", list(
  source = "2PL simulations, set.seed(20261029): anchored items, two groups, adaptive tests with pretest items.",
  calibrations = list(anchored, grouped, online),
  histogram = list(
    name = sprintf("Empirical histogram on the %d fixed operational items of the adaptive tests, base R EM", operational),
    itemIds = colnames(online_data)[1:operational], responses = unname(online_data[, 1:operational]),
    fixed = histogram_items, tolerance = histogram_tolerance, iterations = cycle, weights = weights,
    logLikelihood = histogram_logLikelihood
  )
))

# --- Summed-score tables and reliability ---------------------------------------------------------------------------
# mirt's fscores(method = "EAPsum") with the standard normal prior; base R's Lord-Wingersky recursion on probtrace for
# another prior, with the nodes placed on it, measurekit's rule. mirt's marginal_rxx and empirical_rxx; base
# R's integrate for the marginal reliability under another prior.

summed_score_table <- function(fit, prior) {
  z <- seq(-6, 6, length.out = POINTS)
  nodes <- prior$mean + prior$sd * z
  weights <- dnorm(z) / sum(dnorm(z))
  likelihoods <- score_likelihoods(fit, nodes)
  probability <- drop(likelihoods %*% weights)
  theta <- drop(likelihoods %*% (weights * nodes)) / probability
  sd <- sqrt(sapply(seq_along(theta), function(s) sum(likelihoods[s, ] * weights * (nodes - theta[s])^2)) / probability)
  data.frame(score = seq_along(theta) - 1, theta = theta, standardError = sd, probability = probability)
}

scoring_case <- function(fit, models, persons) {
  table <- fscores(fit, method = "EAPsum", full.scores = FALSE, quadpts = POINTS, verbose = FALSE)
  standard <- summed_score_table(fit, list(mean = 0, sd = 1))
  stopifnot(max(abs(standard$theta - table$F1), abs(standard$standardError - table$SE_F1),
                abs(standard$probability - table$expected / persons)) < 1e-10)
  prior <- list(mean = 0.4, sd = 1.3)
  marginal <- function(mean, sd) {
    integrate(function(theta) {
      information <- testinfo(fit, theta)
      sd^2 * information / (sd^2 * information + 1) * dnorm(theta, mean, sd)
    }, -Inf, Inf, rel.tol = 1e-12)$value
  }
  stopifnot(abs(marginal(0, 1) - marginal_rxx(fit)) < 1e-6)
  list(
    models = I(models),
    parameters = lapply(psychometrics_items(fit, models), `[[`, "parameters"),
    tables = list(
      list(prior = list(mean = 0, sd = 1), estimates = data.frame(score = table$Sum.Scores, theta = table$F1,
                                                                   standardError = table$SE_F1,
                                                                   probability = table$expected / persons)),
      list(prior = prior, estimates = summed_score_table(fit, prior))
    ),
    marginalReliability = list(
      list(prior = list(mean = 0, sd = 1), value = marginal_rxx(fit)),
      list(prior = prior, value = marginal(prior$mean, prior$sd))
    )
  )
}

eap <- fscores(two_pl, full.scores.SE = TRUE, quadpts = POINTS)
write_fixture("irt-scoring", list(
  source = "mirt::LSAT7 under the 2PL and mirt::Science under the GPCM, the calibrations of irt-lsat7 and irt-science.",
  forms = list(scoring_case(two_pl, models_2pl, nrow(lsat7)),
               scoring_case(polytomous_fits$gpcm, science_models("gpcm"), nrow(science))),
  empiricalReliability = list(
    estimates = lapply(seq_len(nrow(eap)), function(r) list(theta = eap[r, 1], standardError = eap[r, 2])),
    value = unname(empirical_rxx(eap))
  )
))

# --- Cross-product standard errors ---------------------------------------------------------------------------------
# mirt's SE.type = "crossprod": the inverse of the sum over persons of the outer products of their score vectors.

cross_product <- function(name, data, model, itemtype, models) {
  fit <- fit_mirt(data, model, itemtype, "crossprod")
  c(list(name = name, quadrature = list(method = "rectangular", points = POINTS)), mirt_calibration(fit, models))
}

write_fixture("irt-cross-product", list(
  source = "mirt::LSAT7 under the 2PL; the simulated 1PL with missing responses; mirt::Science under the GPCM.",
  cases = list(
    c(list(itemIds = colnames(lsat7), responses = unname(as.matrix(lsat7))),
      cross_product("2PL, mirt crossprod", lsat7, 1, "2PL", models_2pl)),
    c(list(itemIds = colnames(simulated_1pl), responses = unname(simulated_1pl)),
      cross_product("1PL, mirt 2PL with equal slopes, crossprod", as.data.frame(simulated_1pl), equal_slopes_1pl, "2PL",
                    models_1pl)),
    c(list(itemIds = colnames(science), responses = unname(as.matrix(science))),
      cross_product("GPCM, mirt crossprod", science, 1, "gpcm", science_models("gpcm")))
  )
))

# --- Testlet response model ----------------------------------------------------------------------------------------
# The testlet model of Bradlow, Wainer and Wang (1999) is the bifactor model whose specific slopes equal the general
# ones (Li, Bolt and Fu, 2006): mirt's bfactor with each item's two slopes constrained equal and the testlets'
# variances freed, on mirt's grid of 21 nodes from -6 to 6 for theta and for each testlet effect.

TESTLET_POINTS <- 21

testlet_fit <- function(data, testlet) {
  values <- bfactor(data, testlet, pars = "values")
  specific <- grep("^COV_([0-9]+)\\1$", values$name, value = TRUE)
  values$est[values$item == "GROUP" & values$name %in% setdiff(specific, "COV_11")] <- TRUE
  slopes <- lapply(colnames(data), function(id) values$parnum[values$item == id & grepl("^a", values$name) & values$value != 0])
  fit <- bfactor(data, testlet, pars = values, constrain = slopes, TOL = 1e-8, quadpts = TESTLET_POINTS,
                 optimizer = "NR", SE = TRUE, SE.type = "Oakes", verbose = FALSE, technical = list(NCYCLES = 1e5))
  stopifnot(extract.mirt(fit, "converged"))
  fit
}

# The marginal log-likelihood of 2PL items in testlets by the bifactor dimension reduction (Gibbons and Hedeker, 1992):
# at each node of theta, each testlet's Σ_g w_g Π_j P_j(x_j | θ + γ_g), on the grid of θ + γ, the weights of γ ∝ φ(γ
# / σ). The parameters are a, d = -a b and ln σ.
testlet_log_likelihood <- function(par, x, testlet) {
  k <- ncol(x)
  testlets <- sort(unique(testlet))
  a <- par[1:k]
  d <- par[k + 1:k]
  sd <- exp(par[2 * k + seq_along(testlets)])
  nodes <- seq(-6, 6, length.out = TESTLET_POINTS)
  sums <- seq(-12, 12, length.out = 2 * TESTLET_POINTS - 1)
  weights <- dnorm(nodes) / sum(dnorm(nodes))
  log_posterior <- matrix(log(weights), nrow(x), TESTLET_POINTS, byrow = TRUE)
  for (t in seq_along(testlets)) {
    items <- which(testlet == testlets[t])
    z <- outer(sums, a[items]) + matrix(d[items], length(sums), length(items), byrow = TRUE)
    responses <- x[, items, drop = FALSE]
    answered <- !is.na(responses)
    # ln P and ln(1 - P) without rounding to 0 and -Inf in the tails of the grid.
    s <- ifelse(answered, responses, 0) %*% t(plogis(z, log.p = TRUE)) +
      ifelse(answered, 1 - responses, 0) %*% t(plogis(-z, log.p = TRUE))
    v <- dnorm(nodes / sd[t])
    v <- v / sum(v)
    largest <- apply(s, 1, max)
    e <- exp(s - largest)
    integral <- sapply(seq_len(TESTLET_POINTS), function(q) e[, q:(q + TESTLET_POINTS - 1), drop = FALSE] %*% v)
    log_posterior <- log_posterior + largest + log(integral)
  }
  largest <- apply(log_posterior, 1, max)
  sum(largest + log(rowSums(exp(log_posterior - largest))))
}

# BFGS on the negative log-likelihood `objective` with its gradient by central differences, then Newton steps on
# optimHess; the covariance of the parameters is the inverse of the same Hessian.
maximum_of <- function(objective, start) {
  gradient <- function(par) {
    sapply(seq_along(par), function(j) {
      step <- 1e-4 * max(1, abs(par[j]))
      up <- par
      down <- par
      up[j] <- up[j] + step
      down[j] <- down[j] - step
      (objective(up) - objective(down)) / (2 * step)
    })
  }
  par <- optim(start, objective, gradient, method = "BFGS", control = list(reltol = 1e-15, maxit = 5000))$par
  for (step in 1:3) par <- par - solve(optimHess(par, objective, gradient), gradient(par))
  covariance <- solve(optimHess(par, objective, gradient))
  list(par = par, covariance = covariance, logLikelihood = -objective(par), gradient = gradient(par))
}

testlet_maximum <- function(x, testlet, start) {
  maximum_of(function(par) -testlet_log_likelihood(par, x, testlet), start)
}

# The EAP of theta with each testlet's effect integrated out: 61 rectangular nodes z on the N(0, 1) prior for theta and
# σ z for each testlet, weights ∝ φ(z), measurekit's rule.
testlet_eap <- function(pattern, a, d, sd, testlet) {
  z <- seq(-6, 6, length.out = 61)
  w <- dnorm(z) / sum(dnorm(z))
  log_likelihood <- sapply(z, function(theta) {
    sum(sapply(seq_along(sd), function(t) {
      items <- which(testlet == t & !is.na(pattern))
      if (length(items) == 0) return(0)
      at <- sapply(sd[t] * z, function(gamma) {
        p <- plogis(a[items] * (theta + gamma) + d[items])
        sum(ifelse(pattern[items] == 1, log(p), log(1 - p)))
      })
      log(sum(w * exp(at)))
    }))
  })
  posterior <- w * exp(log_likelihood - max(log_likelihood))
  posterior <- posterior / sum(posterior)
  theta <- sum(posterior * z)
  list(theta = theta, standardError = sqrt(sum(posterior * (z - theta)^2)))
}

testlet_case <- function(name, data, testlet, labels, start) {
  k <- ncol(data)
  testlets <- seq_along(labels)
  maximum <- testlet_maximum(as.matrix(data), testlet, start)
  par <- maximum$par
  a <- par[1:k]
  d <- par[k + 1:k]
  sd <- exp(par[2 * k + testlets])
  errors <- sqrt(diag(maximum$covariance))
  items <- lapply(seq_len(k), function(j) {
    jacobian <- c(d[j] / a[j]^2, -1 / a[j])
    block <- maximum$covariance[c(j, k + j), c(j, k + j)]
    list(parameters = list(a = a[j], b = -d[j] / a[j], c = 0, d = 1),
         standardErrors = list(a = errors[j], b = sqrt(drop(t(jacobian) %*% block %*% jacobian))))
  })
  list(name = name, itemIds = colnames(data), responses = unname(as.matrix(data)),
       testlets = setNames(lapply(testlets, function(t) colnames(data)[testlet == t]), labels),
       items = items, sd = sd, sdStandardErrors = sd * errors[2 * k + testlets], logLikelihood = maximum$logLikelihood,
       abilities = lapply(1:10, function(r) testlet_eap(as.numeric(data[r, ]), a, d, sd, testlet)))
}

# mirt's estimates as a start: a, d and ln σ.
mirt_testlet_start <- function(fit, k) {
  coefficients <- coef(fit, simplify = TRUE)
  variances <- diag(coefficients$cov)[-1]
  c(coefficients$items[, "a1"], coefficients$items[, "d"], log(sqrt(variances)))
}

set.seed(20261029)
persons <- 1000
testlet_of <- rep(1:3, each = 4)
a <- round(runif(12, 0.8, 1.8), 2)
b <- round(rnorm(12), 2)
effect_sd <- c(0.5, 0.8, 1)
theta <- rnorm(persons)
effects <- sapply(effect_sd, function(s) rnorm(persons, 0, s))
simulated_testlets <- sapply(1:12, function(j) rbinom(persons, 1, plogis(a[j] * (theta + effects[, testlet_of[j]] - b[j]))))
colnames(simulated_testlets) <- sprintf("q%02d", 1:12)

data(data.read, package = "sirt")
read_testlet <- as.integer(factor(substr(colnames(data.read), 1, 1)))

simulated_fit <- testlet_fit(simulated_testlets, testlet_of)
simulated_case <- testlet_case("simulated testlets, base R maximum", simulated_testlets, testlet_of, c("t1", "t2", "t3"),
                               mirt_testlet_start(simulated_fit, 12))
message("Simulated testlets: the maximum's log-likelihood exceeds mirt's by ",
        simulated_case$logLikelihood - logLik(simulated_fit))
stopifnot(abs(simulated_case$logLikelihood - logLik(simulated_fit)) < 1e-3)
# On data.read mirt's bfactor reaches a degenerate solution, testlet variances near 58; base R starts from the 2PL.
read_fit <- testlet_fit(data.read, read_testlet)
two_pl_read <- coef(mirt(data.read, 1, verbose = FALSE), simplify = TRUE)$items
read_case <- testlet_case("data.read, base R maximum", data.read, read_testlet, c("A", "B", "C"),
                          c(two_pl_read[, "a1"], two_pl_read[, "d"], log(rep(0.5, 3))))
message("data.read: the maximum's log-likelihood exceeds mirt's by ", read_case$logLikelihood - logLik(read_fit))

write_fixture("irt-testlet", list(
  source = "A 2PL testlet simulation, set.seed(20261029): 1000 persons, 3 testlets of 4 items; sirt's data.read.",
  points = TESTLET_POINTS,
  cases = list(simulated_case, read_case)
))

# --- Covariance of the item parameters, item priors and posteriors of theta -----------------------------------------
# The full covariance matrix of the package's estimated item parameters, by the delta method on mirt's vcov() with the
# Jacobian by central differences; item priors by direct maximization of the log-posterior in base R; the posterior of
# theta given a response pattern, written out on mirt's probtrace.

# The covariance of every item's estimated parameters (a and b for the 2PL, the polytomous parameters as in
# delta_errors), with the item and the name of each row.
parameter_covariance <- function(fit, models) {
  items <- colnames(extract.mirt(fit, "data"))
  covariance <- vcov(fit)
  values <- mod2values(fit)
  parnums <- lapply(strsplit(colnames(covariance), ".", fixed = TRUE), function(x) as.integer(x[-1]))
  base <- sapply(parnums, function(p) values$value[values$parnum == p[1]])
  estimates <- function(x) {
    for (j in seq_along(x)) values$value[values$parnum %in% parnums[[j]]] <- x[j]
    lapply(seq_along(items), function(i) {
      v <- item_values(values, items[i])
      if (models[i] %in% LOGISTIC) return(c(a = v[["a1"]], b = -v[["d"]] / v[["a1"]]))
      estimated_parameters(models[i], polytomous_parameters(models[i], v))
    })
  }
  centers <- estimates(base)
  jacobian <- sapply(seq_along(base), function(j) {
    h <- 1e-6 * max(1, abs(base[j]))
    up <- base
    down <- base
    up[j] <- up[j] + h
    down[j] <- down[j] - h
    (unlist(estimates(up)) - unlist(estimates(down))) / (2 * h)
  })
  list(
    parameters = data.frame(item = rep(items, lengths(centers)), name = unlist(lapply(centers, names)),
                            row.names = NULL),
    matrix = unname(jacobian %*% covariance %*% t(jacobian))
  )
}

covariance_case <- function(name, fit, models) {
  covariance <- parameter_covariance(fit, models)
  # The diagonal reproduces the standard errors of the calibration fixtures.
  errors <- unlist(lapply(psychometrics_items(fit, models), function(item) unlist(item$standardErrors)))
  stopifnot(max(abs(sqrt(diag(covariance$matrix)) - errors) / errors) < 1e-6)
  list(name = name, quadrature = list(method = "rectangular", points = POINTS), models = I(models),
       covariance = covariance)
}

# Maximizes the marginal log-likelihood of the 2PL plus the log priors of a and b by BFGS, then by Newton steps, and
# returns the estimates with the covariance from the Hessian of the log-posterior (optimHess). `log_prior` takes the
# items' parameters (a, b, c, d) by rows and returns the sum of the log prior densities; `prior_gradient` returns their
# derivatives with respect to a and b, items by rows.
item_prior_mode <- function(x, start, log_prior, prior_gradient) {
  k <- nrow(start)
  # The parameters are a of every item, then b of every item; the covariance is returned item by item, a then b.
  order <- as.vector(rbind(seq_len(k), k + seq_len(k)))
  parameters <- function(v) cbind(matrix(v, k, 2), 0, 1)
  objective <- function(v) -(marginal(x, parameters(v))$logLikelihood + log_prior(parameters(v)))
  gradient <- function(v) {
    g <- marginal(x, parameters(v))$gradient[, 1:2] + prior_gradient(parameters(v))
    -as.vector(g)
  }
  estimate <- optim(as.vector(start), objective, gradient, method = "BFGS",
                    control = list(reltol = 1e-16, maxit = 10000))$par
  for (step in 1:4) {
    estimate <- estimate - solve(optimHess(estimate, objective, gradient), gradient(estimate))
  }
  stopifnot(max(abs(gradient(estimate))) < 1e-8)
  pars <- parameters(estimate)
  list(items = lapply(seq_len(k), function(i) list(parameters = list(a = pars[i, 1], b = pars[i, 2], c = 0, d = 1))),
       covariance = unname(solve(optimHess(estimate, objective, gradient))[order, order]),
       logLikelihood = marginal(x, pars)$logLikelihood, logPosterior = -objective(estimate))
}

# a ~ lognormal(0.3, 0.5) for every item but the fourth, whose a is lognormal(0, 0.3); b of the second item is
# normal(0.5, 0.4). The package's `priors` and `itemPriors` options.
general_a <- list(meanLog = 0.3, sdLog = 0.5)
item_priors <- list(`Item.4` = list(a = list(type = "log-normal", meanLog = 0, sdLog = 0.3)),
                    `Item.2` = list(b = list(type = "normal", mean = 0.5, sd = 0.4)))
a_specs <- rep(list(general_a), 5)
a_specs[[4]] <- list(meanLog = 0, sdLog = 0.3)
a_mean_log <- sapply(a_specs, `[[`, "meanLog")
a_sd_log <- sapply(a_specs, `[[`, "sdLog")
b_prior <- list(item = 2, mean = 0.5, sd = 0.4)
prior_mode <- item_prior_mode(
  lsat7,
  parameter_matrix(psychometrics_items(two_pl, models_2pl))[, 1:2],
  function(p) {
    sum(dlnorm(p[, 1], a_mean_log, a_sd_log, log = TRUE)) +
      dnorm(p[b_prior$item, 2], b_prior$mean, b_prior$sd, log = TRUE)
  },
  function(p) {
    g <- cbind(-(1 + (log(p[, 1]) - a_mean_log) / a_sd_log^2) / p[, 1], 0)
    g[b_prior$item, 2] <- (b_prior$mean - p[b_prior$item, 2]) / b_prior$sd^2
    g
  }
)
prior_mode$parameters <- data.frame(item = rep(colnames(lsat7), each = 2), name = rep(c("a", "b"), 5), row.names = NULL)
# The marginal likelihood of the written-out 2PL reproduces mirt's log-likelihood at mirt's estimates.
two_pl_parameters <- cbind(parameter_matrix(psychometrics_items(two_pl, models_2pl))[, 1:2], 0, 1)
stopifnot(abs(marginal(lsat7, two_pl_parameters)$logLikelihood - logLik(two_pl)) < 1e-6)

write_fixture("irt-covariance", list(
  source = paste("mirt::LSAT7 under the 2PL and mirt::Science under the GPCM, the calibrations of irt-lsat7 and",
                 "irt-science: the covariance matrix of all estimated item parameters. The 2PL with priors on a and b",
                 "is the base R maximum of the log-posterior."),
  cases = list(
    c(list(data = "irt-lsat7"), covariance_case("2PL, mirt Richardson", two_pl, models_2pl)),
    c(list(data = "irt-science"),
      covariance_case("GPCM, mirt Richardson", polytomous_fits$gpcm, science_models("gpcm"))),
    list(
      data = "irt-lsat7",
      name = "2PL with a lognormal prior on a, and item priors on the a of item 4 and the b of item 2, base R mode",
      quadrature = list(method = "rectangular", points = POINTS), models = I(models_2pl),
      priors = list(a = list(type = "log-normal", meanLog = general_a$meanLog, sdLog = general_a$sdLog)),
      itemPriors = item_priors, items = prior_mode$items, logLikelihood = prior_mode$logLikelihood,
      covariance = list(parameters = prior_mode$parameters, matrix = prior_mode$covariance)
    )
  )
))

# The posterior of theta given a response pattern under a normal prior on `rule`'s nodes: the weights of the prior's
# rectangular rule times the likelihood of the answered responses, normalized.
posterior_weights <- function(fit, pattern, prior) {
  z <- seq(-6, 6, length.out = POINTS)
  nodes <- prior$mean + prior$sd * z
  log_weights <- log(dnorm(z) / sum(dnorm(z)))
  for (i in which(!is.na(pattern))) {
    log_weights <- log_weights + log(probtrace(extract.item(fit, i), nodes)[, pattern[i] + 1])
  }
  weights <- exp(log_weights - max(log_weights))
  weights / sum(weights)
}

posterior_case <- function(fit, models, patterns) {
  standard <- list(mean = 0, sd = 1)
  shifted <- list(mean = 0.4, sd = 1.3)
  # Under the standard normal prior the posterior mean is mirt's EAP.
  z <- seq(-6, 6, length.out = POINTS)
  eap <- fscores(fit, method = "EAP", response.pattern = patterns, quadpts = POINTS)[, 1]
  means <- sapply(seq_len(nrow(patterns)), function(r) sum(posterior_weights(fit, patterns[r, ], standard) * z))
  stopifnot(max(abs(means - eap)) < 1e-8)
  list(
    models = I(models),
    parameters = lapply(psychometrics_items(fit, models), `[[`, "parameters"),
    patterns = patterns,
    priors = lapply(list(standard, shifted), function(prior) {
      list(prior = prior, nodes = prior$mean + prior$sd * z,
           weights = t(sapply(seq_len(nrow(patterns)), function(r) posterior_weights(fit, patterns[r, ], prior))))
    })
  )
}

science_selected <- rbind(c(0, 0, 0, 0), c(3, 3, 3, 3), c(1, 2, 3, 0), c(2, 2, 1, 3), science_patterns[-seq_len(256), ])
write_fixture("irt-posterior", list(
  source = paste("mirt::LSAT7 under the 2PL and mirt::Science under the GPCM, the calibrations of irt-lsat7 and",
                 "irt-science: the posterior of theta on 61 rectangular points by base R, on mirt's probtrace."),
  forms = list(posterior_case(two_pl, models_2pl, patterns),
               posterior_case(polytomous_fits$gpcm, science_models("gpcm"), science_selected))
))

# --- Testlet response model in groups -------------------------------------------------------------------------------
# The testlets' integrals Σ_g w_g Π_j P_j(x_j | θ + γ_g) do not depend on the group, so the marginal likelihood of
# persons in groups multiplies each person's group prior at the nodes of theta by the testlets' integrals. Base R
# maximizes it, as above, over the free items' a and d, each testlet's ln σ, which the groups share, and the mean and ln
# SD of each estimated normal group, on the grid of 21 nodes. A histogram of theta on fixed items is base R's EM with
# measurekit's updates and stopping rule. Every person answered every item.

# For each testlet: ln ℓ over the grid of θ + γ scaled by its largest value per person, the weights of γ and the
# integral at each node of theta.
testlet_pieces <- function(x, testlet, a, d, sd) {
  nodes <- seq(-6, 6, length.out = TESTLET_POINTS)
  sums <- seq(-12, 12, length.out = 2 * TESTLET_POINTS - 1)
  lapply(seq_along(sd), function(t) {
    items <- which(testlet == t)
    z <- outer(sums, a[items]) + matrix(d[items], length(sums), length(items), byrow = TRUE)
    responses <- x[, items, drop = FALSE]
    s <- responses %*% t(plogis(z, log.p = TRUE)) + (1 - responses) %*% t(plogis(-z, log.p = TRUE))
    v <- dnorm(nodes / sd[t])
    v <- v / sum(v)
    largest <- apply(s, 1, max)
    e <- exp(s - largest)
    integral <- sapply(seq_len(TESTLET_POINTS), function(q) e[, q:(q + TESTLET_POINTS - 1), drop = FALSE] %*% v)
    list(e = e, v = v, integral = integral, log_integral = largest + log(integral))
  })
}

testlet_normal_weights <- function(mean, sd) {
  w <- dnorm(seq(-6, 6, length.out = TESTLET_POINTS), mean, sd)
  w / sum(w)
}

# A problem: the responses, each item's testlet, each person's group (1, 2, ...), the distribution of each group
# ("standard-normal", "normal" or "empirical-histogram"), the items' a and d where fixed, the free items, and the weights
# of the groups that have a histogram. The parameters are the free items' a, then their d, then ln σ of each testlet, then
# the mean and ln SD of each normal group.
group_problem <- function(x, testlet, group, kinds, a, d, free, histograms = list()) {
  list(x = x, testlet = testlet, group = group, kinds = kinds, a = a, d = d, free = free, histograms = histograms)
}

unpack_groups <- function(par, problem) {
  k <- length(problem$free)
  t <- length(unique(problem$testlet))
  a <- problem$a
  d <- problem$d
  a[problem$free] <- par[seq_len(k)]
  d[problem$free] <- par[k + seq_len(k)]
  list(a = a, d = d, sd = exp(par[2 * k + seq_len(t)]), latent = matrix(par[-seq_len(2 * k + t)], ncol = 2, byrow = TRUE))
}

# The log weights of each group at the nodes of theta, by group, then node.
group_log_weights <- function(u, problem) {
  normal <- 0
  t(sapply(seq_along(problem$kinds), function(g) {
    if (problem$kinds[g] == "normal") {
      normal <<- normal + 1
      return(log(testlet_normal_weights(u$latent[normal, 1], exp(u$latent[normal, 2]))))
    }
    if (problem$kinds[g] == "empirical-histogram") return(log(problem$histograms[[g]]))
    log(testlet_normal_weights(0, 1))
  }))
}

# Each person's posterior over the nodes of theta, and the log-likelihood.
group_posterior <- function(u, problem, pieces) {
  log_joint <- group_log_weights(u, problem)[problem$group, , drop = FALSE]
  for (piece in pieces) log_joint <- log_joint + piece$log_integral
  largest <- apply(log_joint, 1, max)
  posterior <- exp(log_joint - largest)
  totals <- rowSums(posterior)
  list(posterior = posterior / totals, logLikelihood = sum(largest + log(totals)))
}

# The negative log-likelihood. The testlets' integrals depend on the items and σ only, so they are kept while the
# parameters of the groups move, as they do in the gradient by central differences.
group_objective <- function(problem) {
  last <- NULL
  pieces <- NULL
  function(par) {
    u <- unpack_groups(par, problem)
    key <- c(u$a, u$d, u$sd)
    if (is.null(last) || any(key != last)) {
      pieces <<- testlet_pieces(problem$x, problem$testlet, u$a, u$d, u$sd)
      last <<- key
    }
    -group_posterior(u, problem, pieces)$logLikelihood
  }
}

# The fixture of one case: the data, the options and the estimates with their standard errors, by the delta method for
# b and for each SD.
group_case <- function(name, ids, problem, labels, group_labels, fixed, start) {
  maximum <- maximum_of(group_objective(problem), start)
  stopifnot(max(abs(maximum$gradient)) < 1e-4)
  u <- unpack_groups(maximum$par, problem)
  errors <- sqrt(diag(maximum$covariance))
  k <- length(problem$free)
  t <- length(labels)
  items <- lapply(seq_along(ids), function(j) {
    m <- match(j, problem$free)
    if (is.na(m)) return(list(parameters = fixed[[ids[j]]], standardErrors = NULL))
    jacobian <- c(u$d[j] / u$a[j]^2, -1 / u$a[j])
    block <- maximum$covariance[c(m, k + m), c(m, k + m)]
    list(parameters = list(a = u$a[j], b = -u$d[j] / u$a[j], c = 0, d = 1),
         standardErrors = list(a = errors[m], b = sqrt(drop(t(jacobian) %*% block %*% jacobian))))
  })
  normal <- 0
  distributions <- lapply(seq_along(problem$kinds), function(g) {
    if (problem$kinds[g] != "normal") return(list(mean = 0, sd = 1, standardErrors = NULL))
    normal <<- normal + 1
    at <- 2 * k + t + 2 * (normal - 1) + 1:2
    sd <- exp(maximum$par[at[2]])
    list(mean = maximum$par[at[1]], sd = sd, standardErrors = list(mean = errors[at[1]], sd = sd * errors[at[2]]))
  })
  list(name = name, itemIds = ids, responses = unname(problem$x), groups = group_labels[problem$group],
       testlets = setNames(lapply(seq_len(t), function(s) ids[problem$testlet == s]), labels),
       latent = setNames(as.list(problem$kinds), unique(group_labels))[problem$kinds != "standard-normal"],
       fixed = fixed, items = items, sd = u$sd, sdStandardErrors = u$sd * errors[2 * k + seq_len(t)],
       distributions = distributions, parameterCount = length(maximum$par), logLikelihood = maximum$logLikelihood)
}

# Responses of persons in groups to 2PL items in testlets: theta ~ N(mean, sd²) in each group, and a testlet effect with
# its own SD for each testlet.
simulate_testlet_groups <- function(sizes, means, sds, effect_sd, a, b, testlet) {
  group <- rep(seq_along(sizes), sizes)
  theta <- rnorm(length(group), means[group], sds[group])
  effects <- sapply(effect_sd, function(s) rnorm(length(group), 0, s))
  x <- sapply(seq_along(a), function(j) rbinom(length(group), 1, plogis(a[j] * (theta + effects[, testlet[j]] - b[j]))))
  colnames(x) <- sprintf("q%02d", seq_along(a))
  list(x = x, group = group, theta = theta)
}

# mirt's cross-check: bfactor with the groups, each item's slopes and intercept equal across them, each item's two
# slopes equal, and the variance of each testlet equal across them and free, with the specific means fixed at 0. The
# first group is standard normal and the other groups' general mean and variance are free.
mirt_testlet_groups <- function(x, testlet, group) {
  data <- as.data.frame(x)
  group <- factor(group)
  values <- bfactor(data, testlet, group = group, pars = "values",
                    invariance = c("slopes", "intercepts", "free_means", "free_var"))
  specific <- sprintf("COV_%d%d", 1 + unique(testlet), 1 + unique(testlet))
  values$est[values$name %in% specific] <- TRUE
  values$est[values$name %in% sprintf("MEAN_%d", 1 + unique(testlet))] <- FALSE
  values$value[values$name %in% sprintf("MEAN_%d", 1 + unique(testlet))] <- 0
  slopes <- unlist(lapply(levels(group), function(g) {
    lapply(colnames(data), function(id) {
      values$parnum[values$group == g & values$item == id & grepl("^a", values$name) & values$value != 0]
    })
  }), recursive = FALSE)
  variances <- lapply(specific, function(name) values$parnum[values$name == name])
  fit <- bfactor(data, testlet, group = group, pars = values, constrain = c(slopes, variances),
                 invariance = c("slopes", "intercepts"), TOL = 1e-8, quadpts = TESTLET_POINTS, verbose = FALSE,
                 technical = list(NCYCLES = 1e5))
  stopifnot(extract.mirt(fit, "converged"))
  as.numeric(logLik(fit))
}

set.seed(20261029)
group_a <- round(runif(12, 0.8, 1.8), 2)
group_b <- round(rnorm(12), 2)
group_effect_sd <- c(0.5, 0.8, 1)
group_testlet <- rep(1:3, each = 4)
group_ids <- sprintf("q%02d", 1:12)
testlet_labels <- c("t1", "t2", "t3")

# The 2PL fit of all persons as a start: a, d, ln σ and the groups' means and ln SDs.
pooled_start <- function(x, free, normal) {
  pooled <- coef(mirt(as.data.frame(x), 1, itemtype = "2PL", verbose = FALSE), simplify = TRUE)$items
  c(pooled[free, "a1"], pooled[free, "d"], log(rep(0.5, 3)), rep(c(0, 0), normal))
}

group_cases <- list()
empty_object <- setNames(list(), character(0))

# Two groups: the second has mean 0.6 and SD 1.2; the first is standard normal.
draw <- simulate_testlet_groups(c(1000, 1000), c(0, 0.6), c(1, 1.2), group_effect_sd, group_a, group_b, group_testlet)
problem <- group_problem(draw$x, group_testlet, draw$group, c("standard-normal", "normal"),
                         rep(1, 12), rep(0, 12), 1:12)
group_cases[[1]] <- group_case("two groups, 2PL testlets, base R maximum", group_ids, problem, testlet_labels,
                               c("a", "b"), empty_object, pooled_start(draw$x, 1:12, 1))
group_cases[[1]]$mirtLogLikelihood <- mirt_testlet_groups(draw$x, group_testlet, draw$group)
message("Two groups: the maximum's log-likelihood exceeds mirt's by ",
        group_cases[[1]]$logLikelihood - group_cases[[1]]$mirtLogLikelihood)
stopifnot(abs(group_cases[[1]]$logLikelihood - group_cases[[1]]$mirtLogLikelihood) < 1e-3)

# Three groups: the second has mean 0.5 and SD 1.1, the third mean -0.4 and SD 0.8.
draw <- simulate_testlet_groups(c(700, 700, 700), c(0, 0.5, -0.4), c(1, 1.1, 0.8), group_effect_sd, group_a, group_b,
                                  group_testlet)
problem <- group_problem(draw$x, group_testlet, draw$group, c("standard-normal", "normal", "normal"),
                         rep(1, 12), rep(0, 12), 1:12)
group_cases[[2]] <- group_case("three groups, 2PL testlets, base R maximum", group_ids, problem, testlet_labels,
                               c("a", "b", "c"), empty_object, pooled_start(draw$x, 1:12, 2))

# Fixed items, one in each testlet and a second in the first, with a normal distribution in both groups: the anchors set
# the scale, so that the means and SDs of both are estimated.
anchors <- c(1, 2, 5, 9)
fixed_items <- setNames(lapply(anchors, function(j) list(a = group_a[j], b = group_b[j], c = 0, d = 1)), group_ids[anchors])
draw <- simulate_testlet_groups(c(800, 800), c(0.3, 0.9), c(0.9, 1.1), group_effect_sd, group_a, group_b, group_testlet)
problem <- group_problem(draw$x, group_testlet, draw$group, c("normal", "normal"),
                         group_a, -group_a * group_b, setdiff(1:12, anchors))
group_cases[[3]] <- group_case("fixed items and normal groups, 2PL testlets, base R maximum", group_ids, problem,
                               testlet_labels, c("a", "b"), fixed_items,
                               pooled_start(draw$x, setdiff(1:12, anchors), 2))

# A histogram on fixed items, base R's EM from the standard normal weights and testlet SDs of 1: each cycle the second
# group's weights are its expected counts over their number, and each SD is the one whose weights have the second moment
# of the expected counts of its testlet effect, until no weight and no SD changes by more than the tolerance.
nodes <- seq(-6, 6, length.out = TESTLET_POINTS)
latent_draw <- function(n) ifelse(runif(n) < 0.6, rnorm(n, -0.3, 0.6), rnorm(n, 1.2, 0.5))
group <- rep(1:2, each = 800)
theta <- ifelse(group == 1, rnorm(1600), latent_draw(1600))
effects <- sapply(group_effect_sd, function(s) rnorm(1600, 0, s))
x <- sapply(1:12, function(j) rbinom(1600, 1, plogis(group_a[j] * (theta + effects[, group_testlet[j]] - group_b[j]))))
colnames(x) <- group_ids
histogram_tolerance <- 1e-4
weights <- testlet_normal_weights(0, 1)
sd <- rep(1, 3)
for (cycle in 1:1e5) {
  problem <- group_problem(x, group_testlet, group, c("standard-normal", "empirical-histogram"), group_a,
                           -group_a * group_b, integer(0), list(NULL, weights))
  pieces <- testlet_pieces(x, group_testlet, group_a, -group_a * group_b, sd)
  e <- group_posterior(unpack_groups(c(log(sd)), problem), problem, pieces)
  updated <- colSums(e$posterior[group == 2, ]) / sum(group == 2)
  updated_sd <- sapply(seq_along(sd), function(t) {
    piece <- pieces[[t]]
    factor <- e$posterior / piece$integral
    counts <- piece$v * sapply(seq_len(TESTLET_POINTS), function(h) {
      sum(factor * piece$e[, h:(h + TESTLET_POINTS - 1)])
    })
    target <- sum(counts * nodes^2) / sum(counts)
    excess <- function(log_sd) {
      w <- dnorm(nodes / exp(log_sd))
      sum(w * nodes^2) / sum(w) - target
    }
    exp(uniroot(excess, log(c(1e-4, 1e4)), tol = 1e-14)$root)
  })
  change <- max(abs(updated_sd - sd), abs(updated - weights))
  weights <- updated
  sd <- updated_sd
  if (change <= histogram_tolerance) break
}
stopifnot(change <= histogram_tolerance)
problem <- group_problem(x, group_testlet, group, c("standard-normal", "empirical-histogram"), group_a,
                         -group_a * group_b, integer(0), list(NULL, weights))
histogram_log_likelihood <- group_posterior(unpack_groups(log(sd), problem), problem,
                                            testlet_pieces(x, group_testlet, group_a, -group_a * group_b, sd))$logLikelihood
all_fixed <- setNames(lapply(1:12, function(j) list(a = group_a[j], b = group_b[j], c = 0, d = 1)), group_ids)

write_fixture("irt-testlet-groups", list(
  source = paste("2PL testlet simulations with groups, set.seed(20261029): two groups of 1000 and 800, three of 700, in 3",
                 "testlets of 4 items; base R's maximum of the marginal likelihood with the testlet SDs shared by the",
                 "groups, mirt's bfactor with the same constraints as a cross-check, and base R's EM for a histogram."),
  points = TESTLET_POINTS,
  cases = group_cases,
  histogram = list(
    name = "a histogram of the second group on 12 fixed items in testlets, base R EM",
    itemIds = group_ids, responses = unname(x), groups = c("a", "b")[group], testlets = list(t1 = group_ids[1:4],
      t2 = group_ids[5:8], t3 = group_ids[9:12]), fixed = all_fixed, tolerance = histogram_tolerance,
    iterations = cycle, weights = weights, sd = sd, logLikelihood = histogram_log_likelihood
  )
))
