# Adaptive testing fixtures from catR: the next item by maximum Fisher information, Kullback-Leibler, thOpt and, within
# aStratified's strata, bOpt (nextItem); the estimate and its standard error after each response (thetaEst, semTheta);
# content balancing (nextItem with cbControl) and stopping decisions (checkStopRule). Banks are catR's generated banks;
# each run's responses are drawn once from a true theta with a fixed seed and are then a fixed sequence.
#
# Adaptive cognitive diagnosis: tests replayed step by step on the items of GDINA calibrations in base R, written out
# from the formulas of each rule, and the item sequences of cdcatR's cdcat for the rules it has. cdcatR imports
# cdmTools, which only its nonparametric rules use and whose dependency clarabel needs a Rust compiler, so
# install.packages("cdcatR") fails. Download the source with download.packages, remove cdmTools from the Imports of
# DESCRIPTION and `import(cdmTools)` from NAMESPACE, and run R CMD INSTALL on the directory.
#
# Testlets: tests replayed step by step in base R, written out from the documented rules: a testlet scored as a unit by the
# mean information of its eligible items at the EAP estimate, its items then administered together in pool order or by
# information, the test ending between testlets. catR's Ii gives the information. catR has no testlets; mirtCAT's ordered
# constraint administers a group in order once its most informative item is selected and does not end the test inside
# it, which is the same where each group's items are equally informative, so mirtCAT replays that case, ended by precision:
# at a length that falls inside a group mirtCAT stops with "Ran out of items to administer".
#
# Multidimensional adaptive tests: tests replayed step by step in base R, with the information matrices of the
# compensatory logistic items written out and the posterior mode found to a gradient of 1e-12, and the item sequences of
# mirtCAT's DPrule, which is the D-rule with the inverse of the prior covariance added.

source("validation/psychometrics/common.R")
suppressPackageStartupMessages(library(catR))

SEED <- 20261029
RANGE <- c(-6, 6) # measurekit's default theta range for ML, WLE and MAP
# measurekit's EAP integrates over 61 equally spaced points on [-6, 6] with weights proportional to the prior; catR's
# trapezoid rule on the same points halves the two end weights, where the density is 6e-9.
EAP_GRID <- c(-6, 6, 61)
KL_GRID <- c(-4, 4, 33) # catR's default parInt for KL, which measurekit uses
LENGTH <- 15

item_id <- function(i) sprintf("item%03d", i)

# measurekit's parameters of catR's 4PL and GPCM bank rows; catR uses D = 1 by default, as measurekit does.
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
# by the largest likelihood on the grid as measurekit scales it.
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

# ---- Adaptive cognitive diagnosis ---------------------------------------------------------------------------------

CD_LENGTH <- 10 # items per test
CD_PERSONS <- 4 # persons per case, the first rows of the data
# The prior in multiples of 1 / 4096, so that its sum is exactly 1 as cdcatR's att.prior check requires.
PRIOR_UNITS <- 4096

# The Q-matrix's columns in the package's order: attributes by first appearance over the items, so that both libraries
# order the profiles and each item's latent groups alike.
psychometrics_order <- function(Q) {
  first <- unique(unlist(lapply(seq_len(nrow(Q)), function(j) which(Q[j, ] == 1))))
  Q[, first, drop = FALSE]
}

# The prior of a fit's profile proportions in multiples of 1 / PRIOR_UNITS, each profile `allowed` at least one.
rounded_prior <- function(proportions, allowed) {
  units <- ifelse(allowed, pmax(round(proportions * PRIOR_UNITS), 1), 0)
  units[which.max(units)] <- units[which.max(units)] + PRIOR_UNITS - sum(units)
  units / PRIOR_UNITS
}

# P(correct) of every item for every profile: the item's success probability in the latent group of the profile.
profile_probabilities <- function(fit, Q) {
  patterns <- GDINA::attributepattern(ncol(Q))
  probabilities <- coef(fit, digits = 15)
  t(sapply(seq_len(nrow(Q)), function(j) {
    attributes <- which(Q[j, ] == 1)
    groups <- apply(GDINA::attributepattern(length(attributes)), 1, paste, collapse = "")
    unname(probabilities[[j]][match(apply(patterns[, attributes, drop = FALSE], 1, paste, collapse = ""), groups)])
  }))
}

# The Kullback-Leibler divergence of a correct-response probability q from p, summed over both responses.
divergence <- function(p, q) p * log(p / q) + (1 - p) * log((1 - p) / (1 - q))

# The score of an item, whose probabilities of a correct response by profile are `lc`, at the posterior `post`, from the
# definition of each rule over the profiles.
score_item <- function(rule, lc, post) {
  mode <- lc[which.max(post)]
  switch(rule,
    KL = sum(divergence(mode, lc)),
    PWKL = sum(post * divergence(mode, lc)),
    MPWKL = sum(sapply(seq_along(lc), function(d) post[d] * sum(post * divergence(lc[d], lc)))),
    GDI = sum(post * (lc - sum(post * lc))^2),
    # Minus the expected entropy of the posterior after the response.
    SHE = -sum(sapply(c(1, 0), function(x) {
      weights <- if (x == 1) lc else 1 - lc
      probability <- sum(post * weights)
      after <- post * weights / probability
      after <- after[after > 0]
      probability * -sum(after * log(after))
    }))
  )
}

posterior_after <- function(post, lc, response) {
  weights <- if (response == 1) lc else 1 - lc
  post * weights / sum(post * weights)
}

# The state of a test: the posterior, the attribute mastery and the MAP profile (the first on ties).
estimate <- function(post, patterns) {
  list(posterior = post, mastery = as.vector(post %*% patterns), map = unname(patterns[which.max(post), ]))
}

# A test of CD_LENGTH items: before each item the scores of the items not administered at the posterior so far, then the
# item with the highest score, the first on ties, and the person's response to it in `row`.
replay <- function(rule, lc, prior, patterns, row) {
  administered <- integer(0)
  post <- prior
  steps <- lapply(seq_len(CD_LENGTH), function(k) {
    scores <- sapply(seq_len(nrow(lc)), function(j) score_item(rule, lc[j, ], post))
    scores[administered] <- NA
    item <- which.max(scores)
    step <- c(estimate(post, patterns), list(scores = scores, item = item, response = row[item]))
    administered <<- c(administered, item)
    post <<- posterior_after(post, lc[item, ], row[item])
    step
  })
  list(rule = rule, person = NA, steps = steps, final = estimate(post, patterns), items = administered)
}

# The stopping decisions along a replay: whether each rule stops before item k + 1, as the posterior stands.
stopping_decisions <- function(run) {
  states <- c(lapply(run$steps, function(step) step[c("posterior", "mastery")]), list(run$final))
  lapply(seq_along(states), function(k) {
    posterior <- states[[k]]$posterior
    mastery <- states[[k]]$mastery
    list(administered = k - 1,
         posterior = sapply(c(0.7, 0.9), function(p) max(posterior) >= p),
         mastery = sapply(c(0.7, 0.9), function(p) all(mastery >= p | mastery <= 1 - p)))
  })
}

# cdcatR's PWKL takes the point estimate by sample(which(posterior == max(posterior)), 1), which for one mode m draws a
# number from 1:m, so its item sequences are not reproducible. Its scores at the MAP profile are the reference instead.
check_pwkl <- function(run, lc) {
  for (step in run$steps) {
    reference <- cdcatR:::PWKL.M(lc, step$posterior, point.est = which.max(step$posterior))
    seen <- !is.na(step$scores)
    stopifnot(isTRUE(all.equal(step$scores[seen], reference[seen], tolerance = 1e-12)))
  }
}

# cdcatR's item sequences: the fixed-length test by `rule`, starting with the item of the highest score at the prior.
cdcat_items <- function(fit, data, rule, prior, precision = NULL) {
  result <- cdcatR::cdcat(fit = fit, dat = data, itemSelect = rule, MAXJ = CD_LENGTH, FIXED.LENGTH = is.null(precision),
                          startRule = "max", precision.cut = if (is.null(precision)) 0.8 else precision,
                          att.prior = prior, initial.distr = prior, n.cores = 2, seed = 1, print.progress = FALSE)
  lapply(result$est, function(person) person$item.usage)
}

# A case: the items of a calibration as pool items, the prior, and for CD_PERSONS persons a replay by each rule, with
# cdcatR's item sequences where it has the rule and reproducible sequences. cdcatR's JSD is the mutual information of
# the response and the profile: the Shannon entropy rule's score plus the entropy of the posterior, the same for every
# item, so it ranks items alike.
cd_case <- function(name, calibration_data, Q, model, hierarchy = NULL) {
  Q <- psychometrics_order(Q)
  colnames(calibration_data) <- paste0("item", seq_len(ncol(calibration_data)))
  fit <- GDINA::GDINA(dat = calibration_data, Q = Q, model = model, control = list(conv.crit = 1e-9, maxitr = 20000),
                      verbose = 0)
  stopifnot(GDINA::extract(fit, "convergence"))
  data <- as.matrix(calibration_data[seq_len(CD_PERSONS), ])
  patterns <- GDINA::attributepattern(ncol(Q))
  allowed <- rep(TRUE, nrow(patterns))
  if (!is.null(hierarchy)) allowed <- apply(patterns, 1, paste, collapse = "") %in% hierarchy
  prior <- rounded_prior(c(GDINA::extract(fit, "posterior.prob")), allowed)
  lc <- profile_probabilities(fit, Q)
  rules <- c("GDI", "MPWKL", "PWKL", "SHE", "KL")
  cdcat_rules <- c(GDI = "GDI", MPWKL = "MPWKL", SHE = "JSD")
  runs <- unlist(lapply(rules, function(rule) {
    reference <- if (rule %in% names(cdcat_rules)) cdcat_items(fit, data, cdcat_rules[[rule]], prior)
    lapply(seq_len(CD_PERSONS), function(i) {
      run <- replay(rule, lc, prior, patterns, data[i, ])
      run$person <- i
      if (rule == "PWKL") check_pwkl(run, lc)
      if (!is.null(reference)) {
        if (!identical(as.integer(reference[[i]]), as.integer(run$items))) {
          stop(sprintf("%s, %s, person %d: cdcatR's items %s, written out %s", name, rule, i,
                       paste(reference[[i]], collapse = " "), paste(run$items, collapse = " ")))
        }
        run$cdcat <- I(reference[[i]])
      }
      run
    })
  }), recursive = FALSE)
  gdi_runs <- Filter(function(run) run$rule == "GDI", runs)
  precision <- cdcat_items(fit, data, "GDI", prior, precision = 0.8)
  list(
    name = name,
    attributes = colnames(Q),
    items = lapply(seq_len(nrow(Q)), function(j) list(
      id = colnames(calibration_data)[j], attributes = I(colnames(Q)[Q[j, ] == 1]),
      probabilities = unname(coef(fit, digits = 15)[[j]]))),
    profiles = lapply(seq_len(nrow(patterns)), function(c) {
      list(profile = unname(patterns[c, ]), proportion = prior[c])
    }),
    responses = unname(data),
    runs = runs,
    stopping = lapply(gdi_runs, stopping_decisions),
    precision = lapply(seq_along(precision), function(i) list(person = i, items = I(precision[[i]])))
  )
}

data(sim30GDINA, package = "GDINA")
ecpe_q <- as.matrix(CDM::data.ecpe$q.matrix)
ecpe <- as.matrix(CDM::data.ecpe$data[, -1])

write_fixture("cat-cdm", list(
  source = paste("GDINA's sim30GDINA (1000 persons, 30 items, 5 attributes) and CDM's data.ecpe (2922 persons,",
                 "28 items, 3 attributes), calibrated by GDINA; the first", CD_PERSONS, "persons take each test of",
                 CD_LENGTH, "items"),
  length = CD_LENGTH,
  thresholds = c(0.7, 0.9),
  cases = list(
    cd_case("sim30GDINA with the G-DINA model", as.matrix(sim30GDINA$simdat), as.matrix(sim30GDINA$simQ), "GDINA"),
    cd_case("ECPE with the G-DINA model", ecpe, ecpe_q, "GDINA"),
    cd_case("ECPE with the DINA model", ecpe, ecpe_q, "DINA"),
    cd_case("ECPE with the G-DINA model and a linear hierarchy", ecpe, ecpe_q, "GDINA",
            hierarchy = c("000", "100", "110", "111"))
  )
))

# ---- Multidimensional adaptive tests ----------------------------------------------------------------------------

suppressPackageStartupMessages(library(mirtCAT))

MD_LENGTH <- 10 # items per test
MD_PERSONS <- 4 # persons per case

# The sums over a person's items of the logistic M3PL item: P = g + (1 - g) logistic(a'theta + d), with the information
# matrix ((1 - g) L (1 - L))^2 / (P (1 - P)) a a' of a dichotomous item.
md_probability <- function(items, theta) {
  logistic <- plogis(drop(items$a %*% theta) + items$d)
  items$g + (1 - items$g) * logistic
}
md_information <- function(items, j, theta) {
  logistic <- plogis(sum(items$a[j, ] * theta) + items$d[j])
  p <- items$g[j] + (1 - items$g[j]) * logistic
  slope <- (1 - items$g[j]) * logistic * (1 - logistic)
  slope^2 / (p * (1 - p)) * outer(items$a[j, ], items$a[j, ])
}
md_total_information <- function(items, administered, theta) {
  total <- matrix(0, ncol(items$a), ncol(items$a))
  for (j in administered) total <- total + md_information(items, j, theta)
  total
}

# The posterior mode of theta under N(mean, covariance) from the responses x to the items `administered`: BFGS, then
# Fisher scoring to a gradient below 1e-12. Without items, the prior mean.
md_mode <- function(items, administered, x, prior_mean, precision) {
  if (length(administered) == 0) return(prior_mean)
  log_posterior <- function(theta) {
    p <- md_probability(list(a = items$a[administered, , drop = FALSE], d = items$d[administered],
                             g = items$g[administered]), theta)
    sum(ifelse(x == 1, log(p), log(1 - p))) - 0.5 * drop(t(theta - prior_mean) %*% precision %*% (theta - prior_mean))
  }
  gradient <- function(theta) {
    g <- -drop(precision %*% (theta - prior_mean))
    for (k in seq_along(administered)) {
      j <- administered[k]
      logistic <- plogis(sum(items$a[j, ] * theta) + items$d[j])
      p <- items$g[j] + (1 - items$g[j]) * logistic
      slope <- (1 - items$g[j]) * logistic * (1 - logistic)
      g <- g + items$a[j, ] * slope * (if (x[k] == 1) 1 / p else -1 / (1 - p))
    }
    g
  }
  theta <- optim(prior_mean, function(t) -log_posterior(t), function(t) -gradient(t), method = "BFGS",
                 control = list(reltol = 1e-16, maxit = 1000))$par
  for (iteration in 1:50) {
    step <- solve(precision + md_total_information(items, administered, theta), gradient(theta))
    theta <- theta + step
    if (max(abs(gradient(theta))) < 1e-12) break
  }
  stopifnot(max(abs(gradient(theta))) < 1e-10)
  theta
}

# The estimate after the responses so far: the mode, the covariance (the inverse of the prior precision plus the
# information of the administered items at the mode) and the standard errors.
md_estimate <- function(items, administered, x, prior_mean, covariance) {
  precision <- solve(covariance)
  theta <- md_mode(items, administered, x, prior_mean, precision)
  posterior_covariance <- solve(precision + md_total_information(items, administered, theta))
  list(theta = theta, standardErrors = sqrt(diag(posterior_covariance)), covariance = posterior_covariance)
}

# The D-rule score of every item not administered: det(prior precision + information of the administered items + its
# own) at the estimate; NA for the administered ones.
md_scores <- function(items, administered, theta, precision) {
  base <- precision + md_total_information(items, administered, theta)
  scores <- sapply(seq_len(nrow(items$a)), function(j) det(base + md_information(items, j, theta)))
  scores[administered] <- NA
  scores
}

# A test of MD_LENGTH items for the responses `row` to every item: before each item the estimate from the responses so
# far and the score of each item, then the first item with the highest score.
md_replay <- function(items, covariance, row) {
  precision <- solve(covariance)
  prior_mean <- rep(0, ncol(items$a))
  administered <- integer(0)
  steps <- lapply(seq_len(MD_LENGTH), function(k) {
    estimate <- md_estimate(items, administered, row[administered], prior_mean, covariance)
    scores <- md_scores(items, administered, estimate$theta, precision)
    item <- which.max(scores)
    step <- list(theta = estimate$theta, standardErrors = estimate$standardErrors,
                 covariance = unname(estimate$covariance), scores = scores, item = item, response = row[item])
    administered <<- c(administered, item)
    step
  })
  final <- md_estimate(items, administered, row[administered], prior_mean, covariance)
  list(steps = steps, final = list(theta = final$theta, standardErrors = final$standardErrors,
                                   covariance = unname(final$covariance)), items = administered)
}

# mirtCAT's item sequence for the same responses: DPrule with the MAP estimate, from the item DPrule selects at the
# prior.
md_mirtcat <- function(model, row) {
  result <- mirtCAT(mo = model, criteria = "DPrule", start_item = "DPrule", method = "MAP",
                    local_pattern = matrix(row, 1), design = list(max_items = MD_LENGTH), verbose = FALSE)
  as.integer(result$items_answered)
}

# The stopping decisions along a replay: whether every standard error is at most each threshold before item k + 1.
md_stopping <- function(run) {
  states <- c(lapply(run$steps, function(step) step$standardErrors), list(run$final$standardErrors))
  lapply(seq_along(states), function(k) {
    list(administered = k - 1, stops = sapply(c(0.45, 0.6), function(threshold) all(states[[k]] <= threshold)))
  })
}

md_case <- function(name, a, d, g, covariance, thetas, seed) {
  items <- list(a = a, d = d, g = g)
  J <- nrow(a)
  pars <- data.frame(a, d = d, g = g)
  colnames(pars)[seq_len(ncol(a))] <- paste0("a", seq_len(ncol(a)))
  model <- generate.mirt_object(pars, if (any(g > 0)) "3PL" else "2PL", latent_covariance = covariance)
  set.seed(seed)
  responses <- t(sapply(seq_len(nrow(thetas)), function(i) rbinom(J, 1, md_probability(items, thetas[i, ]))))
  runs <- lapply(seq_len(nrow(thetas)), function(i) {
    run <- md_replay(items, covariance, responses[i, ])
    reference <- md_mirtcat(model, responses[i, ])
    if (!identical(reference, as.integer(run$items))) {
      stop(sprintf("%s, person %d: mirtCAT's items %s, written out %s", name, i, paste(reference, collapse = " "),
                   paste(run$items, collapse = " ")))
    }
    c(list(person = i, mirtCAT = I(reference), stopping = md_stopping(run)), run)
  })
  list(
    name = name,
    dimensions = ncol(a),
    covariance = unname(covariance),
    items = lapply(seq_len(J), function(j) {
      list(id = sprintf("item%02d", j), a = I(a[j, ]), intercept = d[j], c = g[j], d = 1)
    }),
    responses = unname(responses),
    runs = runs
  )
}

# Pools of between-dimension items (one slope) and within-dimension items (several), with item intercepts around 0.
md_pool <- function(J, dimensions, guessing, seed) {
  set.seed(seed)
  a <- matrix(runif(J * dimensions, 0.4, 1.6), J)
  for (j in seq_len(J)) {
    zero <- if (j <= J / 2) sample(dimensions, dimensions - 1) else integer(0)
    a[j, zero] <- 0
  }
  list(a = a, d = rnorm(J, 0, 0.8), g = if (guessing) runif(J, 0, 0.2) else rep(0, J))
}

correlated <- matrix(c(1, 0.5, 0.5, 1), 2)
pool2 <- md_pool(30, 2, FALSE, SEED + 1)
pool2g <- md_pool(30, 2, TRUE, SEED + 2)
pool3 <- md_pool(36, 3, FALSE, SEED + 3)
set.seed(SEED + 4)
thetas2 <- matrix(rnorm(MD_PERSONS * 2), MD_PERSONS)
thetas3 <- matrix(rnorm(MD_PERSONS * 3), MD_PERSONS)

write_fixture("cat-mirt", list(
  source = paste("Pools of 30 to 36 compensatory logistic items on 2 and 3 dimensions, set in mirt",
                 "(generate.mirt_object) and seeded; the first", MD_PERSONS, "persons of each case take a test of",
                 MD_LENGTH, "items."),
  length = MD_LENGTH,
  thresholds = c(0.45, 0.6),
  cases = list(
    md_case("M2PL, 2 dimensions", pool2$a, pool2$d, pool2$g, diag(2), thetas2, SEED + 5),
    md_case("M3PL, 2 dimensions", pool2g$a, pool2g$d, pool2g$g, diag(2), thetas2, SEED + 6),
    md_case("M2PL, 2 dimensions, correlated", pool2$a, pool2$d, pool2$g, correlated, thetas2, SEED + 7),
    md_case("M2PL, 3 dimensions", pool3$a, pool3$d, pool3$g, diag(3), thetas3, SEED + 8)
  )
))

# ---- Testlets -----------------------------------------------------------------------------------------------------

# A 3PL bank of 30 items, 16 of them in five testlets of 3, 4, 2, 3 and 4 items scattered through the pool and 14 alone.
# Four persons answer every item once, at their true thetas, so that each replay consumes a fixed set of responses.
TESTLET_PERSONS <- c(-1.4, -0.2, 0.7, 1.6)
TESTLET_LENGTH <- 12

testlet_nodes <- seq(-6, 6, length.out = 61)
testlet_weights <- dnorm(testlet_nodes) / sum(dnorm(testlet_nodes))

# The 3PL probability of a correct response at each node, with catR's parameters a, b, c and d.
testlet_probability <- function(bank, i, theta) bank[i, "c"] + (bank[i, "d"] - bank[i, "c"]) * plogis(bank[i, "a"] * (theta - bank[i, "b"]))

# The posterior mean and SD of theta over the 61 nodes under N(0, 1), weights proportional to the density: measurekit's EAP.
testlet_eap <- function(bank, out, x) {
  posterior <- testlet_weights
  for (k in seq_along(out)) {
    p <- testlet_probability(bank, out[k], testlet_nodes)
    posterior <- posterior * if (x[k] == 1) p else 1 - p
  }
  posterior <- posterior / sum(posterior)
  theta <- sum(posterior * testlet_nodes)
  list(theta = theta, standardError = sqrt(sum(posterior * (testlet_nodes - theta)^2)))
}

# A replay of one person's test. Rules: before each item, the estimate; items administered, withheld or enemies of an
# administered item are not eligible; the testlet in progress, that of the last administered item while it has eligible
# items, continues without the stopping rule or exposure control, with its first eligible item in pool order or, adaptively,
# the one of highest information; otherwise the test ends if `stop` says so, else each testlet with eligible items is a unit
# scored by the mean information of those items, given to its first item to be administered, and the highest unit is taken,
# the first in pool order on ties, unless the entry item's exposure parameter, 0 or 1, withholds all its eligible items: a
# parameter of 0 does, except for the last eligible unit.
testlet_replay <- function(bank, testlet, exposure, enemies, within, stop, responses) {
  n <- nrow(bank)
  enemy_of <- lapply(seq_len(n), function(i) unlist(lapply(enemies, function(set) if (i %in% set) setdiff(set, i))))
  out <- integer(0)
  x <- numeric(0)
  withheld <- integer(0)
  estimates <- list()
  repeat {
    current <- testlet_eap(bank, out, x)
    estimates[[length(estimates) + 1]] <- current
    eligible <- rep(TRUE, n)
    eligible[c(out, withheld)] <- FALSE
    for (a in out) eligible[enemy_of[[a]]] <- FALSE
    information <- Ii(current$theta, bank)$Ii
    last <- if (length(out) > 0) out[length(out)] else NA
    remaining <- if (!is.na(last) && !is.na(testlet[last])) which(eligible & !is.na(testlet) & testlet == testlet[last]) else integer(0)
    entry_of <- function(members) if (within == "in-order") members[1] else members[which.max(information[members])]
    if (length(remaining) > 0) {
      item <- entry_of(remaining)
    } else if (!any(eligible) || stop(out, current)) {
      break
    } else {
      repeat {
        scores <- rep(-Inf, n)
        for (i in which(eligible)) {
          members <- if (is.na(testlet[i])) i else which(eligible & !is.na(testlet) & testlet == testlet[i])
          if (is.na(testlet[i])) {
            scores[i] <- information[i]
          } else if (i == entry_of(members)) {
            scores[i] <- mean(information[members])
          }
        }
        item <- which.max(scores)
        unit <- if (is.na(testlet[item])) item else which(eligible & !is.na(testlet) & testlet == testlet[item])
        if (sum(eligible) == length(unit) || exposure[item] >= 1) break
        withheld <- c(withheld, unit)
        eligible[unit] <- FALSE
      }
    }
    out <- c(out, item)
    x <- c(x, responses[item])
  }
  list(administered = I(item_id(out)), responses = I(x), estimates = estimates[-1], initial = estimates[[1]],
       withheld = I(item_id(withheld)))
}

set.seed(SEED + 600)
testlet_bank <- as.matrix(genDichoMatrix(items = 30, model = "3PL", seed = SEED + 601))
testlet_sizes <- c(3, 4, 2, 3, 4)
testlet_of <- rep(NA_character_, 30)
members <- sample(30, sum(testlet_sizes))
testlet_of[members] <- rep(sprintf("set%d", seq_along(testlet_sizes)), testlet_sizes)
testlet_responses <- t(sapply(TESTLET_PERSONS, function(theta) {
  as.numeric(runif(30) < sapply(seq_len(30), function(i) testlet_probability(testlet_bank, i, theta)))
}))
stopifnot(abs(testlet_probability(testlet_bank, 1, 0.3) - Pi(0.3, testlet_bank[1, , drop = FALSE])$Pi) < 1e-14,
          abs(testlet_eap(testlet_bank, c(1, 2), c(1, 0))$theta -
                thetaEst(testlet_bank[1:2, ], c(1, 0), method = "EAP", parInt = EAP_GRID)) < 1e-6)

testlet_exposure <- rep(1, 30)
testlet_exposure[which(testlet_of == "set2")] <- 0
testlet_exposure[setdiff(which(is.na(testlet_of)), which(is.na(testlet_of))[1:6])[1:3]] <- 0
set1 <- which(testlet_of == "set1")
set3 <- which(testlet_of == "set3")
testlet_enemies <- list(c(set1[1], set1[3]), c(set3[2], which(is.na(testlet_of))[2]))

# mirtCAT's items for the same responses: criterion MI with the EAP estimate, from the item of most information at the prior
# mean; each testlet an ordered group, whose first item is administered once any of its items is selected; the test ends
# when the standard error is at most `precision`.
testlet_mirtcat <- function(bank, testlet, precision, row) {
  pars <- data.frame(a1 = bank[, "a"], d = -bank[, "a"] * bank[, "b"], g = bank[, "c"])
  model <- generate.mirt_object(pars, "3PL")
  groups <- lapply(unique(na.omit(testlet)), function(set) which(testlet == set))
  result <- mirtCAT(mo = model, criteria = "MI", start_item = "MI", method = "EAP", local_pattern = matrix(row, 1),
                    design = list(min_SEM = precision, constraints = setNames(groups, rep("ordered", length(groups)))),
                    verbose = FALSE)
  as.integer(result$items_answered)
}

testlet_case <- function(name, within, stopping, exposure = rep(1, 30), enemies = list(), bank = testlet_bank,
                         mirtcat = FALSE) {
  stop_rule <- if (stopping$type == "length") {
    function(out, current) length(out) >= stopping$value
  } else {
    function(out, current) current$standardError <= stopping$value
  }
  persons <- lapply(seq_along(TESTLET_PERSONS), function(p) {
    replay <- testlet_replay(bank, testlet_of, exposure, enemies, within, stop_rule, testlet_responses[p, ])
    message(name, ", person ", p, ": ", length(replay$administered), " items ", paste(sub("item0*", "", replay$administered), collapse = " "))
    reference <- NULL
    if (mirtcat) {
      reference <- item_id(testlet_mirtcat(bank, testlet_of, stopping$value, testlet_responses[p, ]))
      if (!identical(reference, as.character(replay$administered))) {
        stop(sprintf("%s, person %d: mirtCAT's items %s, written out %s", name, p, paste(reference, collapse = " "),
                     paste(replay$administered, collapse = " ")))
      }
    }
    c(replay, list(mirtCAT = if (is.null(reference)) NULL else I(reference)))
  })
  list(name = name, within = within, stopping = stopping, enemies = lapply(enemies, item_id),
       items = lapply(seq_len(30), function(i) c(logistic_items(bank)[[i]],
                                                  list(testlet = if (is.na(testlet_of[i])) NULL else testlet_of[i],
                                                       exposure = exposure[i]))),
       persons = persons)
}

# The same bank with each testlet's items given the parameters of its first item, which mirtCAT's selection of a group by its
# most informative item agrees with.
identical_bank <- testlet_bank
for (set in unique(na.omit(testlet_of))) {
  members <- which(testlet_of == set)
  identical_bank[members, ] <- matrix(identical_bank[members[1], ], length(members), ncol(identical_bank), byrow = TRUE)
}

write_fixture("cat-testlets", list(
  source = sprintf("catR's genDichoMatrix(items = 30, model = \"3PL\", seed = %d) with 16 items in five testlets; replays in base R, catR's Ii for the information, and mirtCAT's ordered groups.", SEED + 601),
  responses = unname(testlet_responses),
  cases = list(
    testlet_case("testlets in order", "in-order", list(type = "length", value = TESTLET_LENGTH)),
    testlet_case("testlets adaptively", "adaptive", list(type = "length", value = TESTLET_LENGTH)),
    testlet_case("precision between testlets", "in-order", list(type = "precision", value = 0.45)),
    testlet_case("withheld testlets", "adaptive", list(type = "length", value = TESTLET_LENGTH), testlet_exposure),
    testlet_case("enemies inside and across testlets", "in-order", list(type = "length", value = TESTLET_LENGTH),
                 enemies = testlet_enemies),
    testlet_case("testlets of equally informative items", "in-order", list(type = "precision", value = 0.45),
                 bank = identical_bank, mirtcat = TRUE)
  )
))
