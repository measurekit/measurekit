# Generalizability fixtures. References: base R's aov for the degrees of freedom, sums of squares and mean squares of
# balanced designs; the expected mean squares of each design written out and solved as a linear system with solve();
# lme4's REML estimates, which equal the analysis of variance estimates when every component is positive; and the
# textbook formulas of each design for decision studies (Brennan, 2001, chapters 3 and 4), typed out below.
# gtheory, which wraps lme4 for this, has been removed from CRAN, so lme4 is used directly.

source("validation/psychometrics/common.R")
source("validation/psychometrics/generalizability-simulation.R")
suppressPackageStartupMessages(library(lme4))

SEED <- 20261029
OBJECT <- "person"

# --- A case ---------------------------------------------------------------------------------------------------------
# `codes` are the facets in the design's order, the object first, named by the facet. A term lists an effect's own
# facets and those it is nested in, with its name in the aov and lme4 formulas; the residual is the effect of the most
# facets, which the models leave to the error. `label` names the component in the textbook formulas of the studies.

effect_letters <- function(effect) union(effect$own, effect$nested_in)

# Orders effects by their number of facets, then by the design positions of their facets, the first position first.
order_effects <- function(effects, codes) {
  width <- length(codes)
  keys <- do.call(rbind, lapply(effects, function(effect) {
    positions <- sort(match(effect_letters(effect), codes))
    c(length(positions), positions, rep(0, width - length(positions)))
  }))
  effects[do.call(order, as.data.frame(keys))]
}

# The expected mean squares of the effects solved as a linear system: effect a's mean square is the sum, over the effects
# b whose facets include a's, of the product of the sizes of the facets b lacks times b's variance.
solve_components <- function(effects, mean_squares, n) {
  letters_of <- lapply(effects, effect_letters)
  everything <- names(n)
  coefficients <- outer(seq_along(effects), seq_along(effects), Vectorize(function(a, b) {
    if (all(letters_of[[a]] %in% letters_of[[b]])) prod(n[setdiff(everything, letters_of[[b]])]) else 0
  }))
  solve(coefficients, mean_squares)
}

# lme4 stops where the REML criterion is flat to its tolerance, about 1e-6 from the maximum even with bobyqa's tightest
# setting, so the fit is restarted from its own estimate three times; the estimates then agree with the analysis of
# variance to about 1e-6.
reml_components <- function(formula, data, effects) {
  controls <- lmerControl(check.nobs.vs.nlev = "ignore", check.nobs.vs.rankZ = "ignore",
                          check.nobs.vs.nRE = "ignore", optimizer = "bobyqa",
                          optCtrl = list(rhobeg = 0.2, rhoend = 1e-12, maxfun = 1e5))
  fit <- lmer(formula, data, REML = TRUE, control = controls)
  for (restart in 1:3) fit <- update(fit, start = list(theta = getME(fit, "theta")), control = controls)
  estimates <- as.data.frame(VarCorr(fit))
  sapply(effects, function(effect) {
    estimates$vcov[estimates$grp == if (is.null(effect$term)) "Residual" else effect$term]
  })
}

make_case <- function(name, data, codes, nested, n, effects, studies, reml = TRUE) {
  facets <- names(codes)
  terms <- Filter(function(effect) !is.null(effect$term), effects)
  fit <- aov(as.formula(paste("score ~", paste(sapply(terms, function(effect) effect$term), collapse = " + "))), data)
  table <- summary(fit)[[1]]
  rows <- trimws(rownames(table))
  pick <- function(effect, column) table[[column]][rows == if (is.null(effect$term)) "Residuals" else effect$term]
  effects <- order_effects(effects, codes)
  mean_squares <- sapply(effects, function(effect) pick(effect, "Mean Sq"))
  variances <- solve_components(effects, mean_squares, n)
  by_label <- setNames(variances, sapply(effects, function(effect) effect$label))
  random <- if (reml) reml_components(
    as.formula(paste("score ~ 1 +", paste(sapply(Filter(function(effect) !is.null(effect$term), effects),
                                                  function(effect) paste0("(1 | ", effect$term, ")")), collapse = " + "))),
    data, effects) else NULL
  if (reml && any(variances <= 0)) stop(name, ": a component is not positive, so REML would differ: ", paste(round(variances, 4), collapse = " "))
  names_of <- function(letters) I(names(codes)[sort(match(letters, codes))])
  list(
    name = name,
    design = list(object = OBJECT, facets = I(facets[-1]), nested = nested),
    scores = lapply(seq_len(nrow(data)), function(k) {
      list(levels = setNames(lapply(codes, function(code) as.character(data[[code]][k])), facets), score = data$score[k])
    }),
    components = lapply(seq_along(effects), function(k) {
      effect <- effects[[k]]
      c(list(facets = names_of(effect$own), nestedIn = names_of(effect$nested_in),
             degreesOfFreedom = pick(effect, "Df"), sumOfSquares = pick(effect, "Sum Sq"),
             meanSquare = mean_squares[k], variance = variances[k]),
        if (reml) list(reml = unname(random[k])))
    }),
    studies = lapply(studies, function(study) c(list(options = study$options), study$compute(pmax(by_label, 0))))
  )
}

effect <- function(label, own, nested_in = character(0), term = NULL) {
  list(label = label, own = own, nested_in = nested_in, term = term)
}

options_of <- function(sampleSizes = NULL, fixed = NULL, nested = NULL) {
  options <- list()
  if (!is.null(sampleSizes)) options$sampleSizes <- as.list(sampleSizes)
  if (!is.null(fixed)) options$fixed <- I(fixed)
  if (!is.null(nested)) options$nested <- lapply(nested, I)
  if (length(options) == 0) structure(list(), names = character(0)) else options
}

study <- function(options, compute) list(options = options, compute = compute)

# A decision study's quantities from the variances v named by label.
quantities <- function(universe, relative, absolute) {
  list(universeScoreVariance = universe, relativeErrorVariance = relative, absoluteErrorVariance = absolute,
       generalizability = universe / (universe + relative), dependability = universe / (universe + absolute))
}

no_nesting <- structure(list(), names = character(0))

# --- p x i -----------------------------------------------------------------------------------------------------------
set.seed(SEED)
pxi_grid <- grid_of(i = 1:12, p = 1:40)
pxi <- cbind(pxi_grid, score = simulate_scores(pxi_grid, list(
  list(columns = "p", sd = 1), list(columns = "i", sd = sqrt(0.5))), error_sd = sqrt(2)))

pxi_effects <- list(effect("p", "p", term = "p"), effect("i", "i", term = "i"), effect("pi", c("p", "i")))
d_pxi <- function(n_i) function(v) {
  relative <- v[["pi"]] / n_i
  quantities(v[["p"]], relative, relative + v[["i"]] / n_i)
}

# --- p x i with a negative component --------------------------------------------------------------------------------
# The item effect is 0 and the sample small, so that the estimate of its variance is below 0 for about half of the
# samples; the first such sample is taken.
repeat {
  negative_grid <- grid_of(i = 1:4, p = 1:8)
  negative <- cbind(negative_grid, score = simulate_scores(negative_grid, list(
    list(columns = "p", sd = 1), list(columns = "i", sd = 0)), error_sd = sqrt(2)))
  anova_table <- summary(aov(score ~ p + i, negative))[[1]]
  if ((anova_table[["Mean Sq"]][2] - anova_table[["Mean Sq"]][3]) / 8 < 0) break
}

# --- p x r x i -------------------------------------------------------------------------------------------------------
pri_grid <- grid_of(i = 1:6, r = 1:4, p = 1:40)
pri <- cbind(pri_grid, score = simulate_scores(pri_grid, list(
  list(columns = "p", sd = 1), list(columns = "r", sd = sqrt(0.6)), list(columns = "i", sd = sqrt(0.8)),
  list(columns = c("p", "r"), sd = sqrt(0.5)), list(columns = c("p", "i"), sd = sqrt(0.8)),
  list(columns = c("r", "i"), sd = sqrt(0.5))), error_sd = sqrt(1.5)))

pri_effects <- list(
  effect("p", "p", term = "p"), effect("r", "r", term = "r"), effect("i", "i", term = "i"),
  effect("pr", c("p", "r"), term = "p:r"), effect("pi", c("p", "i"), term = "p:i"),
  effect("ri", c("r", "i"), term = "r:i"), effect("pri", c("p", "r", "i")))
d_pri <- function(n_r, n_i) function(v) {
  relative <- v[["pr"]] / n_r + v[["pi"]] / n_i + v[["pri"]] / (n_r * n_i)
  quantities(v[["p"]], relative, relative + v[["r"]] / n_r + v[["i"]] / n_i + v[["ri"]] / (n_r * n_i))
}
# Items fixed: the person by item effect joins the universe score, and the item effects are not error.
d_pri_fixed_items <- function(n_r, n_i) function(v) {
  relative <- v[["pr"]] / n_r + v[["pri"]] / (n_r * n_i)
  quantities(v[["p"]] + v[["pi"]] / n_i, relative, relative + v[["r"]] / n_r + v[["ri"]] / (n_r * n_i))
}
# Raters nested in persons in the decision study, though crossed in the study: r:p pools r and pr, ri:p pools ri and pri.
d_pri_nested <- function(n_r, n_i) function(v) {
  relative <- (v[["r"]] + v[["pr"]]) / n_r + v[["pi"]] / n_i + (v[["ri"]] + v[["pri"]]) / (n_r * n_i)
  quantities(v[["p"]], relative, relative + v[["i"]] / n_i)
}

# --- (r:p) x i ---------------------------------------------------------------------------------------------------------
# Each of 25 persons has 3 raters of their own, labeled 1 to 3 within the person.
rp_grid <- grid_of(i = 1:4, r = 1:3, p = 1:25)
rp_by_i <- cbind(rp_grid, score = simulate_scores(rp_grid, list(
  list(columns = "p", sd = sqrt(0.9)), list(columns = "i", sd = sqrt(0.5)), list(columns = c("p", "r"), sd = sqrt(0.6)),
  list(columns = c("p", "i"), sd = sqrt(0.7))), error_sd = sqrt(1.2)))

rp_by_i_effects <- list(
  effect("p", "p", term = "p"), effect("i", "i", term = "i"), effect("rp", "r", "p", term = "p:r"),
  effect("pi", c("p", "i"), term = "p:i"), effect("rip", c("r", "i"), "p"))
d_rp_by_i <- function(n_r, n_i) function(v) {
  relative <- v[["rp"]] / n_r + v[["pi"]] / n_i + v[["rip"]] / (n_r * n_i)
  quantities(v[["p"]], relative, relative + v[["i"]] / n_i)
}
d_rp_by_i_fixed_items <- function(n_r, n_i) function(v) {
  relative <- v[["rp"]] / n_r + v[["rip"]] / (n_r * n_i)
  quantities(v[["p"]] + v[["pi"]] / n_i, relative, relative)
}

# --- i:p ------------------------------------------------------------------------------------------------------------
ip_grid <- grid_of(i = 1:6, p = 1:30)
ip <- cbind(ip_grid, score = simulate_scores(ip_grid, list(
  list(columns = "p", sd = sqrt(1.2))), error_sd = sqrt(2.5)))
ip_effects <- list(effect("p", "p", term = "p"), effect("ip", "i", "p"))
d_ip <- function(n_i) function(v) quantities(v[["p"]], v[["ip"]] / n_i, v[["ip"]] / n_i)

# --- r:pi ------------------------------------------------------------------------------------------------------------
# Each person's response to each of 4 items is rated by 2 raters of its own.
rpi_grid <- grid_of(r = 1:2, i = 1:4, p = 1:20)
rpi <- cbind(rpi_grid, score = simulate_scores(rpi_grid, list(
  list(columns = "p", sd = 1), list(columns = "i", sd = sqrt(0.4)), list(columns = c("p", "i"), sd = sqrt(0.6))),
  error_sd = 1))
rpi_effects <- list(
  effect("p", "p", term = "p"), effect("i", "i", term = "i"), effect("pi", c("p", "i"), term = "p:i"),
  effect("rpi", "r", c("p", "i")))
d_rpi <- function(n_r, n_i) function(v) {
  relative <- v[["pi"]] / n_i + v[["rpi"]] / (n_r * n_i)
  quantities(v[["p"]], relative, relative + v[["i"]] / n_i)
}

# --- i:r:p -----------------------------------------------------------------------------------------------------------
# 12 persons, each with 3 raters of their own, each of whom scores 4 items of their own.
irp_grid <- grid_of(i = 1:4, r = 1:3, p = 1:12)
irp <- cbind(irp_grid, score = simulate_scores(irp_grid, list(
  list(columns = "p", sd = sqrt(1.1)), list(columns = c("p", "r"), sd = sqrt(0.7))), error_sd = sqrt(1.4)))
irp_effects <- list(effect("p", "p", term = "p"), effect("rp", "r", "p", term = "p:r"), effect("irp", "i", c("r", "p")))
d_irp <- function(n_r, n_i) function(v) {
  relative <- v[["rp"]] / n_r + v[["irp"]] / (n_r * n_i)
  quantities(v[["p"]], relative, relative)
}

# --- (i:p) x (r:p) ----------------------------------------------------------------------------------------------------
# Each of 20 persons has 3 items and 2 raters of their own; every rater scores every item of the person.
two_grid <- grid_of(r = 1:2, i = 1:3, p = 1:20)
two <- cbind(two_grid, score = simulate_scores(two_grid, list(
  list(columns = "p", sd = 1), list(columns = c("p", "i"), sd = sqrt(1.1)), list(columns = c("p", "r"), sd = sqrt(0.5))),
  error_sd = sqrt(1.3)))
two_effects <- list(
  effect("p", "p", term = "p"), effect("ip", "i", "p", term = "p:i"), effect("rp", "r", "p", term = "p:r"),
  effect("irp", c("i", "r"), "p"))
d_two <- function(n_i, n_r) function(v) {
  relative <- v[["ip"]] / n_i + v[["rp"]] / n_r + v[["irp"]] / (n_i * n_r)
  quantities(v[["p"]], relative, relative)
}

C_PXI <- c(person = "p", item = "i")
C_PRI <- c(person = "p", rater = "r", item = "i")
C_RP_BY_I <- c(person = "p", rater = "r", item = "i")
C_IP <- c(person = "p", item = "i")
C_RPI <- c(person = "p", rater = "r", item = "i")
C_IRP <- c(person = "p", rater = "r", item = "i")
C_TWO <- c(person = "p", item = "i", rater = "r")

cases <- list(
  make_case("p x i", pxi, C_PXI, no_nesting, c(p = 40, i = 12), pxi_effects, list(
    study(options_of(), d_pxi(12)),
    study(options_of(sampleSizes = c(item = 5)), d_pxi(5)),
    study(options_of(sampleSizes = c(item = 40)), d_pxi(40)))),
  make_case("p x i with a negative component", negative, C_PXI, no_nesting, c(p = 8, i = 4), pxi_effects, list(
    study(options_of(), d_pxi(4)),
    study(options_of(sampleSizes = c(item = 10)), d_pxi(10))), reml = FALSE),
  make_case("p x r x i", pri, C_PRI, no_nesting, c(p = 40, r = 4, i = 6), pri_effects, list(
    study(options_of(), d_pri(4, 6)),
    study(options_of(sampleSizes = c(rater = 2, item = 10)), d_pri(2, 10)),
    study(options_of(sampleSizes = c(rater = 1, item = 1)), d_pri(1, 1)),
    study(options_of(sampleSizes = c(rater = 2, item = 8), fixed = "item"), d_pri_fixed_items(2, 8)),
    study(options_of(sampleSizes = c(rater = 2, item = 6), nested = list(rater = "person")), d_pri_nested(2, 6)))),
  make_case("(r:p) x i", rp_by_i, C_RP_BY_I, list(rater = I("person")), c(p = 25, r = 3, i = 4), rp_by_i_effects, list(
    study(options_of(), d_rp_by_i(3, 4)),
    study(options_of(sampleSizes = c(rater = 1, item = 12)), d_rp_by_i(1, 12)),
    study(options_of(sampleSizes = c(rater = 2, item = 4), fixed = "item"), d_rp_by_i_fixed_items(2, 4)))),
  make_case("i:p", ip, C_IP, list(item = I("person")), c(p = 30, i = 6), ip_effects, list(
    study(options_of(), d_ip(6)),
    study(options_of(sampleSizes = c(item = 20)), d_ip(20)))),
  make_case("r:pi", rpi, C_RPI, list(rater = I(c("person", "item"))), c(p = 20, r = 2, i = 4), rpi_effects, list(
    study(options_of(), d_rpi(2, 4)),
    study(options_of(sampleSizes = c(rater = 3, item = 6)), d_rpi(3, 6)))),
  make_case("i:r:p", irp, C_IRP, list(rater = I("person"), item = I("rater")), c(p = 12, r = 3, i = 4), irp_effects,
            list(study(options_of(), d_irp(3, 4)), study(options_of(sampleSizes = c(rater = 5, item = 2)), d_irp(5, 2)))),
  make_case("(i:p) x (r:p)", two, C_TWO, list(item = I("person"), rater = I("person")), c(p = 20, i = 3, r = 2),
            two_effects, list(study(options_of(), d_two(3, 2)), study(options_of(sampleSizes = c(item = 6, rater = 4)),
                                                                        d_two(6, 4))))
)

write_fixture("generalizability", list(
  source = paste("Simulations, set.seed(20261029): p x i; p x i with a negative item component; p x r x i; (r:p) x i with",
                 "rater labels repeated within persons; i:p; r:pi; i:r:p; (i:p) x (r:p). Sums of squares from aov, variance",
                 "components from the expected mean squares solved with solve(), REML estimates from lme4, and decision",
                 "studies from the textbook formulas of each design."),
  cases = cases
))
