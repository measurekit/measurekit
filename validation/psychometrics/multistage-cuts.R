# Routing cuts of multistage panels. No package among the references derives them (mstR is not among them), so the script
# writes the definitions out in base R over mirt's item information (iteminfo) and expected scores (expected.item).
#   Maximum information: the cut between two adjacent modules of a stage is the root of the difference of their
#     information curves between their peaks, by uniroot at a tolerance far below the one that is compared; the peaks are
#     the largest values on a grid of 0.01 over [-6, 6].
#   Routing shares: the theta where the cumulative probability of the population reaches the cumulative share. For the
#     standard normal that is qnorm; for a distribution of nodes with weights, whose weights spread over the cells between
#     the midpoints of its nodes, it is the root of the piecewise linear distribution function through the cell edges.
#   Number-correct: ceiling of the expected score at the cut, less 1e-9, through the modules that routing by theta takes at
#     the cut and then the routing module.

source("validation/psychometrics/common.R")
suppressPackageStartupMessages(library(mirt))

set.seed(20261029)

# --- Items -------------------------------------------------------------------------------------------------------------

coefficients <- data.frame()
models <- character()
modules <- list()

# Adds the items of a module, whose difficulties are spread evenly from `location - spread` to `location + spread` with a
# little noise, with slopes from `slopes`, and returns the module's item numbers. Items cycle through the models given.
module <- function(id, location, spread, count, slopes = c(0.9, 1.6), kinds = c("2PL", "3PL", "graded", "gpcm")) {
  numbers <- integer()
  offsets <- if (count == 1) 0 else seq(-spread, spread, length.out = count)
  for (k in seq_len(count)) {
    kind <- kinds[(k - 1) %% length(kinds) + 1]
    a <- round(runif(1, slopes[1], slopes[2]), 2)
    b <- round(location + offsets[k] + rnorm(1, 0, 0.05), 2)
    steps <- round(b + c(-0.5, 0, 0.5) + rnorm(3, 0, 0.05), 2)
    row <- data.frame(a1 = a, d = NA, g = NA, u = NA, d1 = NA, d2 = NA, d3 = NA)
    if (kind %in% c("2PL", "3PL")) row$d <- -a * b
    if (kind == "3PL") row$g <- round(runif(1, 0.05, 0.25), 2)
    # mirt's graded intercepts are -a b_k, and its gpcm intercepts are cumulative: -a (b_1 + ... + b_k).
    if (kind == "graded") row[c("d1", "d2", "d3")] <- -a * steps
    if (kind == "gpcm") row[c("d1", "d2", "d3")] <- -a * cumsum(steps)
    coefficients <<- rbind(coefficients, row)
    models <<- c(models, kind)
    numbers <- c(numbers, nrow(coefficients))
  }
  list(id = id, items = numbers)
}

# Panels: stages of modules, easiest first in a stage.
panels <- list(
  `1-3-3` = list(
    list(module("routing", 0, 0.8, 6)),
    list(module("easy2", -1.1, 0.35, 5), module("medium2", 0, 0.35, 5), module("hard2", 1.1, 0.35, 5)),
    list(module("easy3", -1.6, 0.35, 4), module("medium3", 0, 0.35, 4), module("hard3", 1.6, 0.35, 4))
  ),
  `1-2-3` = list(
    list(module("routing", 0, 0.8, 6)),
    list(module("low2", -0.7, 0.35, 5), module("high2", 0.7, 0.35, 5)),
    list(module("easy3", -1.4, 0.35, 4), module("medium3", 0, 0.35, 4), module("hard3", 1.4, 0.35, 4))
  ),
  `1-3-3-3` = list(
    list(module("routing", 0, 0.8, 6)),
    list(module("easy2", -1, 0.3, 4), module("medium2", 0, 0.3, 4), module("hard2", 1, 0.3, 4)),
    list(module("easy3", -1.2, 0.3, 4), module("medium3", 0, 0.3, 4), module("hard3", 1.2, 0.3, 4)),
    list(module("easy4", -1.5, 0.3, 4), module("medium4", 0, 0.3, 4), module("hard4", 1.5, 0.3, 4))
  ),
  # A sharp easy module and a flat hard one: their information curves cross also in the left tail, where the flatter one
  # decays more slowly.
  tails = list(
    list(module("routing", 0, 0.6, 4, kinds = "2PL")),
    list(module("sharp", -0.9, 0.2, 4, slopes = c(2, 2.4), kinds = "2PL"),
         module("flat", 1, 0.2, 6, slopes = c(0.6, 0.8), kinds = "2PL"))
  )
)

# The data give mirt the number of categories of each item; the items themselves are set.
persons <- 300
categories <- ifelse(models %in% c("2PL", "3PL"), 2, 4)
ids <- sprintf("item%02d", seq_along(models))
data <- sapply(categories, function(k) sample(0:(k - 1), persons, replace = TRUE))
colnames(data) <- ids
fit <- set_items(data, models, coefficients)
parameters <- coef(fit, IRTpars = TRUE, simplify = TRUE)$items
items <- lapply(seq_len(nrow(coefficients)), extract.item, x = fit)
codes <- c("2PL" = "2pl", "3PL" = "3pl", graded = "grm", gpcm = "gpcm")

information <- function(numbers, theta) Reduce(`+`, lapply(items[numbers], iteminfo, Theta = matrix(theta)))
expected <- function(numbers, theta) Reduce(`+`, lapply(items[numbers], expected.item, Theta = matrix(theta)))

# --- Maximum information -----------------------------------------------------------------------------------------------

GRID <- seq(-6, 6, by = 0.01)

# The root of the difference of the information of two adjacent modules between their peaks, and every sign change of the
# difference on the grid, from which the case checks that there is exactly one between the peaks.
crossing <- function(lower, upper) {
  difference <- function(theta) information(upper$items, theta) - information(lower$items, theta)
  peaks <- c(GRID[which.max(information(lower$items, GRID))], GRID[which.max(information(upper$items, GRID))])
  stopifnot(peaks[1] < peaks[2], difference(peaks[1]) < 0, difference(peaks[2]) > 0)
  between <- GRID >= peaks[1] & GRID <= peaks[2]
  stopifnot(sum(diff(sign(difference(GRID[between]))) != 0) == 1)
  root <- uniroot(difference, peaks, tol = 1e-13)$root
  everywhere <- which(diff(sign(difference(GRID))) != 0)
  list(root = root, crossings = sapply(everywhere, function(i) uniroot(difference, GRID[c(i, i + 1)], tol = 1e-13)$root))
}

# The cuts into each stage; the first stage has none.
information_cuts <- function(stages) {
  lapply(seq_along(stages), function(t) {
    modules <- stages[[t]]
    if (length(modules) == 1) return(numeric())
    sapply(seq_len(length(modules) - 1), function(j) crossing(modules[[j]], modules[[j + 1]])$root)
  })
}

# --- Routing shares ----------------------------------------------------------------------------------------------------

# The cuts of one stage for the shares of its modules, from the distribution function `quantile` of a population.
share_cuts <- function(quantile, shares) {
  quantile(cumsum(shares)[-length(shares)] / sum(shares))
}

normal_quantile <- function(p) qnorm(p)

# A population of nodes with weights: the weights spread uniformly over the cells between the midpoints of the nodes, the
# first and the last cell as wide as the one beside it.
distribution_quantile <- function(nodes, weights) {
  edges <- c(nodes[1] - (nodes[2] - nodes[1]) / 2, (nodes[-1] + nodes[-length(nodes)]) / 2,
             nodes[length(nodes)] + (nodes[length(nodes)] - nodes[length(nodes) - 1]) / 2)
  distribution <- approxfun(edges, c(0, cumsum(weights)))
  function(p) sapply(p, function(target) uniroot(function(theta) distribution(theta) - target, range(edges), tol = 1e-13)$root)
}

# --- Routing modules ---------------------------------------------------------------------------------------------------

# The cuts of every routing module, those of all stages but the last in order, from the cuts into each stage by theta.
panel_cuts <- function(stages, theta_cuts, scale) {
  unlist(lapply(seq_len(length(stages) - 1), function(s) {
    lapply(stages[[s]], function(module) {
      cuts <- theta_cuts[[s + 1]]
      if (scale == "theta") return(cuts)
      sapply(cuts, function(theta) {
        route <- unlist(lapply(seq_len(s - 1), function(t) stages[[t]][[findInterval(theta, theta_cuts[[t]]) + 1]]$items))
        score <- expected(c(route, module$items), theta)
        stopifnot(abs(score - round(score)) > 1e-7)
        max(0, ceiling(score - 1e-9))
      })
    })
  }), recursive = FALSE)
}

# One case of a panel: its options as the function takes them, the description, and the cuts of every routing module.
case <- function(description, options, stages, theta_cuts) {
  list(description = description, options = options,
       cuts = lapply(panel_cuts(stages, theta_cuts, if (is.null(options$scale)) "theta" else options$scale), I))
}

# A population by its density: nodes at a spacing, and the weights proportional to the density.
mixture_density <- function(theta) 0.7 * dnorm(theta, -0.3, 0.9) + 0.3 * dnorm(theta, 1, 0.6)
fine <- seq(-5, 5, by = 0.02)
fine_population <- list(nodes = I(fine), weights = I(mixture_density(fine) / sum(mixture_density(fine))))
coarse_nodes <- c(-2.5, -1.4, -0.6, 0, 0.3, 1.1, 2.4)
coarse_weights <- c(0.05, 0.15, 0.25, 0.2, 0.15, 0.12, 0.08)
coarse_population <- list(nodes = I(coarse_nodes), weights = I(coarse_weights))

panel_cases <- function(name, stages) {
  widths <- sapply(stages, length)
  stage_shares <- lapply(widths[-1], function(k) if (k == 2) c(0.4, 0.6) else c(0.25, 0.45, 0.3))
  equal <- lapply(widths[-1], function(k) rep(1 / k, k))
  by_shares <- function(quantile, shares) c(list(numeric()), lapply(shares, function(s) share_cuts(quantile, s)))
  maximum <- information_cuts(stages)
  list(
    case("maximum information on theta", list(scale = "theta"), stages, maximum),
    case("maximum information on number-correct", list(scale = "number-correct"), stages, maximum),
    case("equal shares of the standard normal", list(method = "routing-shares"), stages,
         by_shares(normal_quantile, equal)),
    case("shares of the standard normal on number-correct",
         list(method = "routing-shares", shares = stage_shares, scale = "number-correct"), stages,
         by_shares(normal_quantile, stage_shares)),
    case("shares of a fine population",
         list(method = "routing-shares", shares = stage_shares, population = fine_population), stages,
         by_shares(distribution_quantile(fine, fine_population$weights), stage_shares)),
    case("shares of a coarse population on number-correct",
         list(method = "routing-shares", shares = stage_shares, population = coarse_population, scale = "number-correct"),
         stages, by_shares(distribution_quantile(coarse_nodes, coarse_weights), stage_shares))
  )
}

named_stages <- function(stages) lapply(stages, function(modules) lapply(modules, function(m) {
  list(id = m$id, items = I(ids[m$items]))
}))

fixture_panels <- lapply(names(panels)[names(panels) != "tails"], function(name) {
  list(name = name, stages = named_stages(panels[[name]]), cases = panel_cases(name, panels[[name]]))
})

# The tails panel: the cut between the two modules is the crossing between their peaks, and the curves cross once more.
tails <- crossing(panels$tails[[2]][[1]], panels$tails[[2]][[2]])
stopifnot(length(tails$crossings) == 2)
tails_panel <- list(
  name = "tails",
  stages = named_stages(panels$tails),
  cases = list(case("maximum information with a crossing in the tail", setNames(list(), character()), panels$tails,
                    list(numeric(), tails$root)))
)

# How closely the cells of a fine population follow the density they discretize: the cuts of the shares 0.2, 0.5 and 0.3
# against the quantiles of the mixture itself, by the root of its distribution function.
mixture_quantile <- function(p) {
  sapply(p, function(target) uniroot(function(theta) 0.7 * pnorm(theta, -0.3, 0.9) + 0.3 * pnorm(theta, 1, 0.6) - target,
                                     c(-8, 8), tol = 1e-13)$root)
}
discretized <- list(
  population = fine_population,
  shares = I(c(0.2, 0.5, 0.3)),
  density = I(mixture_quantile(cumsum(c(0.2, 0.5, 0.3))[-3]))
)

write_fixture("cat-multistage-cuts", list(
  source = paste(
    "Simulated panels (set.seed(20261029)) of 2PL, 3PL, graded and gpcm items set in mirt, whose iteminfo and expected.item",
    "give the information curves and the test characteristic curves; roots by uniroot at a tolerance of 1e-13."
  ),
  items = lapply(seq_along(ids), function(i) {
    list(id = ids[i], model = codes[[models[i]]],
         parameters = psychometrics_parameters(codes[[models[i]]], parameters[i, !is.na(parameters[i, ])]))
  }),
  panels = c(fixture_panels, list(tails_panel)),
  tailCrossings = I(tails$crossings),
  discretized = discretized
))
