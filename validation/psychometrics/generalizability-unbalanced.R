# Unbalanced generalizability fixtures. Reference: Henderson's Method 1, the analogous analysis of variance of Brennan
# (2001, chapters 7 and 8) and the estimator of urGENOVA, written out in base R: for each effect the sum over its cells
# of the squared cell sum over the cell count (T), the coefficients of the variance components in its expectation, from
# the joint table of the cells of two effects, and the linear system solved with solve(). GeneralizIT 0.1.2 checks the
# same components in generalizability-unbalanced.py. gtheory, which wrapped lme4 for this, has been removed from CRAN,
# and lme4 and VCA::anovaVCA are other estimators (REML, and ANOVA with Satterthwaite's approximation), so they differ
# when the data are not balanced. The decision studies are the textbook formulas of Brennan (2001, chapters 3 and 4).

source("validation/psychometrics/common.R")
source("validation/psychometrics/generalizability-simulation.R")
source("validation/psychometrics/generalizability-method-one.R")

SEED <- 20261029
OBJECT <- "person"

# --- Cases -----------------------------------------------------------------------------------------------------------

no_nesting <- structure(list(), names = character(0))

# A case: its design, its scores and the components by Method 1. `studies` are decision studies of the components.
make_case <- function(name, data, facets, nested, studies = list()) {
  stopifnot(facets[1] == OBJECT)
  components <- method_one(data, facets, nested)
  by_name <- setNames(sapply(components, function(component) component$variance),
                      sapply(components, function(component) paste(c(component$facets, component$nestedIn), collapse = "+")))
  list(
    name = name,
    design = list(object = OBJECT, facets = I(facets[-1]), nested = nested, unbalanced = TRUE),
    scores = lapply(seq_len(nrow(data)), function(k) {
      list(levels = setNames(lapply(facets, function(facet) as.character(data[[facet]][k])), facets),
           score = data$score[k])
    }),
    components = components,
    studies = lapply(studies, function(study) c(list(options = study$options), study$compute(pmax(by_name, 0))))
  )
}

study <- function(sizes, compute) list(options = list(sampleSizes = as.list(sizes)), compute = compute)

# A decision study's quantities from the variances v named by their facets.
quantities <- function(universe, relative, absolute) {
  list(universeScoreVariance = universe, relativeErrorVariance = relative, absoluteErrorVariance = absolute,
       generalizability = universe / (universe + relative), dependability = universe / (universe + absolute))
}

# The scores of the cells of the grid that are kept.
without_cells <- function(grid, scores, kept) cbind(grid, score = scores)[kept, ]

# p x r: 300 persons and 6 raters, each score present with probability 0.8.
set.seed(SEED)
grid <- grid_of(rater = paste0("r", 1:6), person = paste0("p", 1:300))
scores <- simulate_scores(grid, list(list(columns = "person", sd = 1), list(columns = "rater", sd = sqrt(0.2))),
                          error_sd = sqrt(0.5))
person_by_rater <- without_cells(grid, scores, runif(nrow(grid)) < 0.8)
stopifnot(all(table(person_by_rater$person) >= 2), all(table(person_by_rater$rater) >= 2))
d_person_by_rater <- function(n) function(v) {
  relative <- v[["person+rater"]] / n
  quantities(v[["person"]], relative, relative + v[["rater"]] / n)
}

# p x r x i: 120 persons, 4 raters and 5 items, 70% of the scores present.
grid <- grid_of(item = paste0("i", 1:5), rater = paste0("r", 1:4), person = paste0("p", 1:120))
scores <- simulate_scores(grid, list(
  list(columns = "person", sd = 1), list(columns = "rater", sd = sqrt(0.2)), list(columns = "item", sd = sqrt(0.3)),
  list(columns = c("person", "rater"), sd = sqrt(0.15)), list(columns = c("person", "item"), sd = sqrt(0.25)),
  list(columns = c("rater", "item"), sd = sqrt(0.1))), error_sd = sqrt(0.5))
person_by_rater_by_item <- without_cells(grid, scores, runif(nrow(grid)) < 0.7)
d_person_by_rater_by_item <- function(n_r, n_i) function(v) {
  relative <- v[["person+rater"]] / n_r + v[["person+item"]] / n_i + v[["person+rater+item"]] / (n_r * n_i)
  quantities(v[["person"]], relative,
             relative + v[["rater"]] / n_r + v[["item"]] / n_i + v[["rater+item"]] / (n_r * n_i))
}

# r:p: 200 persons with 2 to 4 raters of their own, whose labels repeat from one person to the next.
counts <- sample(2:4, 200, replace = TRUE)
rater_in_person <- data.frame(person = rep(paste0("p", 1:200), counts), rater = paste0("r", unlist(lapply(counts, seq_len))))
rater_in_person$score <- simulate_scores(rater_in_person, list(list(columns = "person", sd = 1)), error_sd = sqrt(0.5))
d_rater_in_person <- function(n) function(v) {
  relative <- v[["rater+person"]] / n
  quantities(v[["person"]], relative, relative)
}

# (r:p) x i: 150 persons with 2 or 3 raters of their own, 4 items, 85% of the scores present.
counts <- sample(2:3, 150, replace = TRUE)
raters <- data.frame(person = rep(paste0("p", 1:150), counts), rater = paste0("r", unlist(lapply(counts, seq_len))))
grid <- do.call(rbind, lapply(paste0("i", 1:4), function(item) cbind(raters, item = item)))
scores <- simulate_scores(grid, list(
  list(columns = "person", sd = 1), list(columns = "item", sd = sqrt(0.5)), list(columns = c("person", "rater"), sd = sqrt(0.5)),
  list(columns = c("person", "item"), sd = sqrt(0.4))), error_sd = sqrt(0.6))
rater_in_person_by_item <- without_cells(grid, scores, runif(nrow(grid)) < 0.85)
stopifnot(all(table(rater_in_person_by_item$person) >= 2))

# i:p: 250 persons with 8 to 12 items of their own, whose labels repeat from one person to the next.
counts <- sample(8:12, 250, replace = TRUE)
item_in_person <- data.frame(person = rep(paste0("p", 1:250), counts), item = paste0("i", unlist(lapply(counts, seq_len))))
item_in_person$score <- simulate_scores(item_in_person, list(list(columns = "person", sd = 1)), error_sd = sqrt(0.64))

write_fixture("generalizability-unbalanced", list(
  source = paste("Simulations, set.seed(20261029), with the cells of the first, second and fourth case removed at random: p x r;",
                 "p x r x i; r:p with 2 to 4 raters per person; (r:p) x i; i:p with 8 to 12 items per person. Components by",
                 "Henderson's Method 1 written out in base R; decision studies from the textbook formulas."),
  cases = list(
    make_case("p x r", person_by_rater, c("person", "rater"), no_nesting, list(
      study(c(rater = 3), d_person_by_rater(3)))),
    make_case("p x r x i", person_by_rater_by_item, c("person", "rater", "item"), no_nesting, list(
      study(c(rater = 3, item = 5), d_person_by_rater_by_item(3, 5)))),
    make_case("r:p", rater_in_person, c("person", "rater"), list(rater = I("person")), list(
      study(c(rater = 2), d_rater_in_person(2)))),
    make_case("(r:p) x i", rater_in_person_by_item, c("person", "rater", "item"), list(rater = I("person"))),
    make_case("i:p", item_in_person, c("person", "item"), list(item = I("person")))
  )
))
