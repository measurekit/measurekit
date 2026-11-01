# Molenaar-Sijtsma reliability fixtures. Reference: the mokken package's check.reliability (Sijtsma and van der Ark,
# 2017): MS, which assumes double monotonicity, with coefficient alpha beside it. The references run on the complete
# cases of mokken's data sets and of designed cases that exercise each way the method estimates a joint proportion:
# tied item steps, steps that every person or no person passes. A study of 20 seeded data sets from a model of double
# monotonicity compares MS and alpha with the reliability that two parallel forms give.

source("validation/psychometrics/common.R")
source("validation/psychometrics/mokken-data.R")

# check.reliability takes the proportion of persons who pass an item step with mean() of zeros and ones, and mean()
# rounds the sum of the deviations from the average in the order of the persons, so two steps passed by the same number
# of persons can differ in the last bit. MS treats steps with equal proportions as tied, and which steps tie then
# depends on the order of the persons. The reference is the function itself, run where mean() is the sum over the
# count, the same value for the same count; where no tie is split, it equals check.reliability to rounding.
exact_mean <- function(x, na.rm = FALSE) {
  if (na.rm) x <- x[!is.na(x)]
  sum(x) / length(x)
}
exact_check_reliability <- local({
  f <- mokken::check.reliability
  environment(f) <- list2env(list(mean = exact_mean), parent = asNamespace("mokken"))
  f
})

# The steps passed by the same number of persons whose proportions mean() makes differ, as pairs.
split_ties <- function(X) {
  n <- nrow(X)
  counts <- matrix(t(apply(outer(X, 1:(max(X)), ">=") * 1, c(2, 3), sum)))
  proportions <- matrix(t(apply(outer(X, 1:(max(X)), ">=") * 1, c(2, 3), mean)))
  boundary <- counts > 0 & counts < n
  sum(outer(counts[boundary], counts[boundary], "==") & !outer(proportions[boundary], proportions[boundary], "==")) / 2
}

# MS and alpha for the complete cases of X, whose scores are shifted to start at 0 as the package asks, and mokken's
# own MS beside them.
reliability_of <- function(X) {
  X <- as.matrix(X)
  X <- X[complete.cases(X), , drop = FALSE]
  X <- X - min(X)
  options <- list(MS = TRUE, alpha = TRUE, lambda.2 = FALSE, LCRC = FALSE)
  result <- suppressWarnings(do.call(exact_check_reliability, c(list(X), options)))
  mokken_ms <- suppressWarnings(do.call(check.reliability, c(list(X), options)))$MS
  ties <- split_ties(X)
  stopifnot(ties > 0 || abs(result$MS - mokken_ms) < 1e-12)
  list(X = X, ms = result$MS, alpha = result$alpha, mokkenMs = mokken_ms, splitTies = ties)
}

reliability_case <- function(name, X) {
  reference <- reliability_of(X)
  message(name, ": ", nrow(reference$X), " persons, ", ncol(reference$X), " items; MS ", round(reference$ms, 4),
          " (mokken's own ", round(reference$mokkenMs, 4), ", ", reference$splitTies, " tied pairs split), alpha ",
          round(reference$alpha, 4))
  list(name = name, itemIds = I(colnames(X)), responses = unname(X), completeCases = nrow(reference$X),
       ms = reference$ms, alpha = reference$alpha, mokkenMs = reference$mokkenMs, splitTies = reference$splitTies)
}

# Six items of four categories in three pairs whose members have the same distribution, so that every item step
# is tied with the step of its partner; the second of each pair is a shuffle of part of the first.
set.seed(SEED + 1)
tied_polytomous <- local({
  n <- 400
  theta <- rnorm(n)
  first <- sapply(c(-0.8, 0, 0.8), function(location) {
    findInterval(theta + rnorm(n, sd = 0.6), location + c(-0.7, 0, 0.7)) 
  })
  second <- first
  for (pair in 1:3) {
    swapped <- sample(n, n %/% 2)
    second[swapped, pair] <- sample(first[swapped, pair])
  }
  X <- cbind(first, second)[, c(1, 4, 2, 5, 3, 6)]
  colnames(X) <- paste0("p", 1:6)
  X
})

# Four categories, but the steps of the highest category are passed by no person on two items, and every person
# passes the lowest step of one item: those steps leave the computation.
set.seed(SEED + 2)
unused_steps <- local({
  n <- 300
  theta <- rnorm(n)
  X <- sapply(seq(-1, 1, length.out = 6), function(location) {
    findInterval(theta + rnorm(n, sd = 0.7), location + c(-0.9, 0, 0.9))
  })
  X[, 1] <- pmin(X[, 1], 2)
  X[, 2] <- pmin(X[, 2], 2)
  X[, 6] <- pmax(X[, 6], 1)
  colnames(X) <- paste0("u", 1:6)
  X
})

# Items of the graded response model with one slope, so that the item step response functions do not cross: the
# model of double monotonicity. The reliability of the total score is estimated from a second form of the same items
# answered by the same persons, the correlation of the totals of two parallel forms.
study_set <- function(seed, persons, items, categories, slope) {
  set.seed(seed)
  theta <- rnorm(persons)
  locations <- sort(rnorm(items))
  spread <- seq(-1, 1, length.out = categories - 1)
  form <- function() {
    sapply(locations, function(location) {
      latent <- theta + rlogis(persons) / slope
      findInterval(latent, location + spread)
    })
  }
  first <- form()
  second <- form()
  list(X = first, parallel = cor(rowSums(first), rowSums(second)))
}

study <- lapply(seq_len(20), function(index) {
  categories <- if (index <= 10) 2 else 4
  data <- study_set(SEED + 100 + index, 500, 12, categories, 1.1)
  X <- data$X
  colnames(X) <- sprintf("g%02d", 1:12)
  reference <- reliability_of(X)
  list(categories = categories, responses = unname(X), ms = reference$ms, alpha = reference$alpha,
       mokkenMs = reference$mokkenMs, splitTies = reference$splitTies, parallelForms = data$parallel)
})

cat_means <- function(field, subset) mean(sapply(study[subset], function(set) set[[field]]))
message("study: mean MS ", round(cat_means("ms", 1:20), 3), ", parallel forms ", round(cat_means("parallelForms", 1:20), 3),
        ", alpha ", round(cat_means("alpha", 1:20), 3))

write_fixture("mokken-reliability", list(
  source = "mokken's check.reliability(MS = TRUE, alpha = TRUE) on the complete cases of mokken's data sets, designed cases and 20 seeded sets of the graded response model, set.seed(20261029).",
  cases = c(lapply(cases, function(case) reliability_case(case$name, case$X)),
            list(reliability_case("tied polytomous steps", tied_polytomous),
                 reliability_case("unused steps", unused_steps))),
  study = study
))
