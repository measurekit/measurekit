# Guttman errors and conditional association, steps 3 and 5 of a Mokken scale analysis (mokken-steps.json). References:
# mokken's check.errors, G+ and its upper fence U1, and check.ca(Windex = TRUE), the W1, W2 and W3 indices and their
# flags in its first pass over all items. check.errors breaks ties between item steps that as many persons pass by the
# mean over 1,000 random orders, rounded to an integer; the base R write-up below gives the exact mean over the orders,
# the reference where steps tie. check.ca cannot run on three items, as apply() needs a matrix of rest items, so that case
# runs mokken's own compute.W1, compute.W2, compute.W3 and flag on rest scores taken with drop = FALSE.
#
# Data: the cases of mokken-data.R (complete cases), a seeded test of 10 items with a planted dependent pair, the
# transitive reasoning items with missing responses, and three items of that test.

source("validation/psychometrics/common.R")
source("validation/psychometrics/mokken-data.R")

rows_of <- function(X) unname(lapply(seq_len(nrow(X)), function(i) as.numeric(X[i, ])))

# --- Base R: G+ as the exact mean over the orders of tied steps --------------------------------------------------------

# The steps of every item, each passed by a score of at least s, in the order of the items and then the steps; a pair of
# a step passed and a step failed is an error when more persons pass the failed one, and half an error when as many do.
exact_gplus <- function(X) {
  X <- X - min(X)
  top <- max(X)
  steps <- do.call(cbind, lapply(seq_len(ncol(X)), function(j) sapply(seq_len(top), function(s) as.numeric(X[, j] >= s))))
  counts <- colSums(steps)
  before <- outer(counts, counts, ">") + 0.5 * outer(counts, counts, "==")
  rowSums(((1 - steps) %*% before) * steps)
}

# Tukey's upper fence of G+, as check.errors computes U1 from summary().
upper_fence <- function(g) {
  q <- quantile(g, type = 7, names = FALSE)
  q[4] + 1.5 * (q[4] - q[2])
}

# --- Cases ---------------------------------------------------------------------------------------------------------------

errors_case <- function(name, X) {
  complete <- complete.cases(X)
  Y <- X[complete, , drop = FALSE]
  reference <- check.errors(Y)
  exact <- exact_gplus(Y)
  Z <- Y - min(Y)
  counts <- sapply(seq_len(ncol(Z)), function(j) sapply(seq_len(max(Z)), function(s) sum(Z[, j] >= s)))
  ties <- any(duplicated(as.numeric(counts)))
  fence <- upper_fence(exact)
  # Without tied steps check.errors takes the one order of the steps, and both give the same counts.
  if (!ties) stopifnot(all(exact == reference$Gplus), fence == reference$UGplus$U1Gplus)
  gplus <- rep(NA_real_, nrow(X))
  gplus[complete] <- exact
  list(
    name = name, itemIds = colnames(X), responses = rows_of(X), errors = gplus, upperFence = fence,
    outliers = I(which(!is.na(gplus) & gplus > fence) - 1),
    mokken = list(errors = reference$Gplus, upperFence = reference$UGplus$U1Gplus, ties = ties)
  )
}

# check.ca's first pass: the indices and their flags, over all items.
association_case <- function(name, X, minsize = 4) {
  Y <- X[complete.cases(X), , drop = FALSE]
  if (ncol(Y) >= 4) {
    result <- suppressWarnings(check.ca(Y, Windex = TRUE, MINSIZE = minsize))
    index <- result$Index[[1]]
    flags <- result$Flagged[[1]]
    W1 <- index$W1; W2 <- index$W2; W3 <- index$W3
    F1 <- flags$F1; F2 <- flags$F2; F3 <- flags$F3
  } else {
    # mokken's own steps, with the rest scores of a pair from the third item taken as a matrix.
    Z <- mokken:::check.data(Y)
    J <- ncol(Z)
    R <- list()
    for (i in 1:J) R[[i]] <- list()
    for (i in 1:(J - 1)) for (j in (i + 1):J) {
      R[[i]][[j]] <- R[[j]][[i]] <- mokken:::joinRSG(rowSums(Z[, -c(i, j), drop = FALSE]), minsize)
    }
    W1 <- mokken:::compute.W1(Z, mingroup = 4, covweightoption = "pnorm", nweightoption = "noweight")
    W2 <- mokken:::compute.W2(Z, R, mingroup = 4, covweightoption = "pnorm", nweightoption = "noweight")
    W3 <- mokken:::compute.W3(Z, R, mingroup = 4, covweightoption = "pnorm", nweightoption = "noweight")
    F1 <- mokken:::flag(W1); F2 <- mokken:::flag(W2); F3 <- mokken:::flag(W3)
  }
  ids <- colnames(Y)
  pairs <- function(F, upper) {
    hits <- which(F == 1 & !is.na(F), arr.ind = TRUE)
    if (upper) hits <- hits[hits[, 1] < hits[, 2], , drop = FALSE]
    hits <- hits[order(hits[, 1], hits[, 2]), , drop = FALSE]
    lapply(seq_len(nrow(hits)), function(k) ids[hits[k, ]])
  }
  list(
    name = name, itemIds = ids, responses = rows_of(Y), minimumSize = minsize,
    w1 = unname(lapply(seq_len(nrow(W1)), function(i) W1[i, ])), w2 = as.numeric(W2),
    w3 = unname(lapply(seq_len(nrow(W3)), function(i) W3[i, ])),
    flagged = list(w1 = pairs(F1, FALSE), w2 = I(ids[which(as.numeric(F2) == 1)]), w3 = pairs(F3, TRUE))
  )
}

# Ten items of one dimension for 600 persons, items 9 and 10 sharing a second trait of SD 3: positive local dependence,
# which W1 detects.
set.seed(SEED + 3)
dependent <- local({
  n <- 600
  theta <- rnorm(n)
  shared <- rnorm(n, sd = 3)
  difficulty <- seq(-1.5, 1.5, length.out = 10)
  X <- sapply(1:10, function(i) {
    rbinom(n, 1, plogis(1.4 * (theta - difficulty[i]) + if (i >= 9) shared else 0))
  })
  colnames(X) <- paste0("d", 1:10)
  X
})

# The transitive reasoning items with 3% of their responses missing.
set.seed(SEED + 4)
transreas_missing <- local({
  X <- as_items(transreas[, 2:13])
  X[sample(length(X), round(0.03 * length(X)))] <- NA
  X
})

errors_cases <- c(
  lapply(cases, function(case) errors_case(case$name, case$X)),
  list(errors_case("a dependent pair", dependent), errors_case("transitive reasoning, missing", transreas_missing))
)

association_cases <- c(
  lapply(cases, function(case) association_case(case$name, case$X)),
  list(
    association_case("a dependent pair", dependent),
    association_case("a dependent pair, groups of 30", dependent, minsize = 30),
    association_case("three items", as_items(transreas[, 2:4]))
  )
)
planted <- association_cases[[length(cases) + 1]]$flagged
stopifnot(any(sapply(planted$w1, identical, c("d9", "d10"))), "d9" %in% planted$w2)

write_fixture("mokken-steps", list(
  source = paste("mokken 3.1.2: check.errors and check.ca(Windex = TRUE), its first pass; the exact mean of G+ over the",
                 "orders of tied item steps in base R. Data: the cases of mokken.json (the persons who answered every item),",
                 "10 seeded items with a dependent pair for 600 persons (set.seed(20261032)), transreas with 3% of the",
                 "responses missing (set.seed(20261033)), and three of its items."),
  errors = errors_cases,
  association = association_cases
))
