# Answer similarity (similarity-copying.json): CopyDetect 1.4's omega, generalized binomial test, K, K1, K2, S1, S2 and M4 of
# nominal responses and of their 0/1 scores, on a simulation with planted copying, and the pairs of persons who share a
# room as CopyDetect 1.3's `centers` forms them.
#
# CopyDetect 1.4 exists only on GitHub and exports nothing: its indices are the internal functions omega1, gbt1, k_1,
# ks12_1 and M4_1 of 0/1 scores and omega2, gbt2, k_2, ks12 and M4_2 of nominal responses, which take the item
# parameters and the thetas, so that both libraries take the same ones. Install it into a library of its own:
#
#   git clone https://github.com/jaminsore/CopyDetect.git && git -C CopyDetect checkout 32bdd322fba438b654b4ccc1e78406e208a361ad
#   R CMD INSTALL --no-test-load -l <library> CopyDetect
#   R_LIBS=<library> Rscript validation/psychometrics/similarity.R
#
# The simulation has no missing responses: CopyDetect counts two missing responses as identical incorrect ones but leaves
# them out of the source's wrong answers in K, and gives a source's missing nominal response the copier's likeliest
# distractor in omega, where measurekit leaves an item out of a pair when either response is missing. Its tails are
# 1 - pbinom and 1 - ppois, which round a significance below about 1e-16 to 0; the tests allow for it.

source("validation/psychometrics/common.R")
suppressPackageStartupMessages({
  library(mirt)
  library(CopyDetect)
})

set.seed(20261104)
persons <- 450
item_count <- 40
options <- 5
room_size <- 15
rooms <- persons / room_size
# The rooms whose ordered pairs of persons the indices compare, and the copiers who copy a share of their source's
# responses: some in those rooms and some elsewhere, which the indices compare as pairs passed.
compared_rooms <- 1:2
copy_share <- 0.4

# Nominal response items whose keyed option has the steepest slope, at a random position.
key <- sample(options, item_count, replace = TRUE)
slopes <- t(vapply(seq_len(item_count), function(i) {
  a <- sort(rnorm(options, 0, 0.7))
  a[c(setdiff(seq_len(options), key[i]), key[i])] <- a
  a
}, numeric(options)))
intercepts <- matrix(rnorm(item_count * options, 0, 0.6), item_count, options)
theta <- rnorm(persons)
choose <- function(t) {
  vapply(seq_len(item_count), function(i) {
    z <- slopes[i, ] * t + intercepts[i, ]
    sample(options, 1, prob = exp(z - max(z)))
  }, 0)
}
nominal <- t(vapply(theta, choose, numeric(item_count)))
room <- rep(seq_len(rooms), each = room_size)

# Copier and source pairs: the copier takes the source's responses on a random share of the items.
planted <- rbind(c(1, 2), c(5, 9), c(17, 20), c(24, 29), c(101, 107), c(203, 211), c(330, 338), c(441, 449))
for (k in seq_len(nrow(planted))) {
  copied <- sample(item_count, round(copy_share * item_count))
  nominal[planted[k, 1], copied] <- nominal[planted[k, 2], copied]
}
colnames(nominal) <- sprintf("i%d", seq_len(item_count))
scores <- (nominal == matrix(key, persons, item_count, byrow = TRUE)) * 1

# The items and thetas both libraries take: mirt's nominal response model and the 3PL with a prior on the guessing, and
# the posterior modes CopyDetect 1.3 estimates, under a normal prior of variance 10 and 5.
nrm <- mirt(as.data.frame(nominal), 1, itemtype = "nominal", optimizer = "nlminb", verbose = FALSE)
nrm_irt <- coef(nrm, IRTpars = TRUE, simplify = TRUE)$items
nrm_parameters <- matrix(unname(nrm_irt[, c(paste0("a", 1:options), paste0("c", 1:options))]), item_count)
nrm_theta <- fscores(nrm, method = "MAP", cov = matrix(10))
three_pl_model <- mirt.model(sprintf("F = 1-%d\nPRIOR = (1-%d, g, norm, -1.4, 0.5)", item_count, item_count))
three_pl <- mirt(as.data.frame(scores), three_pl_model, itemtype = "3PL", verbose = FALSE)
three_pl_irt <- coef(three_pl, IRTpars = TRUE, simplify = TRUE)$items
three_pl_parameters <- unname(three_pl_irt[, c("a", "b", "g")])
three_pl_theta <- fscores(three_pl, method = "MAP", cov = matrix(5))
stopifnot(extract.mirt(nrm, "converged"), extract.mirt(three_pl, "converged"))

# Every ordered pair of the persons of a room, as CopyDetect 1.3's `centers` forms them: expand.grid without the pairs
# of one person.
room_pairs <- function(r) {
  members <- which(room == r)
  grid <- expand.grid(copier = members, source = members)
  as.matrix(grid[grid$copier != grid$source, ])
}
compared <- rbind(do.call(rbind, lapply(compared_rooms, room_pairs)), planted[planted[, 1] > room_size * 2, ],
                  planted[planted[, 1] > room_size * 2, c(2, 1)])
compared <- unname(compared)

statistic_or_na <- function(x) if (is.null(x) || length(x) == 0) NA_real_ else unname(x)

indices <- function(form, scored, parameters, thetas, nominal_form) {
  lapply(seq_len(nrow(compared)), function(k) {
    pair <- compared[k, ]
    source_wrong <- sum(scored[pair[2], ] == 0)
    if (nominal_form) {
      omega <- CopyDetect:::omega2(form, parameters, pair, 1:options, key, thetas)
      gbt <- CopyDetect:::gbt2(form, parameters, pair, 1:options, thetas)
      m4 <- CopyDetect:::M4_2(form, parameters, pair, key, 1:options, thetas)
      k <- if (source_wrong > 0) CopyDetect:::k_2(form, scored, pair) else NULL
      variants <- if (source_wrong > 0) CopyDetect:::ks12(form, scored, pair, 1:options) else NULL
    } else {
      omega <- CopyDetect:::omega1(form, parameters, pair, thetas)
      gbt <- CopyDetect:::gbt1(form, parameters, pair, thetas)
      m4 <- CopyDetect:::M4_1(form, parameters, pair, thetas)
      k <- if (source_wrong > 0) CopyDetect:::k_1(form, pair) else NULL
      variants <- if (source_wrong > 0) CopyDetect:::ks12_1(form, pair) else NULL
    }
    list(
      omega = omega$W.value, omegaSignificance = omega$p.value, expectedMatches = omega$exp.match,
      generalizedBinomial = gbt$p.value, m4 = m4$p.value,
      k = statistic_or_na(k$k.index), k1 = statistic_or_na(variants$K1.index), k2 = statistic_or_na(variants$K2.index),
      s1 = statistic_or_na(variants$S1.index), s2 = statistic_or_na(variants$S2.index)
    )
  })
}

nominal_indices <- indices(as.data.frame(nominal), as.data.frame(scores), cbind(nrm_parameters), nrm_theta, TRUE)
score_indices <- indices(as.data.frame(scores), as.data.frame(scores), three_pl_parameters, three_pl_theta, FALSE)

write_fixture("similarity-copying", list(
  source = paste(
    "CopyDetect 1.4 (GitHub, 32bdd322) omega2, gbt2, M4_2, k_2 and ks12 of nominal responses and omega1, gbt1, M4_1, k_1",
    "and ks12_1 of their 0/1 scores, at mirt's NRM and 3PL and their MAP thetas, on a simulation, set.seed(20261104),",
    "of 450 persons in 30 rooms of 15 on 40 items of 5 options, 8 of whom copy 40% of a source's responses; the",
    "pairs of the persons of each room as CopyDetect 1.3's centers forms them."
  ),
  personIds = sprintf("p%d", seq_len(persons)),
  itemIds = sprintf("i%d", seq_len(item_count)),
  key = key - 1,
  nominal = list(
    responses = unname(nominal - 1),
    items = lapply(seq_len(item_count), function(i) psychometrics_parameters("nrm", nrm_irt[i, ])),
    thetas = unname(nrm_theta[, 1])
  ),
  scores = list(
    responses = unname(scores),
    items = lapply(seq_len(item_count), function(i) psychometrics_parameters("3pl", c(three_pl_irt[i, ], u = 1))),
    thetas = unname(three_pl_theta[, 1])
  ),
  rooms = sprintf("r%d", room),
  roomPairs = do.call(rbind, lapply(seq_len(rooms), room_pairs)) - 1,
  planted = planted - 1,
  pairs = compared - 1,
  expected = list(nominal = nominal_indices, scores = score_indices)
))
