# Answer similarity: the pairs of the persons who share a room, and the indices of copying over the ordered pairs of two
# rooms. CopyDetect 1.4 computes each index of one pair at a time, in internal functions that validation/README.md says
# how to install; the pairs of a room are those its predecessor 1.3 forms for `centers`, by expand.grid.

suppressPackageStartupMessages(library(CopyDetect))

set.seed(20261104)
persons <- 1000
item_count <- 40
options <- 5
room_size <- 25
rooms <- rep(seq_len(persons / room_size), each = room_size)

key <- sample(options, item_count, replace = TRUE)
slopes <- matrix(round(rnorm(item_count * options, 0, 0.7), 3), item_count, options)
slopes[cbind(seq_len(item_count), key)] <- apply(slopes, 1, max) + 0.5
intercepts <- matrix(round(rnorm(item_count * options, 0, 0.6), 3), item_count, options)
theta <- rnorm(persons)
responses <- t(vapply(theta, function(t) {
  vapply(seq_len(item_count), function(i) {
    z <- slopes[i, ] * t + intercepts[i, ]
    sample(options, 1, prob = exp(z - max(z)))
  }, 0)
}, numeric(item_count)))
scores <- (responses == matrix(key, persons, item_count, byrow = TRUE)) * 1

center_pairs <- function(r) {
  members <- which(rooms == r)
  grid <- expand.grid(members, members)
  as.matrix(grid[grid[, 1] != grid[, 2], ])
}
compared <- unname(do.call(rbind, lapply(1:2, center_pairs)))

bench_fixture("psychometrics", "similarity", list(
  source = paste(
    "A simulation, set.seed(20261104): 1,000 persons in rooms of 25 choosing among 5 options of 40 items of the",
    "nominal response model, the keyed option the steepest; the ordered pairs of the first two rooms."
  ),
  itemIds = ids(item_count),
  responses = unname(responses - 1),
  items = lapply(seq_len(item_count), function(i) list(a = slopes[i, ], c = intercepts[i, ])),
  thetas = theta,
  key = key - 1,
  rooms = sprintf("r%d", rooms),
  pairs = compared - 1
))

form <- as.data.frame(responses)
form2 <- as.data.frame(scores)
parameters <- cbind(slopes, intercepts)
thetas <- matrix(theta)
# CopyDetect's internal functions take one pair: each task calls one for every pair.
each_pair <- function(index) function() for (k in seq_len(nrow(compared))) index(compared[k, ])
pair_size <- sprintf("%s ordered pairs of 2 rooms of 25, 40 items of 5 options", count(nrow(compared)))

bench_task("similarity.selectPairs", "The pairs of the persons who share a room",
           sprintf("%s persons in rooms of 25, %s pairs", count(persons), count(persons / room_size * choose(room_size, 2))),
           "base R `expand.grid` of each room, as CopyDetect 1.3's `centers`",
           function() lapply(unique(rooms), center_pairs),
           note = "base R forms each pair both ways, and measurekit once")
bench_task("similarity.omegaIndex", "Wollack's omega of each pair, nominal responses", pair_size,
           reference("CopyDetect", "omega2"),
           each_pair(function(pair) CopyDetect:::omega2(form, parameters, pair, 1:options, key, thetas)))
bench_task("similarity.generalizedBinomialTest", "The generalized binomial test of each pair, nominal responses",
           pair_size, reference("CopyDetect", "gbt2"),
           each_pair(function(pair) CopyDetect:::gbt2(form, parameters, pair, 1:options, thetas)))
bench_task("similarity.m4Index", "Maynes' M4 of each pair, nominal responses", pair_size,
           reference("CopyDetect", "M4_2"),
           each_pair(function(pair) CopyDetect:::M4_2(form, parameters, pair, key, 1:options, thetas)))
bench_task("similarity.kIndex", "The K index, K1, K2, S1 and S2 of each pair, nominal responses", pair_size,
           reference("CopyDetect", "k_2 and ks12"),
           each_pair(function(pair) {
             # CopyDetect's wrapper of 1.3 leaves out a source without an incorrect response, where K is undefined.
             if (sum(scores[pair[2], ] == 0) == 0) return(NULL)
             CopyDetect:::k_2(form, form2, pair)
             CopyDetect:::ks12(form, form2, pair, 1:options)
           }))
