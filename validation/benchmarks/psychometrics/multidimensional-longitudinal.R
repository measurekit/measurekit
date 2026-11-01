# The calibration of a trait over three occasions by EM, against mirt's mirt with one dimension per occasion, the
# parameters of an item tied across the occasions and the means, variances and covariances of the later occasions free,
# both at 21 nodes per occasion and mirt's default tolerance. The responses of 1,000 persons to 10 items on each occasion
# are simulated with set.seed(20261101).

source("validation/psychometrics/longitudinal-mirt.R")

set.seed(20261101)
persons <- 1000
occasion_count <- 3
per_occasion <- 10
POINTS <- 21

items <- simulate_items(per_occasion)
thetas <- simulate_abilities(persons, c(0, 0.3, 0.6), c(1, 1.1, 1.2), autoregressive(occasion_count, 0.8))
occasions <- simulate_responses(thetas, items, rep(list(seq_len(per_occasion)), occasion_count),
                                rep(list(integer()), occasion_count))

bench_fixture("psychometrics", "multidimensional-longitudinal", list(
  source = "A simulation, set.seed(20261101): 1,000 persons answer the same 10 items on 3 occasions; abilities with means 0, 0.3, 0.6, standard deviations 1, 1.1, 1.2 and an autoregressive correlation of 0.8.",
  points = POINTS,
  occasions = occasion_records(occasions)
))

wide <- wide_data(occasions)
wide_structure <- wide_model(rep(per_occasion, occasion_count))
ties <- constraints(mirt(wide, wide_structure, pars = "values"), occasions, common_slope = FALSE)

bench_task("multidimensional.calibrateLongitudinal", "Calibration of a trait over three occasions by EM",
           sprintf("%s persons, 3 occasions of %d items", count(persons), per_occasion),
           reference("mirt", sprintf("mirt(model, constrain, quadpts = %d)", POINTS)),
           function() mirt(wide, wide_structure, constrain = ties, quadpts = POINTS, verbose = FALSE),
           note = "the occasions are dimensions, and the items of each occasion are columns tied by constrain")
