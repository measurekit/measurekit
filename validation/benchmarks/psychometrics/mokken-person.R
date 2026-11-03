# The nonparametric person fit of the 2PL responses of irt.R, 2,000 persons and 40 items: the normed Guttman errors, U3
# and Ht of every person, against PerFit's Gnormed, U3 and Ht, one call each. PerFit is not attached, as it attaches
# ltm and its dependencies.

binary <- as.matrix(fromJSON("packages/psychometrics/test/fixtures/bench-irt.json")$binary$responses)

bench_task("mokken.personScalability", "The normed Guttman errors, U3 and Ht of every person",
           sprintf("%s persons, %d dichotomous items", count(nrow(binary)), ncol(binary)),
           reference("PerFit", "Gnormed, U3, Ht"), function() {
             PerFit::Gnormed(binary)
             PerFit::U3(binary)
             PerFit::Ht(binary)
           })
