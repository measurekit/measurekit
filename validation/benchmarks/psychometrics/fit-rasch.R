# The tests of the Rasch model on the Rasch responses of irt-conditional.R, 2,000 persons and 15 items: Andersen's
# likelihood-ratio test of a split at the median score, Martin-Löf's test of a split of the items at their median, and
# the Wald test of each item, against eRm's LRtest, MLoef and Waldtest. Each reference fits the model first, as
# measurekit's tests do. eRm is not attached, as its itemfit and personfit would mask mirt's in fit.R.

rasch <- as.matrix(fromJSON("packages/psychometrics/test/fixtures/bench-irt-conditional.json")$rasch$responses)
size <- sprintf("%s persons, %d items", count(nrow(rasch)), ncol(rasch))

bench_task("fit.andersenTest", "Andersen's likelihood-ratio test of the Rasch model, split at the median score", size,
           reference("eRm", "LRtest(RM)"), function() eRm::LRtest(eRm::RM(rasch)))
bench_task("fit.martinLofTest", "Martin-Löf's test of the Rasch model, the items split at their median", size,
           reference("eRm", "MLoef(RM)"), function() eRm::MLoef(eRm::RM(rasch)))
bench_task("fit.raschWaldTest", "The Wald test of each item of the Rasch model, split at the median score", size,
           reference("eRm", "Waldtest(RM)"), function() eRm::Waldtest(eRm::RM(rasch)))
