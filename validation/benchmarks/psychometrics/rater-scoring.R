# The evaluation of automated scores by automatedScoreAgreement, on the scores of rater-automated-scoring.json from
# ELLIPSE and PERSUADE 2.0. RSMTool is timed by validation/benchmarks/automated-scoring.py; this script names its tasks.

scoring <- fromJSON("packages/psychometrics/test/fixtures/rater-automated-scoring.json")$cases

bench_task("rater.automatedScoring",
           "Machine-human and human-human metrics, their degradation and the disattenuated correlation",
           sprintf("%s essays, two human raters", count(length(scoring$ELLIPSE$first))),
           "RSMTool 12.0.0 `compute_degradation_and_disattenuated_correlations`")
bench_task("rater.automatedScoringGroups", "Machine-human metrics of all essays and of each group, the SMD on the population's SDs",
           sprintf("%s essays in %d groups", count(length(scoring$`PERSUADE 2.0`$human)),
                   length(unique(scoring$`PERSUADE 2.0`$groups))),
           "RSMTool 12.0.0 `metrics_helper`")

# The routing curve has no reference. The isotonic mapping is R's isoreg on the same essays, those of one machine score
# sorted by their human score from the highest, so that each tie pools, as validation/psychometrics/rater-scoring.R does.
bench_task("rater.routingCurve", "The coverage and agreement of the machine at each confidence of the scorer",
           sprintf("%s essays, 89 confidences", count(length(scoring$ELLIPSE$first))))
bench_task("rater.scoreMappingIsotonic", "The isotonic mapping from machine to human scores",
           sprintf("%s essays", count(length(scoring$ELLIPSE$first))), reference("stats", "isoreg"),
           function() {
             ordered <- order(scoring$ELLIPSE$machine, -scoring$ELLIPSE$first)
             isoreg(scoring$ELLIPSE$machine[ordered], scoring$ELLIPSE$first[ordered])
           }, note = "the essays are ordered by machine score and, within one, by human score from the highest")
