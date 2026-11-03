# The evaluation of automated scores by the recipe of rater.md, on the scores of rater-automated-scoring.json from
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
