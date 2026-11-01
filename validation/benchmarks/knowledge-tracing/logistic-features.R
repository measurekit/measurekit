# The features of LKT that logistic.R does not cover, on the sample of LKT with seeded times: the 16 features that measurekit
# adds, against LKT's computefeatures of each name. The data are those of the validation script, which this script runs.

source("validation/knowledge-tracing/lkt-features.R")

bench_task("logistic.features", "The 16 features of LKT's other names, before each event",
           sprintf("%s events of 17 learners, 16 features", count(rows)), reference("LKT", "computefeatures"),
           function() for (s in specs) compute(s),
           note = "measurekit replays the log once with every feature in one model")
