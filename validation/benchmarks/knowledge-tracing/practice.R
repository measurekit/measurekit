# Choosing the next item has no reference: no package ranks candidates by their predicted success. The task is measured for
# measurekit alone.

bench_task("practice.recommend", "The candidates ranked by their distance to the band of success of a learner",
           "500 candidate items, an Elo model")
bench_task("data.eventLog", "A log built from and checked for events", "100,000 events")
