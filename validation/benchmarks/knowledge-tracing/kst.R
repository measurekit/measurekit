# Knowledge space theory: the fit of the basic local independence model by EM and the inductive item tree analysis of a
# log, against pks's blim and DAKS's iita. The responses of 1,000 persons to 12 items are those of a chain of
# prerequisites, with a guess and a slip of 0.1.

suppressPackageStartupMessages({
  library(pks)
  library(DAKS)
})

set.seed(20261029)
persons <- 1000
item_count <- 12

position <- sample(0:item_count, persons, replace = TRUE)
known <- outer(position, seq_len(item_count), ">=")
responses <- matrix(as.integer(ifelse(known, runif(persons * item_count) > 0.1, runif(persons * item_count) < 0.1)),
                    persons, item_count, dimnames = list(NULL, ids(item_count)))

bench_fixture("knowledge-tracing", "kst", list(
  source = "A simulation, set.seed(20261029): a chain of 12 items, guess and slip of 0.1.",
  itemIds = ids(item_count),
  responses = unname(responses)
))

states <- outer(0:item_count, seq_len(item_count), ">=") * 1L
colnames(states) <- ids(item_count)
patterns <- table(apply(responses, 1, paste, collapse = ""))
patterns <- setNames(as.integer(patterns), names(patterns))

bench_task("kst.blim", "Fit of the basic local independence model of a chain of 12 items by EM",
           sprintf("%s persons, %d items, %d states", count(persons), item_count, nrow(states)), reference("pks", "blim"),
           function() blim(states, patterns, method = "ML", tol = 1e-4, maxiter = 1e6))
bench_task("kst.iita", "Inductive item tree analysis of the responses",
           sprintf("%s persons, %d items", count(persons), item_count), reference("DAKS", "iita(v = 1)"),
           function() iita(responses, v = 1))
