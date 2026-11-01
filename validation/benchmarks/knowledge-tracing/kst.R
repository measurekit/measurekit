# Knowledge space theory: the fit of the basic local independence model by EM and by minimum discrepancy maximum
# likelihood, its local identifiability, the validity of a structure and the test of a quasi order, and the inductive
# item tree analysis of a log, against pks's blim and jacobian, kst's kvalidate and DAKS's z_test and iita. The
# responses of 1,000 persons to 12 items are those of a chain of prerequisites, with a guess and a slip of 0.1.

suppressPackageStartupMessages({
  library(pks)
  library(DAKS)
  library(kst)
  library(sets)
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
bench_task("kst.mdml", "Fit of the basic local independence model of a chain of 12 items by minimum discrepancy maximum likelihood",
           sprintf("%s persons, %d items, %d states", count(persons), item_count, nrow(states)), reference("pks", "blim(method = \"MDML\")"),
           function() blim(states, patterns, method = "MDML", tol = 1e-4, maxiter = 1e6),
           note = "pks stops when its parameters change by 1e-4, after 41 iterations; measurekit when the log-likelihood changes by 1e-8, after 331.")
bench_task("kst.identifiability", "Rank of the Jacobian of the pattern probabilities of the model of a chain of 12 items",
           sprintf("%d items, %d states, %d parameters", item_count, nrow(states), nrow(states) - 1 + 2 * item_count),
           reference("pks", "jacobian` and `qr"), function() qr(jacobian(list(K = states)))$rank)

chain <- kstructure(as.set(lapply(seq_len(nrow(states)), function(r) as.set(colnames(states)[states[r, ] == 1]))))
bench_task("kst.validity", "Validity of the chain against the responses: gamma, violational coefficient, solution rates, discrepancy index and distance agreement",
           sprintf("%s persons, %d items, %d states", count(persons), item_count, nrow(states)), reference("kst", "kvalidate"),
           function() for (method in c("gamma", "VC", "percent", "DI", "DA")) kvalidate(chain, responses, method))

# DAKS's variance sums over all 2^items patterns with set operations, and takes 12 seconds at 6 items, so the test is sized there.
test_items <- 6
implications <- do.call(set, unlist(lapply(seq_len(test_items - 1), function(i) lapply((i + 1):test_items, function(j) tuple(i, j))),
                                    recursive = FALSE))
bench_task("kst.diffTest", "Test of the fit of the chain of 6 items to the responses",
           sprintf("%s persons, %d items", count(persons), test_items), reference("DAKS", "z_test(v = 1)"),
           function() z_test(responses[, seq_len(test_items)], implications, v = 1))
