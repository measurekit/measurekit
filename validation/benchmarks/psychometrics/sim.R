# Simulation of 2PL responses: abilities from the standard normal distribution and a response for each of 40 items, against
# mirt's simdata. The items are those of simulate_2pl, so that measurekit simulates the items of the irt benchmarks.

suppressPackageStartupMessages(library(mirt))

source("validation/benchmarks/simulation.R")

set.seed(20261029)
persons <- 2000
item_count <- 40
items <- simulate_2pl(persons, item_count)

bench_task("sim.simulateResponses", "Simulation of the abilities of persons and their responses to 2PL items",
           sprintf("%s persons, %d items", count(persons), item_count), reference("mirt", "simdata(itemtype = \"dich\")"),
           function() simdata(a = items$a, d = -items$a * items$b, N = persons, itemtype = "dich"))
