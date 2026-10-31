# Cognitive diagnosis: calibration of the G-DINA model and of the models it contains by EM, classification of persons,
# classification accuracy, validation of the Q-matrix and nonparametric classification, against GDINA's GDINA,
# personparm, CA and Qval and NPCD's AlphaNP. GDINA's convergence criterion is the largest change of a success
# probability or a proportion, as measurekit's tolerance, set to measurekit's default of 1e-4.

suppressPackageStartupMessages({
  library(GDINA)
  library(NPCD)
})

source("validation/benchmarks/simulation.R")

set.seed(20261029)
persons <- 1000
item_count <- 20
attribute_count <- 4
CONTROL <- list(conv.crit = 1e-4, maxitr = 2000)

# --- Data ----------------------------------------------------------------------------------------------------------

simulated <- simulate_dina(persons, item_count, attribute_count)
Q <- simulated$Q
attributes <- simulated$attributes
responses <- simulated$responses

# --- References -----------------------------------------------------------------------------------------------------

calibrate <- function(model) GDINA(responses, Q, model = model, control = CONTROL, verbose = 0)

gdina_fit <- calibrate("GDINA")

bench_fixture("psychometrics", "cdm", list(
  source = "A simulation, set.seed(20261029): DINA responses with a guessing of 0.2 and a slip of 0.1 over 4 correlated attributes.",
  itemIds = ids(item_count),
  responses = unname(responses),
  qMatrix = setNames(lapply(seq_len(item_count), function(j) I(attributes[Q[j, ] == 1])), ids(item_count))
))

size <- sprintf("%s persons, %d items, %d attributes", count(persons), item_count, attribute_count)
models <- c(gdina = "GDINA", dina = "DINA", dino = "DINO", additive = "ACDM", logit = "LLM", reduced = "RRUM")
tasks <- c(gdina = "G-DINA", dina = "DINA", dino = "DINO", additive = "A-CDM", logit = "LLM", reduced = "R-RUM")
keys <- c(gdina = "Gdina", dina = "Dina", dino = "Dino", additive = "Acdm", logit = "Llm", reduced = "Rrum")
for (name in names(models)) {
  local({
    model <- models[[name]]
    bench_task(paste0("cdm.calibrate", keys[[name]]), sprintf("Calibration of the %s model by EM", tasks[[name]]), size,
               reference("GDINA", sprintf("GDINA(model = \"%s\")", model)), function() calibrate(model))
  })
}
bench_task("cdm.attributeProfiles", "Posterior profiles of the persons of a calibration", size,
           reference("GDINA", "personparm(what = \"EAP\")"), function() personparm(gdina_fit, what = "EAP"))
bench_task("cdm.classificationAccuracy", "Accuracy and consistency of the classification", size,
           reference("GDINA", "CA"), function() CA(gdina_fit))
bench_task("cdm.validateQMatrix", "Validation of the Q-matrix by the proportion of variance accounted for", size,
           reference("GDINA", "Qval(method = \"PVAF\")"), function() Qval(gdina_fit, method = "PVAF", eps = 0.95))
bench_task("cdm.nonparametric", "Nonparametric classification by the Hamming distance", size,
           reference("NPCD", "AlphaNP"), function() AlphaNP(responses, Q, gate = "AND", method = "Hamming"))
