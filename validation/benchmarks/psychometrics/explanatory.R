# Explanatory models of item difficulty. The regression of item difficulties on item features, with their standard errors,
# by REML: metafor's rma, as the validation does; measurekit's regression is the two-stage method, with the difficulties of
# calibrated items as the outcome. The linear logistic test model (LLTM), which estimates the effects of the features
# from the responses: mirt's Rasch model with an item design, and standard errors by Oakes' method.

suppressPackageStartupMessages({
  library(metafor)
  library(mirt)
})

set.seed(20261029)
items <- 300
features <- data.frame(f1 = rnorm(items), f2 = rnorm(items), f3 = rbinom(items, 1, 0.4))
difficulty <- 0.5 * features$f1 - 0.3 * features$f2 + 0.2 * features$f3 + rnorm(items, 0, 0.4)
standard_error <- runif(items, 0.05, 0.2)
estimate <- rnorm(items, difficulty, standard_error)

lltm_items <- 20
lltm_persons <- 1000
lltm_features <- data.frame(steps = round(runif(lltm_items, 1, 6)), density = round(rnorm(lltm_items), 2),
                            open = rbinom(lltm_items, 1, 0.4), row.names = ids(lltm_items))
lltm_difficulty <- drop(as.matrix(cbind(1, lltm_features)) %*% c(-0.5, 0.35, 0.3, 0.5))
lltm_responses <- matrix(rbinom(lltm_persons * lltm_items, 1, plogis(outer(rnorm(lltm_persons, 0, 1.2), lltm_difficulty, "-"))),
                         lltm_persons, lltm_items, dimnames = list(NULL, ids(lltm_items)))

bench_fixture("psychometrics", "explanatory", list(
  source = "A simulation, set.seed(20261029): item difficulties from three features, estimated with their standard errors.",
  itemIds = ids(items),
  difficulties = I(estimate),
  standardErrors = I(standard_error),
  features = unname(as.matrix(features)),
  lltm = list(itemIds = ids(lltm_items), responses = unname(lltm_responses),
              features = setNames(lapply(seq_len(lltm_items), function(i) as.list(lltm_features[i, ])), ids(lltm_items)))
))

bench_task("explanatory.itemFeatureRegression", "Regression of item difficulties on three features by REML",
           sprintf("%d items, 3 features", items), reference("metafor", "rma(method = \"REML\")"),
           function() rma(yi = estimate, sei = standard_error, mods = ~ f1 + f2 + f3, data = features, method = "REML"))

bench_task("explanatory.lltm", "The linear logistic test model: the effects of three features on difficulty, by EM",
           sprintf("%s persons, %d items, 3 features", count(lltm_persons), lltm_items),
           reference("mirt", "mirt(itemtype = \"Rasch\", itemdesign), Oakes standard errors"),
           function() mirt(as.data.frame(lltm_responses), 1, itemtype = "Rasch", item.formula = ~ 1 + steps + density + open,
                           itemdesign = lltm_features, SE = TRUE, SE.type = "Oakes", TOL = 1e-4, quadpts = 61,
                           verbose = FALSE))
