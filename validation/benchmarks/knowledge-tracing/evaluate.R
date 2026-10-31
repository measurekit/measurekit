# The area under the ROC curve of predictions: pROC's auc of roc, with the direction of the predictions given.

suppressPackageStartupMessages(library(pROC))

set.seed(20261029)
count_of_predictions <- 100000
predictions <- runif(count_of_predictions)
outcomes <- rbinom(count_of_predictions, 1, predictions)

bench_fixture("knowledge-tracing", "evaluate", list(
  source = "A simulation, set.seed(20261029): outcomes of calibrated predictions.",
  predictions = I(predictions),
  outcomes = I(outcomes)
))

size <- sprintf("%s predictions", count(count_of_predictions))
bench_task("evaluate.auc", "The area under the ROC curve", size, reference("pROC", "auc(roc(direction = \"<\"))"),
           function() as.numeric(auc(roc(response = outcomes, predictor = predictions, levels = c(0, 1), direction = "<",
                                         quiet = TRUE))))
bench_task("evaluate.metrics", "The AUC, RMSE, log-loss and accuracy of the predictions", size)
