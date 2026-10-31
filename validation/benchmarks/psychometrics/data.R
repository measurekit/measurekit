# Scoring of multiple-choice responses against a key, as base R's comparison of a column with its key.

set.seed(20261029)
persons <- 2000
item_count <- 50
options <- c("A", "B", "C", "D")

key <- sample(options, item_count, replace = TRUE)
choices <- matrix(sample(options, persons * item_count, replace = TRUE, prob = c(0.2, 0.4, 0.2, 0.2)), persons, item_count)

bench_fixture("psychometrics", "data", list(
  source = "A simulation, set.seed(20261029): choices among four options.",
  key = I(key),
  choices = choices
))

bench_task("data.scoreResponses", "Scoring of the choices of persons against a key",
           sprintf("%s persons, %d items of 4 options", count(persons), item_count), reference("base", "t(t(choices) == key)"),
           function() t(t(choices) == key) * 1L)
