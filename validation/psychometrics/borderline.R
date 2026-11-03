# Borderline fixtures (standard-setting-borderline.json). No package implements the borderline methods of OSCE standard
# setting, so the script writes each out in base R. Borderline regression (Kramer et al., 2003; Pell et al., 2010): per
# station, lm of the checklist marks on the global ratings, the cut and its standard error from predict(se.fit = TRUE)
# at the borderline grade, R² from summary(); the total cut is the sum of the stations' cuts, and its standard error
# the root of the sum of the covariances of the stations' cuts, each cut a linear function of its station's marks and
# the stations' residuals correlated within persons. The borderline group (Livingston and Zieky, 1982): per station, the
# mean or the median of the marks of the persons at a borderline grade, the mean with its standard error sd / sqrt(n).

source("validation/psychometrics/common.R")

set.seed(20261104)

rows <- function(m) unname(lapply(seq_len(nrow(m)), function(i) m[i, ]))

# Each station's line, cut, standard error and R²; NA where the ratings do not vary, and the standard error NA with
# fewer than 3 persons, as lm leaves no residual degrees of freedom.
station_regression <- function(marks, ratings, borderline) {
  if (var(ratings) == 0) {
    return(list(cutScore = NA, standardError = NA, rSquared = NA, intercept = NA, slope = NA))
  }
  data <- data.frame(mark = marks, rating = ratings)
  fit <- lm(mark ~ rating, data = data)
  prediction <- predict(fit, data.frame(rating = borderline), se.fit = TRUE)
  r_squared <- if (var(marks) == 0) NA else summary(fit)$r.squared
  list(
    cutScore = unname(prediction$fit),
    standardError = if (nrow(data) < 3) NA else unname(prediction$se.fit),
    rSquared = r_squared,
    intercept = unname(coef(fit)[1]),
    slope = unname(coef(fit)[2])
  )
}

# The total cut and its standard error: the cut of station s is sum_i w_si y_si with
# w_si = 1/n + (x0 - mean(x_s)) (x_si - mean(x_s)) / sum((x_s - mean(x_s))^2), and the covariance of two stations' cuts
# is sigma_st sum_i w_si w_ti, sigma_st estimated by the residuals' cross products over n - 2.
regression_case <- function(name, marks, ratings, borderline) {
  stations <- lapply(seq_len(ncol(marks)), function(s) station_regression(marks[, s], ratings[, s], borderline))
  cuts <- sapply(stations, `[[`, "cutScore")
  n <- nrow(marks)
  total_se <- NA
  if (!anyNA(cuts) && n >= 3) {
    residuals <- sapply(seq_len(ncol(marks)), function(s) residuals(lm(marks[, s] ~ ratings[, s])))
    weights <- sapply(seq_len(ncol(marks)), function(s) {
      x <- ratings[, s]
      1 / n + (borderline - mean(x)) * (x - mean(x)) / sum((x - mean(x))^2)
    })
    covariance <- crossprod(residuals) / (n - 2) * crossprod(weights)
    stopifnot(isTRUE(all.equal(sqrt(diag(covariance)), sapply(stations, `[[`, "standardError"), tolerance = 1e-12)))
    total_se <- sqrt(sum(covariance))
  }
  list(name = name, marks = rows(marks), ratings = rows(ratings), borderline = borderline, stations = stations,
       cutScore = if (anyNA(cuts)) NA else sum(cuts), standardError = total_se)
}

# Each station's cut from the persons at a borderline grade: the mean with sd / sqrt(n), or the median.
group_case <- function(name, marks, ratings, borderline, statistic) {
  stations <- lapply(seq_len(ncol(marks)), function(s) {
    group <- marks[ratings[, s] %in% borderline, s]
    count <- length(group)
    cut <- if (count == 0) NA else if (statistic == "mean") mean(group) else median(group)
    se <- if (statistic == "mean" && count >= 2) sd(group) / sqrt(count) else NA
    list(cutScore = cut, standardError = se, persons = count)
  })
  cuts <- sapply(stations, `[[`, "cutScore")
  list(name = name, marks = rows(marks), ratings = rows(ratings), borderline = I(borderline), statistic = statistic,
       stations = stations, cutScore = if (anyNA(cuts)) NA else sum(cuts))
}

# Six stations of different checklist maxima and a global rating of 1 fail, 2 borderline, 3 pass, 4 superior, from one
# ability per person with station effects, as a national OSCE of six stations; 60 persons.
persons <- 60
ability <- rnorm(persons)
maxima <- c(30, 24, 36, 20, 28, 32)
marks <- sapply(seq_along(maxima), function(s) {
  round(maxima[s] * plogis(0.3 + ability + rnorm(1, 0, 0.3) + rnorm(persons, 0, 0.6)))
})
ratings <- sapply(seq_along(maxima), function(s) {
  findInterval(ability + rnorm(persons, 0, 0.7), c(-1, -0.2, 1.1)) + 1
})

# Eight stations scored in percent, with one decimal, and a six-point rating: 1 clear fail, 2 borderline unsatisfactory,
# 3 borderline satisfactory, 4 pass, 5 good, 6 excellent; the modified borderline group takes grades 2 and 3, and the
# regression reads the line at 2.5; 40 persons.
small <- 40
skill <- rnorm(small)
percent <- sapply(1:8, function(s) round(100 * plogis(0.5 + 0.9 * skill + rnorm(small, 0, 0.5)), 1))
six_point <- sapply(1:8, function(s) findInterval(skill + rnorm(small, 0, 0.6), c(-1.4, -0.6, 0, 0.7, 1.4)) + 1)

# Edges: a station whose ratings all agree has no line, so neither has the total; a station whose marks all agree has no
# R²; a station without a borderline person has no group cut.
edge_marks <- cbind(c(10, 12, 15, 18, 20), c(11, 13, 14, 16, 19), c(15, 15, 15, 15, 15))
edge_ratings <- cbind(c(1, 2, 2, 3, 4), c(2, 2, 2, 2, 2), c(1, 1, 2, 3, 3))
group_edge_ratings <- cbind(c(1, 3, 3, 3, 4), c(2, 2, 2, 2, 2), c(1, 1, 2, 3, 3))

regression <- list(
  regression_case("six stations, ratings 1 to 4", marks, ratings, 2),
  regression_case("eight stations in percent, read at 2.5", percent, six_point, 2.5),
  regression_case("a station without a line", edge_marks, edge_ratings, 2),
  regression_case("constant marks", edge_marks[, c(1, 3)], edge_ratings[, c(1, 3)], 2),
  regression_case("two persons", marks[1:2, 1:2, drop = FALSE], cbind(c(1, 3), c(2, 4)), 2)
)
names(regression) <- sapply(regression, `[[`, "name")

group <- list(
  group_case("six stations, the mean", marks, ratings, 2, "mean"),
  group_case("six stations, the median", marks, ratings, 2, "median"),
  group_case("the modified borderline group, grades 2 and 3", percent, six_point, c(2, 3), "mean"),
  group_case("a station without a borderline person", edge_marks, group_edge_ratings, 2, "mean")
)
names(group) <- sapply(group, `[[`, "name")

write_fixture("standard-setting-borderline", list(
  source = paste("Base R, set.seed(20261104): lm and predict(se.fit = TRUE) per station, the covariance of the",
                 "stations' cuts from the residuals' cross products over n - 2; mean, sd and median of the borderline",
                 "groups."),
  regression = regression,
  group = group
))
