# Assembly of parallel forms (assembleForms), written out in base R from the documented rules, and highs as the reference for
# how well the rotation meets a blueprint that a solver proves feasible.
#
# Replays: forms assembled from a bank of 60 3PL items from catR's genDichoMatrix, with catR's Ii for the information. The
# rules: the units are the items alone and the testlets, taken whole; the forms take one unit each in rotation, in order and
# then in reverse; a unit is available to a form when it fits the places left, none of its items is on the form or an enemy
# of one on the form, and the form would share at most `overlap` items with any other; a form takes the available unit with
# the smallest weighted sum of the deviations of its projected properties, the first on ties, which are deviations that
# differ by less than a share of 1e-12. Sums run item by item, in
# the order of the pool, as the code of the package adds them, so that ties and near ties fall alike.
#
# Blueprints: 85 blueprints on two banks of 300 items, one with two facets and one with three, each built from a hidden
# feasible form of 30 items whose area counts, less or plus a margin of 0 or 1, are the bounds and whose information at
# three thetas is the target. highs confirms that four forms with no item in common meet the bounds, and for the first
# blueprints finds the four forms whose largest gap from the target is smallest (minimax, the model of Boekkooi-Timminga
# and van der Linden), within a time limit of 30 seconds. The test of measurekit assembles four forms for each blueprint
# and expects every bound met and every form within 5% of the target.

source("validation/psychometrics/common.R")
suppressPackageStartupMessages({
  library(catR)
  library(highs)
})

SEED <- 20261029
item_id <- function(i) sprintf("item%03d", i)
logistic_item <- function(bank, i) list(id = item_id(i), a = bank[i, "a"], b = bank[i, "b"], c = bank[i, "c"], d = bank[i, "d"])

# ---- The rules ----------------------------------------------------------------------------------------------------------

# Assembles the forms of a case: `bank`, `areas` (the areas of each item), `constraints` (area, minimum, maximum, weight),
# `targets` (theta, information, weight), `testlet` (NA for an item alone), `enemies` (sets of item numbers), `forms`,
# `overlap` and `length`. Returns the item numbers of each form, in the order of selection.
assemble_forms <- function(case) {
  bank <- case$bank
  n <- nrow(bank)
  constraints <- case$constraints
  targets <- case$targets
  K <- nrow(constraints)
  P <- K + nrow(targets)
  item_values <- matrix(0, n, P)
  for (k in seq_len(K)) item_values[, k] <- as.numeric(vapply(case$areas, function(a) constraints$area[k] %in% a, TRUE))
  for (t in seq_len(nrow(targets))) item_values[, K + t] <- Ii(targets$theta[t], bank)$Ii
  lower <- c(constraints$minimum, targets$information)
  upper <- c(constraints$maximum, targets$information)
  weight <- c(constraints$weight, targets$weight)
  enemy_of <- lapply(seq_len(n), function(i) unlist(lapply(case$enemies, function(set) if (i %in% set) setdiff(set, i))))

  units <- list()
  seen <- character(0)
  for (i in seq_len(n)) {
    if (is.na(case$testlet[i])) {
      units[[length(units) + 1]] <- i
    } else if (!(case$testlet[i] %in% seen)) {
      seen <- c(seen, case$testlet[i])
      units[[length(units) + 1]] <- which(case$testlet == case$testlet[i])
    }
  }
  divided <- vapply(units, function(u) any(unlist(enemy_of[u]) %in% u), TRUE)
  sizes <- lengths(units)
  unit_values <- t(vapply(units, function(u) {
    value <- numeric(P)
    for (i in u) value <- value + item_values[i, ]
    value
  }, numeric(P)))

  F <- case$forms
  chosen <- replicate(F, integer(0), simplify = FALSE)
  totals <- matrix(0, F, P)
  held <- matrix(FALSE, F, n)
  enemies_held <- matrix(0L, F, n)
  shared <- matrix(0L, F, F)
  holders <- replicate(n, integer(0), simplify = FALSE)
  done <- rep(FALSE, F)

  available <- function(f, u, places) {
    members <- units[[u]]
    if (sizes[u] > places || divided[u] || any(held[f, members]) || any(enemies_held[f, members] > 0)) return(FALSE)
    count <- tabulate(unlist(holders[members]), F)
    all(count[-f] + shared[f, -f] <= case$overlap)
  }

  step <- function(f) {
    if (done[f]) return(FALSE)
    places <- case$length - length(chosen[[f]])
    candidates <- Filter(function(u) available(f, u, places), seq_along(units))
    if (length(candidates) == 0) {
      done[f] <<- TRUE
      return(FALSE)
    }
    sums <- numeric(P)
    remaining <- 0
    for (u in candidates) {
      remaining <- remaining + sizes[u]
      sums <- sums + unit_values[u, ]
    }
    best <- 0
    best_deviation <- Inf
    for (u in candidates) {
      deviation <- 0
      for (p in seq_len(P)) {
        value <- unit_values[u, p]
        others <- if (remaining > sizes[u]) (sums[p] - value) / (remaining - sizes[u]) else 0
        projection <- totals[f, p] + value + (places - sizes[u]) * others
        deviation <- deviation + weight[p] * (max(0, lower[p] - projection) + max(0, projection - upper[p]))
      }
      if (deviation < best_deviation * (1 - 1e-12)) {
        best <- u
        best_deviation <- deviation
      }
    }
    for (i in units[[best]]) {
      chosen[[f]] <<- c(chosen[[f]], i)
      held[f, i] <<- TRUE
      for (e in enemy_of[[i]]) enemies_held[f, e] <<- enemies_held[f, e] + 1L
      for (g in holders[[i]]) {
        shared[f, g] <<- shared[f, g] + 1L
        shared[g, f] <<- shared[g, f] + 1L
      }
      holders[[i]] <<- c(holders[[i]], f)
    }
    totals[f, ] <<- totals[f, ] + unit_values[best, ]
    TRUE
  }

  round <- 0
  repeat {
    order <- if (round %% 2 == 0) seq_len(F) else rev(seq_len(F))
    taken <- FALSE
    for (f in order) taken <- step(f) || taken
    if (!taken) break
    round <- round + 1
  }
  lapply(seq_len(F), function(f) list(items = chosen[[f]], counts = totals[f, seq_len(K)], information = totals[f, K + seq_len(nrow(targets))]))
}

# ---- The replays --------------------------------------------------------------------------------------------------------

set.seed(SEED + 800)
N <- 60
replay_bank <- as.matrix(genDichoMatrix(items = N, model = "3PL", seed = SEED + 801))
domain <- c("algebra", "geometry", "number", "statistics")[sample(rep(1:4, length.out = N))]
type <- c("recall", "apply", "reason")[sample(rep(1:3, length.out = N))]
replay_areas <- lapply(seq_len(N), function(i) c(domain[i], type[i]))
replay_constraints <- data.frame(
  area = c("algebra", "geometry", "number", "statistics", "recall", "apply", "reason"),
  minimum = c(2, 2, 2, 1, 3, 3, 2), maximum = c(4, 4, 3, 3, 5, 4, 4), weight = c(1, 1, 1, 2, 1, 1, 1)
)
replay_targets <- data.frame(theta = c(-1, 0.5), information = c(2.5, 3), weight = c(1, 1))
no_testlet <- rep(NA_character_, N)

testlet <- no_testlet
testlet_members <- split(sample(N, 18), rep(1:6, 3))
for (s in seq_along(testlet_members)) testlet[testlet_members[[s]]] <- sprintf("set%d", s)
alone <- which(is.na(testlet))
enemies <- list(c(alone[1], alone[2]), c(alone[3], alone[4], alone[5]), c(testlet_members[[1]][1], alone[6]))
divided_enemies <- list(c(testlet_members[[2]][1], testlet_members[[2]][2]), c(alone[7], alone[8]))

form_case <- function(name, forms, overlap, length, testlet = no_testlet, enemies = list(), constraints = replay_constraints,
                      targets = replay_targets) {
  bank <- replay_bank
  areas <- replay_areas
  case <- list(bank = bank, areas = areas, constraints = constraints, targets = targets, testlet = testlet,
               enemies = enemies, forms = forms, overlap = overlap, length = length)
  result <- assemble_forms(case)
  message(name, ": ", paste(vapply(result, function(f) length(f$items), 0L), collapse = " "), " items; first form ",
          paste(result[[1]]$items, collapse = " "))
  list(
    name = name, forms = forms, overlap = overlap, length = length,
    constraints = lapply(seq_len(nrow(constraints)), function(k) as.list(constraints[k, ])),
    target = lapply(seq_len(nrow(targets)), function(t) as.list(targets[t, ])),
    enemies = lapply(enemies, item_id),
    items = lapply(seq_len(nrow(bank)), function(i) {
      item <- c(logistic_item(bank, i), list(contentAreas = I(areas[[i]])))
      if (!is.na(testlet[i])) item$testlet <- testlet[i]
      item
    }),
    expected = lapply(result, function(f) list(items = I(item_id(f$items)), counts = I(f$counts), information = I(f$information)))
  )
}

replays <- list(
  form_case("one form", 1, 0, 10),
  form_case("three forms with no item in common", 3, 0, 10),
  form_case("three forms with enemies", 3, 0, 10, enemies = enemies),
  form_case("four forms that share two items", 4, 2, 10),
  form_case("two forms that share one item with enemies", 2, 1, 12, enemies = enemies),
  form_case("two forms with testlets taken whole", 2, 0, 12, testlet = testlet),
  form_case("three forms with testlets, enemies and shared items", 3, 2, 12, testlet = testlet, enemies = enemies),
  form_case("a testlet whose items are enemies is never taken", 2, 0, 12, testlet = testlet, enemies = divided_enemies),
  form_case("forms short of items", 4, 0, 20),
  form_case("content only", 3, 1, 10, targets = replay_targets[0, ]),
  form_case("information only", 3, 1, 10, constraints = replay_constraints[0, ])
)

# The forms of the replays are as the guide says.
stopifnot(length(unique(unlist(lapply(replays[[2]]$expected, function(f) f$items)))) == 30)
short <- vapply(replays[[9]]$expected, function(f) length(f$items), 0L)
stopifnot(sum(short) == N, all(short < 20))

# ---- Blueprints and the solver ----------------------------------------------------------------------------------------

BLUEPRINTS <- 85
FORMS <- 4
FORM_LENGTH <- 30
POOL <- 300
THETAS <- c(-1, 0, 1)
TIME_LIMIT <- 30

# The constraint matrix of `forms` disjoint forms of `length` items over items with the 0/1 membership of each area.
blueprint_model <- function(membership, minimum, maximum, forms, length) {
  n <- nrow(membership)
  variables <- forms * n
  index <- function(f, i) (f - 1) * n + i
  rows <- list()
  lhs <- numeric(0)
  rhs <- numeric(0)
  add <- function(coefficients, low, high) {
    rows[[length(rows) + 1]] <<- coefficients
    lhs <<- c(lhs, low)
    rhs <<- c(rhs, high)
  }
  for (i in seq_len(n)) {
    row <- numeric(variables)
    row[index(seq_len(forms), i)] <- 1
    add(row, -Inf, 1)
  }
  for (f in seq_len(forms)) {
    row <- numeric(variables)
    row[index(f, seq_len(n))] <- 1
    add(row, length, length)
    for (a in seq_len(ncol(membership))) {
      row <- numeric(variables)
      row[index(f, seq_len(n))] <- membership[, a]
      add(row, minimum[a], maximum[a])
    }
  }
  list(rows = rows, lhs = lhs, rhs = rhs, variables = variables, index = index)
}

solve_blueprint <- function(model, extra_variable = FALSE, objective = NULL, rows = list(), lhs = numeric(0), rhs = numeric(0),
                            time_limit = Inf) {
  variables <- model$variables + as.integer(extra_variable)
  pad <- function(row) c(row, numeric(variables - length(row)))
  A <- do.call(rbind, lapply(c(model$rows, rows), pad))
  upper <- rep(1, variables)
  if (extra_variable) upper[variables] <- 1
  highs_solve(L = if (is.null(objective)) numeric(variables) else objective, lower = rep(0, variables), upper = upper, A = A,
              lhs = c(model$lhs, lhs), rhs = c(model$rhs, rhs),
              types = c(rep("I", model$variables), if (extra_variable) "C"), control = highs_control(time_limit = time_limit))
}

# A bank of 300 items in the areas of two or three facets, each level with its own share.
make_pool <- function(name, levels, seed) {
  set.seed(seed)
  shares <- lapply(levels, function(k) { w <- runif(k, 0.5, 1.5); w / sum(w) })
  bank <- as.matrix(genDichoMatrix(items = POOL, model = "3PL", seed = seed))
  areas <- do.call(cbind, lapply(names(levels), function(facet) {
    paste0(facet, ":", sample(levels[[facet]], POOL, replace = TRUE, prob = shares[[facet]]))
  }))
  names_of_areas <- sort(unique(as.vector(areas)))
  list(name = name, bank = bank, areas = areas, names = names_of_areas,
       membership = sapply(names_of_areas, function(a) as.numeric(rowSums(areas == a) > 0)))
}

# A blueprint of a pool from a hidden feasible form: the bounds are the form's counts less or plus a margin, the target the
# form's information.
make_blueprint <- function(pool, index) {
  hidden <- sample(POOL, FORM_LENGTH)
  counts <- colSums(pool$membership[hidden, , drop = FALSE])
  margin <- sample(0:1, 1)
  minimum <- pmax(0, counts - sample(0:margin, length(pool$names), replace = TRUE))
  maximum <- counts + sample(0:margin, length(pool$names), replace = TRUE)
  information <- sapply(THETAS, function(theta) sum(Ii(theta, pool$bank)$Ii[hidden]))
  model <- blueprint_model(pool$membership, minimum, maximum, FORMS, FORM_LENGTH)
  feasible <- solve_blueprint(model, time_limit = 10)$status == 7
  list(index = index, pool = pool, minimum = minimum, maximum = maximum, information = information, model = model,
       feasible = feasible, margin = margin)
}

# The gap of the best four forms in minimax: the smallest t with every form's information within t of the target, as a share.
minimax_gap <- function(blueprint) {
  n <- POOL
  extra <- blueprint$model$variables + 1
  rows <- list()
  lhs <- numeric(0)
  rhs <- numeric(0)
  for (f in seq_len(FORMS)) {
    for (k in seq_along(THETAS)) {
      info <- Ii(THETAS[k], blueprint$pool$bank)$Ii
      row <- numeric(extra)
      row[blueprint$model$index(f, seq_len(n))] <- info
      row[extra] <- -blueprint$information[k]
      rows[[length(rows) + 1]] <- row
      lhs <- c(lhs, -Inf)
      rhs <- c(rhs, blueprint$information[k])
      row[extra] <- blueprint$information[k]
      rows[[length(rows) + 1]] <- row
      lhs <- c(lhs, blueprint$information[k])
      rhs <- c(rhs, Inf)
    }
  }
  objective <- c(numeric(blueprint$model$variables), 1)
  started <- proc.time()[["elapsed"]]
  result <- solve_blueprint(blueprint$model, TRUE, objective, rows, lhs, rhs, TIME_LIMIT)
  seconds <- proc.time()[["elapsed"]] - started
  solution <- result$primal_solution
  achieved <- max(unlist(lapply(seq_len(FORMS), function(f) {
    chosen <- which(round(solution[blueprint$model$index(f, seq_len(n))]) == 1)
    sapply(seq_along(THETAS), function(k) abs(sum(Ii(THETAS[k], blueprint$pool$bank)$Ii[chosen]) - blueprint$information[k]) / blueprint$information[k])
  })))
  list(gap = achieved, status = result$status, seconds = seconds)
}

pools <- list(
  make_pool("two facets", list(domain = 4, type = 3), SEED + 910),
  make_pool("three facets", list(domain = 3, type = 3, format = 2), SEED + 920)
)
set.seed(SEED + 930)
blueprints <- list()
for (p in seq_along(pools)) {
  wanted <- if (p == 1) 43 else 42
  found <- 0
  index <- 0
  while (found < wanted) {
    index <- index + 1
    blueprint <- make_blueprint(pools[[p]], index)
    if (blueprint$feasible) {
      found <- found + 1
      blueprints[[length(blueprints) + 1]] <- blueprint
    } else {
      message("blueprint ", index, " of ", pools[[p]]$name, " is infeasible; skipped")
    }
  }
}
stopifnot(length(blueprints) == BLUEPRINTS)
for (k in c(1, 2, 3, 44, 45, 46)) {
  solved <- minimax_gap(blueprints[[k]])
  blueprints[[k]]$minimax <- solved
  message("blueprint ", k, " (", blueprints[[k]]$pool$name, "): minimax gap ", round(solved$gap, 4), " (status ", solved$status, ", ", round(solved$seconds), " s)")
}

write_fixture("form-assembly", list(
  source = sprintf("catR's genDichoMatrix(items = 60, model = \"3PL\", seed = %d) for the replays in base R, and 85 blueprints on two banks of 300 items, whose four disjoint forms highs confirms feasible.", SEED + 801),
  replays = replays,
  forms = FORMS,
  length = FORM_LENGTH,
  pools = lapply(pools, function(pool) list(
    name = pool$name,
    items = lapply(seq_len(POOL), function(i) c(logistic_item(pool$bank, i), list(contentAreas = I(pool$areas[i, ])))),
    blueprints = lapply(Filter(function(b) identical(b$pool$name, pool$name), blueprints), function(b) list(
      margin = b$margin,
      constraints = lapply(seq_along(pool$names), function(a) list(area = pool$names[a], minimum = b$minimum[a], maximum = b$maximum[a])),
      target = lapply(seq_along(THETAS), function(k) list(theta = THETAS[k], information = b$information[k])),
      minimaxGap = if (is.null(b$minimax)) NULL else b$minimax$gap
    ))
  ))
))
