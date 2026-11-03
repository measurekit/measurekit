# Content bounds in adaptive tests, written out in base R from the documented rules (contentBounds, maximumPriorityIndex,
# contentMinimumsMet), and the shadow test of mirtCAT as the reference for how well they hold the bounds.
#
# Replays: the tests of five persons who answer every item of a 3PL bank once, at their true thetas, so that each replay
# consumes a fixed set of responses. catR's Ii gives the information and its thetaEst the EAP estimate, which the
# replay's own 61-point EAP matches. A replay keeps catR's choice of an item (the highest score, the first in the pool
# on ties) and adds the rules of the guide: a unit is an item, or a testlet that has not begun; a unit that breaks the
# length, a maximum or the reach of a minimum scores -Inf, the minimums reaching by the count of items needed when that
# is at most the places left and otherwise by a greedy cover over the eligible items grouped by their set of areas, each
# place taking the group whose areas have the most items still needed; the priority index multiplies by w f1 f2 of Su (2016).
#
# Shadow test: mirtCAT's findNextItem with the integer program of van der Linden (2005), maximizing the information at
# the current estimate subject to the length and the bounds of every area, with the items given fixed. Two blueprints
# built from a hidden feasible form of 30 items out of a bank of 300, 250 persons each, whose answers to every item the
# fixture holds. The test of measurekit gives the same persons the same answers with the rules, so that the two differ by
# their selection alone, and expects no test outside the bounds and a root mean square error at most 1.05 times the
# shadow test's. The final estimate of both is the EAP of this script.

source("validation/psychometrics/common.R")
suppressPackageStartupMessages({
  library(catR)
  library(mirt)
  library(mirtCAT)
})

SEED <- 20261029
item_id <- function(i) sprintf("item%03d", i)

# ---- Estimate and information ---------------------------------------------------------------------------------------

nodes <- seq(-6, 6, length.out = 61)
weights <- dnorm(nodes) / sum(dnorm(nodes))

probability <- function(bank, i, theta) bank[i, "c"] + (bank[i, "d"] - bank[i, "c"]) * plogis(bank[i, "a"] * (theta - bank[i, "b"]))

# The posterior mean and SD of theta over the 61 nodes under N(0, 1): measurekit's posteriorMeanEstimate.
eap <- function(bank, out, x) {
  posterior <- weights
  for (k in seq_along(out)) {
    p <- probability(bank, out[k], nodes)
    posterior <- posterior * if (x[k] == 1) p else 1 - p
  }
  posterior <- posterior / sum(posterior)
  theta <- sum(posterior * nodes)
  list(theta = theta, standardError = sqrt(sum(posterior * (nodes - theta)^2)))
}

check_bank <- as.matrix(genDichoMatrix(items = 5, model = "3PL", seed = SEED))
stopifnot(abs(probability(check_bank, 1, 0.3) - Pi(0.3, check_bank[1, , drop = FALSE])$Pi) < 1e-14,
          abs(eap(check_bank, c(1, 2), c(1, 0))$theta -
                thetaEst(check_bank[1:2, ], c(1, 0), method = "EAP", parInt = c(-6, 6, 61))) < 1e-6)

# ---- The rules --------------------------------------------------------------------------------------------------------

# A replay of one person's test. `case` holds the bank, `areas` (the areas of each item), `testlet` (NA for an item alone),
# `exposure` (0 or 1), `enemies` (sets of item numbers), the `constraints` (area, minimum, maximum, weight), the `length`
# and the `rule`: "bounds", "priority" or "both". Before each item: the estimate; items administered, withheld or enemies of
# an administered item are not eligible; the testlet in progress continues without the filter, the stopping rule or exposure
# control, with its first eligible item or, adaptively, the one of the highest score; otherwise the test ends if no item is
# eligible or the stopping rule says so, else the scores of the rule are given to each unit, as the guide says, the highest
# is taken, and an entry item with exposure 0 withholds its unit unless it is the last. Counts how often each branch of the
# minimums ran.
replay <- function(case, responses) {
  bank <- case$bank
  n <- nrow(bank)
  constraints <- case$constraints
  K <- nrow(constraints)
  L <- case$length
  testlet <- case$testlet
  in_areas <- lapply(case$areas, function(a) which(constraints$area %in% a))
  key <- vapply(in_areas, paste, "", collapse = ",")
  group_of <- match(key, unique(key))
  group_areas <- in_areas[!duplicated(key)]
  enemy_of <- lapply(seq_len(n), function(i) unlist(lapply(case$enemies, function(set) if (i %in% set) setdiff(set, i))))
  # How often a unit was safe by the count of needs, safe by the cover, or only within the maximums, and how often a step
  # fell back to the units that fill the most needs or ended.
  paths <- c(safe_by_count = 0, safe_by_cover = 0, within = 0, fallback = 0, ended = 0)
  members_of <- function(i, eligible) if (is.na(testlet[i])) i else which(eligible & !is.na(testlet) & testlet == testlet[i])
  counts <- function(items) tabulate(as.integer(unlist(in_areas[items])), K)

  priority <- function(scores, eligible, out) {
    x <- counts(out)
    t <- length(out)
    factor <- vapply(seq_len(K), function(k) {
      w <- constraints$weight[k]
      outside <- L - constraints$minimum[k]
      share <- if (outside > 0) (outside - (t - x[k])) / outside else 0
      if (!(share > 0)) return(w)
      room <- if (constraints$maximum[k] > 0) max(0, (constraints$maximum[k] - x[k]) / constraints$maximum[k]) else 0
      w * room * share
    }, 0)
    for (i in which(eligible & scores > -Inf)) scores[i] <- scores[i] * prod(factor[in_areas[[i]]])
    scores
  }

  covers <- function(members, eligible, out, a) {
    x <- counts(out)
    available <- tabulate(group_of[eligible], length(group_areas))
    for (m in members) available[group_of[m]] <- available[group_of[m]] - 1
    need <- pmax(0, constraints$minimum - x - a)
    room <- constraints$maximum - x - a
    places <- L - length(out) - length(members)
    outstanding <- sum(need)
    while (places > 0 && outstanding > 0) {
      best <- 0
      best_gain <- 0
      for (g in seq_along(group_areas)) {
        ar <- group_areas[[g]]
        if (available[g] == 0 || any(room[ar] < 1)) next
        gain <- sum(need[ar])
        if (gain > best_gain) {
          best <- g
          best_gain <- gain
        }
      }
      if (best == 0) return(FALSE)
      available[best] <- available[best] - 1
      ar <- group_areas[[best]]
      room[ar] <- room[ar] - 1
      outstanding <- outstanding - sum(need[ar] > 0)
      need[ar] <- pmax(0, need[ar] - 1)
      places <- places - 1
    }
    outstanding == 0
  }

  bounds <- function(scores, eligible, out) {
    x <- counts(out)
    t <- length(out)
    need <- pmax(0, constraints$minimum - x)
    verdict <- rep(0, n) # 0 out, 1 within the maximums, 2 safe
    filled <- rep(0, n)
    judged <- integer(0)
    for (i in which(eligible)) {
      members <- members_of(i, eligible)
      if (!is.na(testlet[i])) {
        if (testlet[i] %in% judged) next
        judged <- c(judged, testlet[i])
      }
      if (!any(scores[members] > -Inf)) next
      a <- counts(members)
      size <- length(members)
      if (t + size > L || any(x + a > constraints$maximum)) next
      fill <- sum(pmin(a, need))
      places <- L - t - size
      if (sum(need) - fill <= places) {
        v <- 2
        paths["safe_by_count"] <<- paths["safe_by_count"] + 1
      } else if (covers(members, eligible, out, a)) {
        v <- 2
        paths["safe_by_cover"] <<- paths["safe_by_cover"] + 1
      } else {
        v <- 1
        paths["within"] <<- paths["within"] + 1
      }
      verdict[members] <- v
      filled[members] <- fill
    }
    candidates <- eligible & scores > -Inf
    if (any(verdict[candidates] == 2)) {
      keep <- verdict == 2
    } else if (any(verdict[candidates] > 0)) {
      paths["fallback"] <<- paths["fallback"] + 1
      keep <- verdict > 0 & filled == max(filled[candidates & verdict > 0])
    } else {
      keep <- rep(FALSE, n)
    }
    scores[eligible & !keep] <- -Inf
    scores
  }

  out <- integer(0)
  x <- numeric(0)
  withheld <- integer(0)
  estimates <- list()
  repeat {
    current <- eap(bank, out, x)
    estimates[[length(estimates) + 1]] <- current
    eligible <- rep(TRUE, n)
    eligible[c(out, withheld)] <- FALSE
    for (a in out) eligible[enemy_of[[a]]] <- FALSE
    information <- Ii(current$theta, bank)$Ii
    last <- if (length(out) > 0) out[length(out)] else NA
    remaining <- if (!is.na(last) && !is.na(testlet[last])) which(eligible & !is.na(testlet) & testlet == testlet[last]) else integer(0)
    rule_scores <- function(eligible_now) {
      scores <- ifelse(eligible_now, information, -Inf)
      if (case$rule %in% c("priority", "both")) scores <- priority(scores, eligible_now, out)
      scores
    }
    entry_of <- function(members, scores) {
      if (case$within == "in-order") return(members[1])
      best <- members[which.max(scores[members])]
      if (scores[best] > -Inf) best else members[1]
    }
    if (length(remaining) > 0) {
      scores <- rule_scores(seq_len(n) %in% remaining)
      item <- entry_of(remaining, scores)
    } else if (!any(eligible) || case$stop(out, current)) {
      break
    } else {
      repeat {
        scores <- rule_scores(eligible)
        if (case$rule %in% c("bounds", "both")) scores <- bounds(scores, eligible, out)
        unit_scores <- rep(-Inf, n)
        for (i in which(eligible)) {
          members <- members_of(i, eligible)
          if (is.na(testlet[i])) {
            unit_scores[i] <- scores[i]
          } else if (i == entry_of(members, scores)) {
            scored <- members[scores[members] > -Inf]
            unit_scores[i] <- if (length(scored) == 0) -Inf else mean(scores[scored])
          }
        }
        if (all(unit_scores[eligible] == -Inf)) {
          item <- NA
          break
        }
        item <- which.max(unit_scores)
        unit <- members_of(item, eligible)
        if (sum(eligible) == length(unit) || case$exposure[item] >= 1) break
        withheld <- c(withheld, unit)
        eligible[unit] <- FALSE
      }
      if (is.na(item)) {
        paths["ended"] <- paths["ended"] + 1
        break
      }
    }
    out <- c(out, item)
    x <- c(x, responses[item])
  }
  list(administered = I(item_id(out)), responses = I(x), estimates = estimates[-1], initial = estimates[[1]],
       counts = I(counts(out)), paths = paths)
}

# ---- The bank and the cases -------------------------------------------------------------------------------------------

set.seed(SEED + 700)
PERSONS <- c(-1.5, -0.5, 0.3, 1.0, 1.8)
N <- 60
LENGTH <- 12
bank <- as.matrix(genDichoMatrix(items = N, model = "3PL", seed = SEED + 701))
# Twelve cells of four domains and three types, five items each, scattered through the pool.
cell <- sample(rep(seq_len(12), each = 5))
domain <- c("algebra", "geometry", "number", "statistics")[(cell - 1) %% 4 + 1]
type <- c("recall", "apply", "reason")[(cell - 1) %/% 4 + 1]
responses <- t(sapply(PERSONS, function(theta) as.numeric(runif(N) < sapply(seq_len(N), function(i) probability(bank, i, theta)))))

constraint <- function(area, minimum, maximum, weight = 1) data.frame(area = area, minimum = minimum, maximum = maximum, weight = weight)
blueprint <- function(...) do.call(rbind, list(...))

length_stop <- function(out, current) length(out) >= LENGTH
precision_stop <- function(areas, constraints, precision, maximum) {
  function(out, current) {
    counts <- tabulate(as.integer(unlist(lapply(out, function(i) which(constraints$area %in% areas[[i]])))), nrow(constraints))
    length(out) >= maximum || (current$standardError <= precision && all(counts >= constraints$minimum))
  }
}

no_testlet <- rep(NA_character_, N)
no_enemies <- list()
all_exposed <- rep(1, N)

make_case <- function(name, areas, constraints, rule, stop = length_stop, within = "in-order", testlet = no_testlet,
                      exposure = all_exposed, enemies = no_enemies, length = LENGTH, stopping = list(type = "length")) {
  list(name = name, bank = bank, areas = areas, constraints = constraints, rule = rule, stop = stop, within = within,
       testlet = testlet, exposure = exposure, enemies = enemies, length = length, stopping = stopping)
}

run_case <- function(case) {
  persons <- lapply(seq_along(PERSONS), function(p) {
    result <- replay(case, responses[p, ])
    message(case$name, ", person ", p, ": ", length(result$administered), " items ",
            paste(sub("item0*", "", result$administered), collapse = " "), "  ", paste(result$paths, collapse = "/"))
    PATHS <<- PATHS + result$paths
    result[setdiff(names(result), "paths")]
  })
  list(name = case$name, rule = case$rule, within = case$within, length = case$length, stopping = case$stopping,
       constraints = lapply(seq_len(nrow(case$constraints)), function(k) as.list(case$constraints[k, ])),
       enemies = lapply(case$enemies, item_id),
       items = lapply(seq_len(N), function(i) {
         item <- list(id = item_id(i), a = bank[i, "a"], b = bank[i, "b"], c = bank[i, "c"], d = bank[i, "d"],
                      contentAreas = I(case$areas[[i]]), exposure = case$exposure[i])
         if (!is.na(case$testlet[i])) item$testlet <- case$testlet[i]
         item
       }),
       persons = persons)
}
PATHS <- c(safe_by_count = 0, safe_by_cover = 0, within = 0, fallback = 0, ended = 0)
by_domain <- lapply(seq_len(N), function(i) domain[i])
by_domain_type <- lapply(seq_len(N), function(i) c(domain[i], type[i]))
domains <- c("algebra", "geometry", "number", "statistics")
types <- c("recall", "apply", "reason")

# One facet: the domains.
c_domains <- blueprint(constraint("algebra", 2, 4), constraint("geometry", 2, 4), constraint("number", 2, 4),
                       constraint("statistics", 2, 4, 2))
# Two facets whose minimums each sum to the length, so that the cover decides at every step, with exact bounds and with
# margins.
c_exact <- blueprint(constraint("algebra", 3, 3), constraint("geometry", 3, 3), constraint("number", 3, 3),
                     constraint("statistics", 3, 3), constraint("recall", 4, 4), constraint("apply", 4, 4),
                     constraint("reason", 4, 4))
c_margin <- blueprint(constraint("algebra", 2, 4), constraint("geometry", 2, 4), constraint("number", 3, 4),
                      constraint("statistics", 2, 3), constraint("recall", 3, 5), constraint("apply", 3, 5),
                      constraint("reason", 3, 4))
# Three areas that compete, and items in none: where the index spreads them.
by_three <- lapply(seq_len(N), function(i) if (domain[i] == "statistics") character(0) else domain[i])
c_three <- blueprint(constraint("algebra", 3, 5), constraint("geometry", 3, 5), constraint("number", 3, 5))
# Two areas that every item is in one of, with exact bounds, where the index gives the filter's test.
by_half <- lapply(seq_len(N), function(i) if (domain[i] %in% c("algebra", "geometry")) "first half" else "second half")
c_half <- blueprint(constraint("first half", 6, 6), constraint("second half", 6, 6))
# Domains with maximums that leave no item after 8: a shorter test.
c_full <- blueprint(constraint("algebra", 0, 2), constraint("geometry", 0, 2), constraint("number", 0, 2),
                    constraint("statistics", 0, 2))

# Testlets of three or four items across domains, drawn from the 30 most informative items so that the tests take them,
# enemies inside and across testlets, and testlets and items that exposure control always withholds.
informative <- order(-rowMeans(sapply(PERSONS, function(theta) Ii(theta, bank)$Ii)))[1:30]
testlet_of <- no_testlet
testlet_members <- split(sample(informative, 21), rep(1:6, c(3, 4, 3, 4, 3, 4)))
for (s in seq_along(testlet_members)) testlet_of[testlet_members[[s]]] <- sprintf("set%d", s)
testlet_exposure <- all_exposed
testlet_exposure[which(testlet_of == "set2")] <- 0
alone <- intersect(informative, which(is.na(testlet_of)))
testlet_exposure[alone[c(1, 4)]] <- 0
testlet_enemies <- list(c(testlet_members[[1]][1], testlet_members[[1]][3]), c(testlet_members[[3]][2], alone[2], alone[6]))

# Enemies that leave statistics short of its minimum of 4: at most two of its items can be given.
statistics <- which(domain == "statistics")
enemy_statistics <- list(statistics[1:8], statistics[9:15])
c_short <- blueprint(constraint("algebra", 2, 4), constraint("geometry", 2, 4), constraint("number", 2, 4),
                     constraint("statistics", 4, 6))

cases <- list(
  run_case(make_case("the domains", by_domain, c_domains, "bounds")),
  run_case(make_case("two facets with exact bounds", by_domain_type, c_exact, "bounds")),
  run_case(make_case("two facets with exact bounds and the index", by_domain_type, c_exact, "both")),
  run_case(make_case("two facets with margins", by_domain_type, c_margin, "bounds")),
  run_case(make_case("two facets with margins and the index", by_domain_type, c_margin, "both")),
  run_case(make_case("the index alone", by_three, c_three, "priority")),
  run_case(make_case("three areas with the filter", by_three, c_three, "bounds")),
  run_case(make_case("three areas with the filter and the index", by_three, c_three, "both")),
  run_case(make_case("two exact areas with the filter", by_half, c_half, "bounds")),
  run_case(make_case("two exact areas with the filter and the index", by_half, c_half, "both")),
  run_case(make_case("maximums that end the test", by_domain, c_full, "bounds")),
  run_case(make_case("testlets, enemies and exposure", by_domain_type, c_margin, "both", testlet = testlet_of,
                     exposure = testlet_exposure, enemies = testlet_enemies)),
  run_case(make_case("testlets taken adaptively", by_domain_type, c_margin, "both", within = "adaptive", testlet = testlet_of,
                     exposure = testlet_exposure, enemies = testlet_enemies)),
  run_case(make_case("enemies that leave a minimum out of reach", by_domain, c_short, "bounds", enemies = enemy_statistics)),
  run_case(make_case("precision once the minimums are met", by_domain_type, c_margin, "bounds", length = 20,
                     stop = precision_stop(by_domain_type, c_margin, 0.5, 20), stopping = list(type = "precision", value = 0.5, length = 20)))
)

sequences <- function(name) lapply(cases[[which(vapply(cases, function(case) case$name, "") == name)]]$persons, function(p) as.character(p$administered))

# Every branch of the minimums runs in some case, and the order of a testlet's items changes a test.
stopifnot(!identical(sequences("testlets, enemies and exposure"), sequences("testlets taken adaptively")))
message("Branches of the filter: ", paste(names(PATHS), PATHS, collapse = ", "))
stopifnot(all(PATHS > 0))

# ---- Claims of the guide that the cases show --------------------------------------------------------------------------

# With two areas and exact bounds, the index gives the filter's test.
stopifnot(identical(sequences("two exact areas with the filter"), sequences("two exact areas with the filter and the index")))
# With three areas, it does not.
stopifnot(!identical(sequences("three areas with the filter"), sequences("three areas with the filter and the index")))
# The cover decides where the minimums of two facets sum to more than the places left; the exact blueprint is met.
for (case_name in c("two facets with exact bounds", "two facets with exact bounds and the index")) {
  for (person in cases[[which(vapply(cases, function(case) case$name, "") == case_name)]]$persons) {
    stopifnot(identical(as.numeric(person$counts), c(3, 3, 3, 3, 4, 4, 4)), length(person$administered) == 12)
  }
}
# Maximums end the test early; enemies leave the minimum of statistics unmet.
stopifnot(all(vapply(cases[[which(vapply(cases, function(case) case$name, "") == "maximums that end the test")]]$persons,
                     function(person) length(person$administered) == 8, TRUE)))
stopifnot(all(vapply(cases[[which(vapply(cases, function(case) case$name, "") == "enemies that leave a minimum out of reach")]]$persons,
                     function(person) person$counts[4] < 4, TRUE)))

# ---- The shadow test ----------------------------------------------------------------------------------------------------

SHADOW_BANK <- 300
SHADOW_PERSONS <- 250
SHADOW_LENGTH <- 30

# A blueprint of facets, each with levels in the given shares, from a hidden feasible form of 30 items drawn from a bank of
# 300; each area's bounds are the form's count less or plus a margin of 0 to `margin`.
shadow_blueprint <- function(name, facets, margin, seed) {
  set.seed(seed)
  shadow_bank <- as.matrix(genDichoMatrix(items = SHADOW_BANK, model = "3PL", seed = seed))
  areas <- do.call(cbind, lapply(names(facets), function(facet) {
    levels <- facets[[facet]]
    paste0(facet, ":", sample(seq_along(levels), SHADOW_BANK, replace = TRUE, prob = levels))
  }))
  hidden <- sample(SHADOW_BANK, SHADOW_LENGTH)
  all_areas <- sort(unique(as.vector(areas)))
  in_area <- sapply(all_areas, function(a) as.numeric(rowSums(areas == a) > 0))
  form_counts <- colSums(in_area[hidden, , drop = FALSE])
  constraints <- data.frame(
    area = all_areas,
    minimum = pmax(0, form_counts - sample(0:margin, length(all_areas), replace = TRUE)),
    maximum = form_counts + sample(0:margin, length(all_areas), replace = TRUE),
    weight = 1
  )
  constr_fun <- function(design, person, test) {
    lhs <- rbind(rep(1, SHADOW_BANK), t(in_area), t(in_area))
    data.frame(lhs, c("==", rep(">=", length(all_areas)), rep("<=", length(all_areas))),
               c(SHADOW_LENGTH, constraints$minimum, constraints$maximum))
  }
  feasible <- lpSolve::lp("max", rep(1, SHADOW_BANK), rbind(rep(1, SHADOW_BANK), t(in_area), t(in_area)),
                          c("==", rep(">=", length(all_areas)), rep("<=", length(all_areas))),
                          c(SHADOW_LENGTH, constraints$minimum, constraints$maximum), all.bin = TRUE)
  stopifnot(feasible$status == 0)
  thetas <- rnorm(SHADOW_PERSONS)
  answers <- t(sapply(thetas, function(theta) as.numeric(runif(SHADOW_BANK) < sapply(seq_len(SHADOW_BANK), function(i) probability(shadow_bank, i, theta)))))
  model <- generate.mirt_object(data.frame(a1 = shadow_bank[, "a"], d = -shadow_bank[, "a"] * shadow_bank[, "b"], g = shadow_bank[, "c"]), "3PL")
  next_item <- function(design, person, test) {
    objective <- computeCriteria(person = person, design = design, test = test, criteria = "MI")
    findNextItem(person = person, design = design, test = test, objective = objective)
  }
  started <- proc.time()[["elapsed"]]
  chunks <- split(seq_len(SHADOW_PERSONS), ceiling(seq_len(SHADOW_PERSONS) / 250))
  administered <- list()
  for (chunk in chunks) {
    result <- mirtCAT(mo = model, local_pattern = answers[chunk, , drop = FALSE], method = "EAP",
                      design = list(min_items = SHADOW_LENGTH, max_items = SHADOW_LENGTH, constr_fun = constr_fun,
                                    customNextItem = next_item), verbose = FALSE)
    administered <- c(administered, lapply(result, function(r) as.integer(r$items_answered)))
    message(name, ": ", length(administered), " persons, ", round(proc.time()[["elapsed"]] - started), " s")
  }
  estimates <- t(sapply(seq_along(administered), function(p) {
    e <- eap(shadow_bank, administered[[p]], answers[p, administered[[p]]])
    c(e$theta, e$standardError)
  }))
  outside <- sum(vapply(administered, function(items) {
    counts <- colSums(in_area[items, , drop = FALSE])
    any(counts < constraints$minimum | counts > constraints$maximum)
  }, TRUE))
  stopifnot(all(lengths(administered) == SHADOW_LENGTH))
  message(name, ": ", outside, " tests outside the bounds, rmse ", sqrt(mean((estimates[, 1] - thetas)^2)))
  list(name = name, bank = lapply(seq_len(SHADOW_BANK), function(i) {
    list(id = item_id(i), a = shadow_bank[i, "a"], b = shadow_bank[i, "b"], c = shadow_bank[i, "c"], d = 1,
         contentAreas = I(areas[i, ]))
  }), constraints = lapply(seq_len(nrow(constraints)), function(k) as.list(constraints[k, ])),
  length = SHADOW_LENGTH, thetas = I(thetas), answers = answers, shadow = list(
    rmse = sqrt(mean((estimates[, 1] - thetas)^2)), bias = mean(estimates[, 1] - thetas), outside = outside,
    meanStandardError = mean(estimates[, 2])))
}

# The two blueprints run in two processes: each takes about 30 s.
shadow <- parallel::mclapply(
  list(
    list(name = "two facets", facets = list(domain = c(0.3, 0.25, 0.25, 0.2), type = c(0.4, 0.35, 0.25)), seed = SEED + 710),
    list(name = "three facets", facets = list(domain = c(0.4, 0.35, 0.25), type = c(0.5, 0.3, 0.2), format = c(0.6, 0.4)), seed = SEED + 720)
  ),
  function(spec) shadow_blueprint(spec$name, spec$facets, 1, spec$seed),
  mc.cores = 2
)
stopifnot(!any(vapply(shadow, inherits, TRUE, "try-error")))

write_fixture("cat-bounds", list(
  source = sprintf("catR's genDichoMatrix(items = 60, model = \"3PL\", seed = %d) in 12 cells of 4 domains and 3 types; replays in base R, catR's Ii for the information; the shadow test of mirtCAT's findNextItem on banks of 300 items.", SEED + 701),
  responses = unname(responses),
  cases = cases,
  shadowTests = shadow
))
