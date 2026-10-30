"""BKT fixtures from pyBKT: inference at fixed parameters and EM from matched starts for each extension, best fits from
pyBKT's random starts, a recovery study, and opportunities to mastery written out.

Run from the repository root: validation/knowledge-tracing/.venv/bin/python validation/knowledge-tracing/bkt.py
"""

import json
import math

import numpy as np
import pandas as pd
from pyBKT.fit import EM_fit, M_step, predict_onestep
from pyBKT.generate import random_model_uni
from pyBKT.util import data_helper

FIXTURES = "packages/knowledge-tracing/test/fixtures"
SEED = 20261029
# EM stops when the log-likelihood changes by at most TOLERANCE, in pyBKT and in the tests alike.
TOLERANCE = 1e-9
MAX_ITERATIONS = 10000
# Recovery thresholds, fixed before the simulation was first run: the largest absolute error of the estimates.
PRIOR_THRESHOLD = 0.1
RATE_THRESHOLD = 0.05


def write_fixture(name, value):
    path = f"{FIXTURES}/{name}.json"
    with open(path, "w") as file:
        # Python writes the shortest representation that reads back as the same double.
        json.dump(value, file, separators=(",", ":"), allow_nan=False)
    print("Wrote", path)


COLUMNS = {
    "order_id": "order",
    "user_id": "learner",
    "skill_name": "skill",
    "correct": "correct",
    "multigs": "item",
    "multilearn": "item",
    "multiprior": "group",
}


def pybkt_data(rows, skill, options, gs_names=None, resource_names=None):
    """pyBKT's data of one skill: model_type is (multilearn, multiprior, multipair, multigs)."""
    model_type = (options.get("itemLearning", False), options.get("priorGroups", False), False,
                  options.get("itemGuessSlip", False))
    return data_helper.convert_data(
        pd.DataFrame(rows), skill, defaults=dict(COLUMNS), model_type=model_type,
        gs_refs=None if gs_names is None else {skill: {"gs_names": gs_names}},
        resource_refs=None if resource_names is None else {skill: {"resource_names": resource_names}})[skill]


def pybkt_model(parameters, data, options):
    """pyBKT's model of the package's parameters of a skill. With prior groups, pyBKT's prior is 0 and each group's prior is
    the learn rate of a phantom first step."""
    resources = sorted(data["resource_names"].items(), key=lambda entry: entry[1])
    learning = []
    for name, _ in resources:
        if options.get("priorGroups") and name != "Default":
            learning.append((parameters["groupPriors"][name], 0.0))
        elif options.get("itemLearning"):
            rates = parameters["itemLearning"][name]
            learning.append((rates["learn"], rates["forget"]))
        else:
            learning.append((parameters["learn"], parameters["forget"]))
    answers = []
    for name, _ in sorted(data["gs_names"].items(), key=lambda entry: entry[1]):
        rates = parameters["itemGuessSlip"][name] if options.get("itemGuessSlip") else parameters
        answers.append((rates["guess"], rates["slip"]))
    prior = 0.0 if options.get("priorGroups") else parameters["prior"]
    learns, forgets = (np.array(values, dtype=float) for values in zip(*learning))
    guesses, slips = (np.array(values, dtype=float) for values in zip(*answers))
    return {
        "learns": learns, "forgets": forgets, "guesses": guesses, "slips": slips, "prior": prior,
        "As": np.array([[[1 - l, f], [l, 1 - f]] for l, f in learning]),
        "emissions": np.array([[[1 - g, g], [s, 1 - s]] for g, s in answers]),
        "pi_0": np.array([[1 - prior], [prior]]),
    }


def pybkt_predictions(model, data):
    """pyBKT's P(correct) and P(L) before each row of the data, by row label."""
    correct, states = predict_onestep.run(model, data, parallel=False)
    known = states[1]
    if data["multiprior_index"] is not None:
        correct = np.delete(correct, data["multiprior_index"])
        known = np.delete(known, data["multiprior_index"])
    return dict(zip(data["index"], zip(correct, known)))


def expectation_maximization(model, data, tolerance=TOLERANCE):
    """pyBKT's EM_fit loop, keeping the expected counts of the last M-step. Returns the model, the log-likelihoods and
    the counts: transitions by resource, to, from; emissions by guess and slip class, state, outcome."""
    resources, classes = len(model["learns"]), len(model["guesses"])
    likelihoods, counts = [], None
    for i in range(MAX_ITERATIONS):
        result = EM_fit.run(data, model, None, None, None, 1, parallel=False)
        for j in range(resources):
            result["all_trans_softcounts"][j] = result["all_trans_softcounts"][j].transpose()
        for j in range(classes):
            result["all_emission_softcounts"][j] = result["all_emission_softcounts"][j].transpose()
        likelihoods.append(float(result["total_loglike"][0][0]))
        if i > 1 and abs(likelihoods[i] - likelihoods[i - 1]) <= tolerance:
            break
        counts = (result["all_trans_softcounts"].copy(), result["all_emission_softcounts"].copy())
        model = M_step.run(model, result["all_trans_softcounts"], result["all_emission_softcounts"],
                           result["all_initial_softcounts"])
    return model, likelihoods, counts


def pooled(numerator, denominator):
    return float(numerator.sum() / denominator.sum())


def knowledge_tracing_parameters(model, data, options, counts):
    """the package's parameters of a skill from a pyBKT fit. A rate that groups or items have of their own is, for the skill,
    the ratio of the expected counts pooled over them, from the counts of the M-step that made the fit."""
    transitions, emissions = counts
    names = data["resource_names"]
    skill = {}
    if options.get("priorGroups"):
        default = names["Default"] - 1
        skill["groupPriors"] = {name: float(model["learns"][index - 1]) for name, index in names.items()
                                if name != "Default"}
        phantoms = [index - 1 for name, index in names.items() if name != "Default"]
        skill["prior"] = pooled(transitions[phantoms, 1, 0], transitions[phantoms, :, 0])
        skill["learn"], skill["forget"] = float(model["learns"][default]), float(model["forgets"][default])
    else:
        skill["prior"] = float(model["prior"])
        if options.get("itemLearning"):
            skill["itemLearning"] = {name: {"learn": float(model["learns"][index - 1]),
                                            "forget": float(model["forgets"][index - 1])}
                                     for name, index in names.items()}
            skill["learn"] = pooled(transitions[:, 1, 0], transitions[:, :, 0])
            skill["forget"] = pooled(transitions[:, 0, 1], transitions[:, :, 1])
        else:
            skill["learn"], skill["forget"] = float(model["learns"][0]), float(model["forgets"][0])
    if options.get("itemGuessSlip"):
        skill["itemGuessSlip"] = {name: {"guess": float(model["guesses"][index]), "slip": float(model["slips"][index])}
                                  for name, index in data["gs_names"].items()}
        skill["guess"] = pooled(emissions[:, 0, 1], emissions[:, 0, :])
        skill["slip"] = pooled(emissions[:, 1, 0], emissions[:, 1, :])
    else:
        skill["guess"], skill["slip"] = float(model["guesses"][0]), float(model["slips"][0])
    return skill


def outcome_log_likelihood(p, y):
    return math.log(p) if y == 1 else math.log1p(-p)


# The practice log: 40 learners in two prior groups answer 20 items each, drawn from 9 items of 3 skills, one item with
# two skills. Outcomes follow BKT with item rates, forgetting and group priors; an item with two skills is answered
# correctly with the mean of its skills' probabilities.
ITEM_SKILLS = {"a1": ["a"], "a2": ["a"], "a3": ["a"], "ab": ["a", "b"], "b1": ["b"], "b2": ["b"], "b3": ["b"],
               "c1": ["c"], "c2": ["c"]}
SKILLS = ["a", "b", "c"]
rng = np.random.default_rng(SEED)
TRUE = {skill: {
    "groupPriors": {"high": 0.6, "low": 0.2},
    "itemLearning": {item: {"learn": float(rng.uniform(0.05, 0.3)), "forget": float(rng.uniform(0, 0.05))}
                     for item, skills in ITEM_SKILLS.items() if skill in skills},
    "itemGuessSlip": {item: {"guess": float(rng.uniform(0.1, 0.3)), "slip": float(rng.uniform(0.05, 0.15))}
                      for item, skills in ITEM_SKILLS.items() if skill in skills},
} for skill in SKILLS}
LEARNERS = [f"L{index:02d}" for index in range(1, 41)]
GROUPS = {learner: "high" if index % 3 == 0 else "low" for index, learner in enumerate(LEARNERS)}
events = []
for index, learner in enumerate(LEARNERS):
    known = {skill: rng.random() < TRUE[skill]["groupPriors"][GROUPS[learner]] for skill in SKILLS}
    for attempt in range(20):
        item = str(rng.choice(sorted(ITEM_SKILLS)))
        skills = ITEM_SKILLS[item]
        rates = [TRUE[skill]["itemGuessSlip"][item] for skill in skills]
        chance = np.mean([1 - r["slip"] if known[skill] else r["guess"] for skill, r in zip(skills, rates)])
        events.append({"learner": learner, "item": item, "skills": skills, "time": attempt * 100 + index,
                       "outcome": int(rng.random() < chance)})
        for skill in skills:
            learning = TRUE[skill]["itemLearning"][item]
            known[skill] = rng.random() >= learning["forget"] if known[skill] else rng.random() < learning["learn"]
events.sort(key=lambda event: event["time"])
rows = [{"order": order, "learner": event["learner"], "skill": skill, "correct": event["outcome"],
         "item": event["item"], "group": GROUPS[event["learner"]], "event": order, "position": position}
        for order, event in enumerate(events) for position, skill in enumerate(event["skills"])]

# pyBKT drops the emission counts of a learner's only event of a skill, and rates every item without a following
# attempt as learned for sure, so the log avoids both; each group answers every item, for predictions by group.
for skill in SKILLS:
    for learner in LEARNERS:
        steps = [row for row in rows if row["skill"] == skill and row["learner"] == learner]
        assert len(steps) >= 2, (skill, learner)
    followed = {row["item"] for row in rows if row["skill"] == skill and any(
        later["skill"] == skill and later["learner"] == row["learner"] and later["order"] > row["order"]
        for later in rows)}
    assert followed == set(TRUE[skill]["itemGuessSlip"]), skill
    for group in ["high", "low"]:
        answered = {row["item"] for row in rows if row["skill"] == skill and row["group"] == group}
        assert answered == set(TRUE[skill]["itemGuessSlip"]), (skill, group)


def parameters_for(options, values):
    """the package's parameters with the rates that `options` uses, from per-skill values of every rate."""
    skills = {}
    for skill in SKILLS:
        v = values[skill]
        entry = {"prior": v["prior"], "learn": v["learn"], "forget": v["forget"] if options.get("forgetting") else 0.0,
                 "guess": v["guess"], "slip": v["slip"]}
        if options.get("priorGroups"):
            entry["groupPriors"] = v["groupPriors"]
        if options.get("itemLearning"):
            entry["itemLearning"] = {item: {"learn": r["learn"], "forget": r["forget"] if options.get("forgetting")
                                            else 0.0} for item, r in v["itemLearning"].items()}
        if options.get("itemGuessSlip"):
            entry["itemGuessSlip"] = v["itemGuessSlip"]
        skills[skill] = entry
    return {"skills": skills}


def rates(skill, prior, learn, forget, guess, slip, shift):
    """Every rate of a skill: the given ones, and those of groups and items around them."""
    items = sorted(TRUE[skill]["itemGuessSlip"])
    return {
        "prior": prior, "learn": learn, "forget": forget, "guess": guess, "slip": slip,
        "groupPriors": {"high": min(prior + 0.3, 0.9), "low": prior},
        "itemLearning": {item: {"learn": learn + shift * k, "forget": forget + shift * k / 4}
                         for k, item in enumerate(items)},
        "itemGuessSlip": {item: {"guess": guess + shift * k, "slip": slip + shift * k / 2}
                          for k, item in enumerate(items)},
    }


FIXED = {skill: rates(skill, 0.3 + 0.1 * k, 0.15, 0.03, 0.2, 0.1, 0.02) for k, skill in enumerate(SKILLS)}
STARTS = {skill: rates(skill, 0.5, 0.25 - 0.05 * k, 0.08, 0.3, 0.2, 0.015) for k, skill in enumerate(SKILLS)}
CONFIGURATIONS = [
    ("standard", {}),
    ("forgetting", {"forgetting": True}),
    ("item guess and slip", {"itemGuessSlip": True}),
    ("item learning with forgetting", {"itemLearning": True, "forgetting": True}),
    ("prior groups", {"priorGroups": True}),
    ("item guess, slip and learning with forgetting", {"itemGuessSlip": True, "itemLearning": True, "forgetting": True}),
    ("prior groups with item rates", {"priorGroups": True, "itemGuessSlip": True, "itemLearning": True,
                                      "forgetting": True}),
]


def predictions_of(options, parameters):
    """pyBKT's P(correct) and P(L) of each event at fixed parameters: the mean P(correct) of its skills and the P(L) of
    each. pyBKT combines prior groups with no item extension, so with both each group is predicted on its own with its
    prior as pyBKT's prior."""
    by_row = {}
    for skill in SKILLS:
        skill_rows = [row for row in rows if row["skill"] == skill]
        skill_parameters = parameters["skills"][skill]
        if options.get("priorGroups") and (options.get("itemGuessSlip") or options.get("itemLearning")):
            whole = pybkt_data(skill_rows, skill, {**options, "priorGroups": False})
            for group, prior in skill_parameters["groupPriors"].items():
                part = [row for row in skill_rows if row["group"] == group]
                data = pybkt_data(part, skill, {**options, "priorGroups": False}, whole["gs_names"],
                                  whole["resource_names"])
                model = pybkt_model({**skill_parameters, "prior": prior}, data, {**options, "priorGroups": False})
                by_row.update({part[k]["order"] * 2 + part[k]["position"]: v
                               for k, v in pybkt_predictions(model, data).items()})
        else:
            data = pybkt_data(skill_rows, skill, options)
            model = pybkt_model(skill_parameters, data, options)
            by_row.update({skill_rows[k]["order"] * 2 + skill_rows[k]["position"]: v
                           for k, v in pybkt_predictions(model, data).items()})
    predictions, mastery = [], []
    for order, event in enumerate(events):
        values = [by_row[order * 2 + position] for position in range(len(event["skills"]))]
        predictions.append(float(np.mean([correct for correct, _ in values])))
        mastery.append([float(known) for _, known in values])
    return predictions, mastery


inference = []
for name, options in CONFIGURATIONS:
    parameters = parameters_for(options, FIXED)
    predictions, mastery = predictions_of(options, parameters)
    inference.append({"name": name, "options": options, "parameters": parameters, "predictions": predictions,
                      "mastery": mastery})

fits = []
for name, options in CONFIGURATIONS[:-1]:
    start = parameters_for(options, STARTS)
    fitted, iterations, likelihoods = {}, [], []
    for skill in SKILLS:
        skill_rows = [row for row in rows if row["skill"] == skill]
        data = pybkt_data(skill_rows, skill, options)
        model = pybkt_model(start["skills"][skill], data, options)
        reference, _ = EM_fit.EM_fit({key: np.copy(value) for key, value in model.items()}, data, tol=TOLERANCE,
                                     maxiter=MAX_ITERATIONS, parallel=False)
        model, lls, counts = expectation_maximization(model, data)
        for key in ["learns", "forgets", "guesses", "slips"]:
            assert np.array_equal(model[key], reference[key]), (name, skill, key)
        fitted[skill] = knowledge_tracing_parameters(model, data, options, counts)
        iterations.append(len(lls))
        likelihoods.append(lls[-1])
    parameters = {"skills": fitted}
    predictions, _ = predictions_of(options, parameters)
    fits.append({
        "name": name, "options": options, "start": start, "parameters": parameters,
        "iterations": max(iterations), "skillLogLikelihoods": dict(zip(SKILLS, likelihoods)),
        "logLikelihood": sum(outcome_log_likelihood(p, e["outcome"]) for p, e in zip(predictions, events)),
    })

# pyBKT's random starts, as Model.fit draws them with seed 42 and 5 fits, run to the default tolerance 1e-6 of the package.
random = np.random.RandomState(42)
restarts = {}
for skill in SKILLS:
    data = pybkt_data([row for row in rows if row["skill"] == skill], skill, {})
    best = None
    for _ in range(5):
        model = random_model_uni.random_model_uni(1, 1, rand=random)
        model, lls, counts = expectation_maximization(model, data, 1e-6)
        if best is None or lls[-1] > best[1]:
            best = (knowledge_tracing_parameters(model, data, {}, counts), lls[-1])
    restarts[skill] = {"parameters": best[0], "logLikelihood": best[1]}

write_fixture("bkt-practice", {
    "source": "pyBKT 1.4.3: predict_onestep at fixed parameters, EM_fit from matched starts, random_model_uni starts.",
    "tolerance": TOLERANCE,
    "maxIterations": MAX_ITERATIONS,
    "events": events,
    "priorGroups": GROUPS,
    "inference": inference,
    "fits": fits,
    "restarts": restarts,
})

# Recovery: 1000 learners answer 10 items of each of two skills, interleaved, under the standard model.
RECOVERY = {"s1": {"prior": 0.25, "learn": 0.15, "forget": 0.0, "guess": 0.2, "slip": 0.1},
            "s2": {"prior": 0.5, "learn": 0.08, "forget": 0.0, "guess": 0.25, "slip": 0.15}}
recovery_rows = []
for learner in range(1000):
    for k, (skill, truth) in enumerate(RECOVERY.items()):
        known = rng.random() < truth["prior"]
        for attempt in range(10):
            correct = int(rng.random() < (1 - truth["slip"] if known else truth["guess"]))
            recovery_rows.append({"order": (attempt * 2 + k) * 1000 + learner, "learner": f"r{learner}",
                                  "skill": skill, "correct": correct, "item": skill, "group": ""})
            if not known:
                known = rng.random() < truth["learn"]
recovery_rows.sort(key=lambda row: row["order"])
true_log_likelihood = 0.0
for skill, truth in RECOVERY.items():
    skill_rows = [row for row in recovery_rows if row["skill"] == skill]
    data = pybkt_data(skill_rows, skill, {})
    for label, (correct, _) in pybkt_predictions(pybkt_model(truth, data, {}), data).items():
        true_log_likelihood += outcome_log_likelihood(correct, skill_rows[label]["correct"])
write_fixture("bkt-recovery", {
    "source": f"numpy default_rng({SEED}): BKT simulation; the log-likelihood at the true parameters by pyBKT.",
    "thresholds": {"prior": PRIOR_THRESHOLD, "rate": RATE_THRESHOLD},
    "truth": RECOVERY,
    "learners": [int(row["learner"][1:]) for row in recovery_rows],
    "skills": [int(row["skill"][1:]) - 1 for row in recovery_rows],
    "outcomes": [row["correct"] for row in recovery_rows],
    "trueLogLikelihood": true_log_likelihood,
})


def opportunities(mastery, learn, forget, threshold):
    """The opportunities until P(L), moved by the learn and forget rates alone, reaches the threshold: iterated."""
    if mastery >= threshold:
        return 0
    for n in range(1, 2_000_000):
        mastery = mastery * (1 - forget) + (1 - mastery) * learn
        if mastery >= threshold:
            return n
    return None


CASES = [(0.2, 0.1, 0.0, 0.95), (0.6, 0.25, 0.0, 0.95), (0.96, 0.1, 0.0, 0.95), (0.3, 0.0, 0.0, 0.95),
         (0.3, 0.0, 0.05, 0.95), (0.3, 1e-5, 0.0, 0.95), (0.0, 1.0, 0.0, 0.95), (0.5, 0.3, 0.05, 0.95),
         (0.5, 0.3, 0.01, 0.95), (0.1, 0.9, 0.3, 0.8), (0.1, 0.6, 0.5, 0.95), (0.4, 0.2, 0.02, 0.7),
         (0.2, 0.5, 0.5, 0.45), (0.05, 0.12, 0.0, 0.5)]
write_fixture("bkt-opportunities", {
    "source": "Python: the recurrence P(L) <- P(L) (1 - P(F)) + (1 - P(L)) P(T) iterated; null when it never reaches.",
    "cases": [{"mastery": m, "learn": l, "forget": f, "threshold": t, "opportunities": opportunities(m, l, f, t)}
              for m, l, f, t in CASES],
})
