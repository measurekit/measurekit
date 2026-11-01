"""BKT fixtures from pyBKT: inference at fixed parameters and EM from matched starts for each extension, best fits from
pyBKT's random starts, a recovery study, and opportunities to mastery written out.

Run from the repository root: validation/knowledge-tracing/.venv/bin/python validation/knowledge-tracing/bkt.py
"""

import json
import math
import re

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
    "multilearn": "resource",
    "multipair": "resource",
    "multiprior": "group",
}


def pybkt_data(rows, skill, options, gs_names=None, resource_names=None):
    """pyBKT's data of one skill: model_type is (multilearn, multiprior, multipair, multigs)."""
    model_type = (options.get("resourceLearning", False), options.get("priorGroups", False),
                  options.get("pairLearning", False), options.get("itemGuessSlip", False))
    return data_helper.convert_data(
        pd.DataFrame(rows), skill, defaults=dict(COLUMNS), model_type=model_type,
        gs_refs=None if gs_names is None else {skill: {"gs_names": gs_names}},
        resource_refs=None if resource_names is None else {skill: {"resource_names": resource_names}})[skill]


def pybkt_model(parameters, data, options):
    """pyBKT's model of measurekit's parameters of a skill. With prior groups, pyBKT's prior is 0 and each group's prior
    is the learn rate of a phantom first step."""
    resources = sorted(data["resource_names"].items(), key=lambda entry: entry[1])
    learning = []
    for name, _ in resources:
        if options.get("priorGroups") and name != "Default":
            learning.append((parameters["groupPriors"][name], 0.0))
        elif options.get("pairLearning") and name != "Default":
            # pyBKT names a pair "['current'] ['previous']".
            current, previous = re.fullmatch(r"\['(.*)'\] \['(.*)'\]", name).groups()
            rates = parameters["pairLearning"][previous][current]
            learning.append((rates["learn"], rates["forget"]))
        elif options.get("resourceLearning"):
            rates = parameters["resourceLearning"][name]
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


def expectation_maximization(model, data, tolerance=TOLERANCE, fixed=None):
    """pyBKT's EM_fit loop, keeping the expected counts of the last M-step. Returns the model, the log-likelihoods and
    the counts: transitions by resource, to, from; emissions by guess and slip class, state, outcome."""
    resources, classes = len(model["learns"]), len(model["guesses"])
    likelihoods, counts = [], None
    for i in range(MAX_ITERATIONS):
        result = EM_fit.run(data, model, None, None, None, 1, parallel=False, fixed=fixed or {})
        for j in range(resources):
            result["all_trans_softcounts"][j] = result["all_trans_softcounts"][j].transpose()
        for j in range(classes):
            result["all_emission_softcounts"][j] = result["all_emission_softcounts"][j].transpose()
        likelihoods.append(float(result["total_loglike"][0][0]))
        if i > 1 and abs(likelihoods[i] - likelihoods[i - 1]) <= tolerance:
            break
        counts = (result["all_trans_softcounts"].copy(), result["all_emission_softcounts"].copy())
        model = M_step.run(model, result["all_trans_softcounts"], result["all_emission_softcounts"],
                           result["all_initial_softcounts"], fixed=fixed or {})
    return model, likelihoods, counts


def pooled(numerator, denominator):
    return float(numerator.sum() / denominator.sum())


def knowledge_tracing_parameters(model, data, options, counts):
    """measurekit's parameters of a skill from a pyBKT fit. A rate that groups or items have of their own is, for the
    skill, the ratio of the expected counts pooled over them, from the counts of the M-step that made the fit."""
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
        if options.get("pairLearning"):
            default = names["Default"] - 1
            skill["learn"], skill["forget"] = float(model["learns"][default]), float(model["forgets"][default])
            skill["pairLearning"] = {}
            for name, index in names.items():
                if name != "Default":
                    current, previous = re.fullmatch(r"\['(.*)'\] \['(.*)'\]", name).groups()
                    skill["pairLearning"].setdefault(previous, {})[current] = {
                        "learn": float(model["learns"][index - 1]), "forget": float(model["forgets"][index - 1])}
        elif options.get("resourceLearning"):
            skill["resourceLearning"] = {name: {"learn": float(model["learns"][index - 1]),
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
    "resourceLearning": {item: {"learn": float(rng.uniform(0.05, 0.3)), "forget": float(rng.uniform(0, 0.05))}
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
            learning = TRUE[skill]["resourceLearning"][item]
            known[skill] = rng.random() >= learning["forget"] if known[skill] else rng.random() < learning["learn"]
events.sort(key=lambda event: event["time"])
rows = [{"order": order, "learner": event["learner"], "skill": skill, "correct": event["outcome"],
         "item": event["item"], "resource": event["item"], "group": GROUPS[event["learner"]], "event": order,
         "position": position}
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
    """measurekit's parameters with the rates that `options` uses, from per-skill values of every rate."""
    skills = {}
    for skill in SKILLS:
        v = values[skill]
        entry = {"prior": v["prior"], "learn": v["learn"], "forget": v["forget"] if options.get("forgetting") else 0.0,
                 "guess": v["guess"], "slip": v["slip"]}
        if options.get("priorGroups"):
            entry["groupPriors"] = v["groupPriors"]
        if options.get("resourceLearning"):
            entry["resourceLearning"] = {item: {"learn": r["learn"], "forget": r["forget"] if options.get("forgetting")
                                            else 0.0} for item, r in v["resourceLearning"].items()}
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
        "resourceLearning": {item: {"learn": learn + shift * k, "forget": forget + shift * k / 4}
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
    ("resource learning with forgetting", {"resourceLearning": True, "forgetting": True}),
    ("prior groups", {"priorGroups": True}),
    ("item guess, slip and learning with forgetting", {"itemGuessSlip": True, "resourceLearning": True, "forgetting": True}),
    ("prior groups with item rates", {"priorGroups": True, "itemGuessSlip": True, "resourceLearning": True,
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
        if options.get("priorGroups") and (options.get("itemGuessSlip") or options.get("resourceLearning")):
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

# pyBKT's random starts, as Model.fit draws them with seed 42 and 5 fits, run to measurekit's default tolerance 1e-6.
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
                                  "skill": skill, "correct": correct, "item": skill, "resource": skill, "group": ""})
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


# Events without an outcome: 150 learners take 14 steps at one skill, each a lesson, which pyBKT records as correct = -1,
# or an attempt at one of 3 items. Lessons and practice have learn rates of their own; items have guess and slip.
LESSON_RATES = {"prior": 0.3, "resources": {"lesson": (0.35, 0.02), "practice": (0.1, 0.02)},
                "items": {"p1": (0.25, 0.1), "p2": (0.15, 0.2), "p3": (0.2, 0.15)}}
LESSON_ITEM = "lesson"
lesson_rng = np.random.default_rng(SEED + 1)
lesson_events = []
for index in range(150):
    known = lesson_rng.random() < LESSON_RATES["prior"]
    for step in range(14):
        is_lesson = lesson_rng.random() < 0.3
        item = LESSON_ITEM if is_lesson else str(lesson_rng.choice(sorted(LESSON_RATES["items"])))
        resource = "lesson" if is_lesson else "practice"
        outcome = None
        if not is_lesson:
            guess, slip = LESSON_RATES["items"][item]
            outcome = int(lesson_rng.random() < (1 - slip if known else guess))
        lesson_events.append({"learner": f"M{index:02d}", "item": item, "resource": resource, "skills": ["s"],
                              "time": step * 100 + index, "outcome": outcome})
        learn, forget = LESSON_RATES["resources"][resource]
        known = lesson_rng.random() >= forget if known else lesson_rng.random() < learn
lesson_events.sort(key=lambda event: event["time"])
lesson_rows = [{"order": order, "learner": event["learner"], "skill": "s",
                "correct": -1 if event["outcome"] is None else event["outcome"], "item": event["item"],
                "resource": event["resource"], "group": ""} for order, event in enumerate(lesson_events)]
# pyBKT rates every resource without a following attempt as learned for sure, so each resource must be followed by an
# outcome somewhere, and every learner needs at least 2 steps.
followed = {row["resource"] for row in lesson_rows if any(
    later["learner"] == row["learner"] and later["order"] > row["order"] and later["correct"] != -1
    for later in lesson_rows)}
assert followed == {"lesson", "practice"}, followed

LESSON_CONFIGURATIONS = [
    ("lessons and practice", {"resourceLearning": True, "itemGuessSlip": True}),
    ("lessons and practice with forgetting", {"resourceLearning": True, "itemGuessSlip": True, "forgetting": True}),
    ("one learn rate", {}),
]


def lesson_parameters(options, prior, learn, forget, guess, slip, shift, with_lesson_item=True):
    """measurekit's parameters of the skill: the rates, those of the resources and of the items around them."""
    skill = {"prior": prior, "learn": learn, "forget": forget if options.get("forgetting") else 0.0,
             "guess": guess, "slip": slip}
    if options.get("resourceLearning"):
        skill["resourceLearning"] = {name: {"learn": learn + shift * k, "forget": skill["forget"]}
                                     for k, name in enumerate(["lesson", "practice"])}
    if options.get("itemGuessSlip"):
        names = sorted(LESSON_RATES["items"]) + ([LESSON_ITEM] if with_lesson_item else [])
        skill["itemGuessSlip"] = {name: {"guess": guess + shift * k, "slip": slip + shift * k / 2}
                                  for k, name in enumerate(names)}
    return skill


def lesson_model(options, skill):
    """pyBKT's model and data of the lesson log for measurekit's parameters of the skill."""
    data = pybkt_data(lesson_rows, "s", options)
    return pybkt_model(skill, data, options), data


lesson_inference, lesson_fits = [], []
for name, options in LESSON_CONFIGURATIONS:
    fixed = lesson_parameters(options, 0.3, 0.2, 0.03, 0.2, 0.1, 0.05)
    model, data = lesson_model(options, fixed)
    by_row = pybkt_predictions(model, data)
    inference_skill = {key: value for key, value in fixed.items()}
    if options.get("itemGuessSlip"):
        inference_skill["itemGuessSlip"] = {item: rates for item, rates in fixed["itemGuessSlip"].items()
                                            if item != LESSON_ITEM}
    lesson_inference.append({
        "name": name, "options": options, "parameters": {"skills": {"s": inference_skill}},
        "predictions": [None if row["correct"] == -1 else float(by_row[order][0])
                        for order, row in enumerate(lesson_rows)],
        "mastery": [float(by_row[order][1]) for order in range(len(lesson_rows))],
    })

    start = lesson_parameters(options, 0.5, 0.3, 0.08, 0.3, 0.2, 0.03)
    model, data = lesson_model(options, start)
    reference, _ = EM_fit.EM_fit({key: np.copy(value) for key, value in model.items()}, data, tol=TOLERANCE,
                                 maxiter=MAX_ITERATIONS, parallel=False)
    model, lls, counts = expectation_maximization(model, data)
    for key in ["learns", "forgets", "guesses", "slips"]:
        assert np.array_equal(model[key], reference[key]), (name, key)
    fitted = knowledge_tracing_parameters(model, data, options, counts)
    if options.get("itemGuessSlip"):
        fitted["itemGuessSlip"] = {item: rates for item, rates in fitted["itemGuessSlip"].items()
                                   if item != LESSON_ITEM}
        start["itemGuessSlip"] = {item: rates for item, rates in start["itemGuessSlip"].items()
                                  if item != LESSON_ITEM}
    fitted_model = pybkt_model({**fitted, **({"itemGuessSlip": {**fitted["itemGuessSlip"],
                                                                 LESSON_ITEM: {"guess": 0.5, "slip": 0.5}}}
                                             if options.get("itemGuessSlip") else {})}, data, options)
    by_row = pybkt_predictions(fitted_model, data)
    lesson_fits.append({
        "name": name, "options": options, "start": {"skills": {"s": start}},
        "parameters": {"skills": {"s": fitted}}, "iterations": len(lls),
        "logLikelihood": sum(outcome_log_likelihood(by_row[order][0], row["correct"])
                             for order, row in enumerate(lesson_rows) if row["correct"] != -1),
    })

write_fixture("bkt-lessons", {
    "source": "pyBKT 1.4.3: correct = -1 for a lesson, multilearn as resource, multigs as item; predict_onestep at "
              "fixed parameters and EM_fit from matched starts.",
    "tolerance": TOLERANCE,
    "maxIterations": MAX_ITERATIONS,
    "truth": {"skills": {"s": {"prior": LESSON_RATES["prior"],
                               "resourceLearning": {name: {"learn": learn, "forget": forget}
                                                    for name, (learn, forget) in LESSON_RATES["resources"].items()}}}},
    "events": lesson_events,
    "inference": lesson_inference,
    "fits": lesson_fits,
})


# Fixed rates: pyBKT's `fixed`, where an array holds a value for each resource or item class and a negative one leaves
# it free. Standard configurations hold a skill's learn rate, or its guess and slip, on the practice log; the lessons'
# log holds the learn rate of the lesson while the practice's is estimated.
fixed_fits = []
FIXED_SKILL = {"learn": 0.12, "guess": 0.25, "slip": 0.08}
for name, options, fixed_of in [
    ("fixed learn", {}, lambda data: {"learns": np.full(len(data["resource_names"]), FIXED_SKILL["learn"])}),
    ("fixed guess and slip", {}, lambda data: {"guesses": np.full(len(data["gs_names"]), FIXED_SKILL["guess"]),
                                              "slips": np.full(len(data["gs_names"]), FIXED_SKILL["slip"])}),
]:
    start = parameters_for(options, STARTS)
    fitted, iterations = {}, []
    for skill in SKILLS:
        skill_rows = [row for row in rows if row["skill"] == skill]
        data = pybkt_data(skill_rows, skill, options)
        fixed = fixed_of(data)
        model = pybkt_model(start["skills"][skill], data, options)
        reference, _ = EM_fit.EM_fit({key: np.copy(value) for key, value in model.items()}, data, tol=TOLERANCE,
                                     maxiter=MAX_ITERATIONS, parallel=False, fixed=fixed)
        model, lls, counts = expectation_maximization(model, data, fixed=fixed)
        for key in ["learns", "forgets", "guesses", "slips"]:
            assert np.array_equal(model[key], reference[key]), (name, skill, key)
        fitted[skill] = knowledge_tracing_parameters(model, data, options, counts)
        iterations.append(len(lls))
    parameters = {"skills": fitted}
    predictions, _ = predictions_of(options, parameters)
    held = {"fixed learn": {"learn": FIXED_SKILL["learn"]},
            "fixed guess and slip": {"guess": FIXED_SKILL["guess"], "slip": FIXED_SKILL["slip"]}}[name]
    fixed_fits.append({
        "name": name, "options": options, "log": "practice", "fixed": {skill: held for skill in SKILLS},
        "start": start, "parameters": parameters, "iterations": max(iterations),
        "logLikelihood": sum(outcome_log_likelihood(p, e["outcome"]) for p, e in zip(predictions, events)),
    })

# The lessons' log with the lesson's learn rate held at 0.3.
HELD_LESSON = 0.3
options = {"resourceLearning": True, "itemGuessSlip": True}
start = lesson_parameters(options, 0.5, 0.3, 0.08, 0.3, 0.2, 0.03)
data = pybkt_data(lesson_rows, "s", options)
fixed = {"learns": np.array([HELD_LESSON if name == "lesson" else -1.0
                            for name, _ in sorted(data["resource_names"].items(), key=lambda entry: entry[1])])}
model = pybkt_model(start, data, options)
reference, _ = EM_fit.EM_fit({key: np.copy(value) for key, value in model.items()}, data, tol=TOLERANCE,
                             maxiter=MAX_ITERATIONS, parallel=False, fixed=fixed)
model, lls, counts = expectation_maximization(model, data, fixed=fixed)
for key in ["learns", "forgets", "guesses", "slips"]:
    assert np.array_equal(model[key], reference[key]), ("held lesson", key)
fitted = knowledge_tracing_parameters(model, data, options, counts)
fitted["itemGuessSlip"] = {item: rates for item, rates in fitted["itemGuessSlip"].items() if item != LESSON_ITEM}
start["itemGuessSlip"] = {item: rates for item, rates in start["itemGuessSlip"].items() if item != LESSON_ITEM}
fitted_model = pybkt_model({**fitted, "itemGuessSlip": {**fitted["itemGuessSlip"], LESSON_ITEM: {"guess": 0.5, "slip": 0.5}}},
                           data, options)
by_row = pybkt_predictions(fitted_model, data)
fixed_fits.append({
    "name": "a lesson held, practice estimated", "options": options, "log": "lessons",
    "fixed": {"s": {"resources": {"lesson": {"learn": HELD_LESSON}}}},
    "start": {"skills": {"s": start}}, "parameters": {"skills": {"s": fitted}}, "iterations": len(lls),
    "logLikelihood": sum(outcome_log_likelihood(by_row[order][0], row["correct"])
                         for order, row in enumerate(lesson_rows) if row["correct"] != -1),
})
write_fixture("bkt-fixed", {
    "source": "pyBKT 1.4.3: EM_fit from matched starts with `fixed`, M_step holding the rates of the classes it names.",
    "tolerance": TOLERANCE,
    "maxIterations": MAX_ITERATIONS,
    "fits": fixed_fits,
})

# Pairs: 120 learners take 12 steps at one skill, each at one of 4 items; the rate after an attempt depends on the item
# before it and its own. pyBKT's `multipair` has 17 resources: the 16 pairs and Default, a learner's first attempt.
PAIR_ITEMS = ["i1", "i2", "i3", "i4"]
pair_rng = np.random.default_rng(SEED + 2)
PAIR_TRUE = {"prior": 0.25,
             "pairs": {previous: {current: (0.05 + 0.04 * ((a + b) % 4), 0.01 * ((a * b) % 3))
                                  for b, current in enumerate(PAIR_ITEMS)} for a, previous in enumerate(PAIR_ITEMS)},
             "default": (0.2, 0.02), "items": {"i1": (0.25, 0.1), "i2": (0.15, 0.2), "i3": (0.2, 0.15),
                                               "i4": (0.3, 0.1)}}
pair_events = []
for index in range(120):
    known = pair_rng.random() < PAIR_TRUE["prior"]
    previous = None
    for step in range(12):
        item = str(pair_rng.choice(PAIR_ITEMS))
        guess, slip = PAIR_TRUE["items"][item]
        pair_events.append({"learner": f"P{index:03d}", "item": item, "skills": ["p"], "time": step * 1000 + index,
                            "outcome": int(pair_rng.random() < (1 - slip if known else guess))})
        learn, forget = PAIR_TRUE["default"] if previous is None else PAIR_TRUE["pairs"][previous][item]
        known = pair_rng.random() >= forget if known else pair_rng.random() < learn
        previous = item
pair_events.sort(key=lambda event: event["time"])
pair_rows = [{"order": order, "learner": event["learner"], "skill": "p", "correct": event["outcome"],
              "item": event["item"], "resource": event["item"], "group": ""} for order, event in enumerate(pair_events)]
pair_options = {"pairLearning": True, "itemGuessSlip": True, "forgetting": True}
pair_data = pybkt_data(pair_rows, "p", pair_options)
assert len(pair_data["resource_names"]) == 17 and list(pair_data["resource_names"].items())[0] == ("Default", 1)
# pyBKT rates every resource without a following attempt as learned for sure: each pair must be followed.
last_of_learner = {}
for row in pair_rows:
    last_of_learner[row["learner"]] = row["order"]
previous_item, seen_pairs = {}, set()
for row in pair_rows:
    if row["order"] != last_of_learner[row["learner"]]:
        before = previous_item.get(row["learner"])
        seen_pairs.add("Default" if before is None else (before, row["item"]))
    previous_item[row["learner"]] = row["item"]
assert len(seen_pairs) == 17, len(seen_pairs)


def pair_parameters(prior, learn, forget, guess, slip, shift):
    return {"prior": prior, "learn": learn, "forget": forget, "guess": guess, "slip": slip,
            "pairLearning": {previous: {current: {"learn": learn + shift * ((a + b) % 4), "forget": forget + shift * (b % 2) / 4}
                                        for b, current in enumerate(PAIR_ITEMS)}
                             for a, previous in enumerate(PAIR_ITEMS)},
            "itemGuessSlip": {item: {"guess": guess + shift * k, "slip": slip + shift * k / 2}
                              for k, item in enumerate(PAIR_ITEMS)}}


pair_start = pair_parameters(0.5, 0.25, 0.08, 0.3, 0.2, 0.01)
pair_fixed_point = pair_parameters(0.3, 0.15, 0.03, 0.2, 0.1, 0.02)
pair_model = pybkt_model(pair_fixed_point, pair_data, pair_options)
pair_by_row = pybkt_predictions(pair_model, pair_data)
model = pybkt_model(pair_start, pair_data, pair_options)
reference, _ = EM_fit.EM_fit({key: np.copy(value) for key, value in model.items()}, pair_data, tol=TOLERANCE,
                             maxiter=MAX_ITERATIONS, parallel=False)
model, lls, counts = expectation_maximization(model, pair_data)
for key in ["learns", "forgets", "guesses", "slips"]:
    assert np.array_equal(model[key], reference[key]), ("pairs", key)
pair_fitted = knowledge_tracing_parameters(model, pair_data, pair_options, counts)
fitted_by_row = pybkt_predictions(pybkt_model(pair_fitted, pair_data, pair_options), pair_data)
write_fixture("bkt-pairs", {
    "source": "pyBKT 1.4.3: multipair, 17 resources on 4 items with Default first; predict_onestep at fixed parameters "
              "and EM_fit from a matched start.",
    "tolerance": TOLERANCE,
    "maxIterations": MAX_ITERATIONS,
    "options": pair_options,
    "events": pair_events,
    "inference": {"parameters": {"skills": {"p": pair_fixed_point}},
                  "predictions": [float(pair_by_row[order][0]) for order in range(len(pair_rows))],
                  "mastery": [float(pair_by_row[order][1]) for order in range(len(pair_rows))]},
    "fit": {"start": {"skills": {"p": pair_start}}, "parameters": {"skills": {"p": pair_fitted}},
            "iterations": len(lls),
            "logLikelihood": sum(outcome_log_likelihood(fitted_by_row[order][0], row["correct"])
                                 for order, row in enumerate(pair_rows))},
})
