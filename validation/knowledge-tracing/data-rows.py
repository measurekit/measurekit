"""Rows of ASSISTments and DataShop exports, and the sequences of outcomes that pyBKT's convert_data makes of them.

Seeded rows of each export have the columns that pyBKT and the recipe of docs/knowledge-tracing/data.md read. For each
skill, convert_data gives the outcomes of each learner in the order of the rows; the fixture holds them by learner. A
DataShop step with several skills has them joined by "~~"; pyBKT takes one skill per row, so the script splits the step
into one row for each skill.

Run from the repository root: validation/knowledge-tracing/.venv/bin/python validation/knowledge-tracing/data-rows.py
"""

import json

import numpy as np
import pandas as pd
from pyBKT.util import data_helper

FIXTURES = "packages/knowledge-tracing/test/fixtures"
SEED = 20261029
LEARNERS = 9
STEPS = 14
SKILLS = ["Addition", "Subtraction", "Place value", "Rounding"]


def sequences(rows, columns):
    """The outcomes of each learner, in order, for each skill of pyBKT's data: {skill: {learner: [outcome]}}."""
    frame = pd.DataFrame(rows)
    result = {}
    for skill in sorted(frame[columns["skill_name"]].unique()):
        data = data_helper.convert_data(frame.copy(), skill, defaults=dict(columns))[skill]
        learners = sorted(frame[frame[columns["skill_name"]] == skill][columns["user_id"]].unique())
        # pyBKT codes 1 incorrect and 2 correct; starts count from 1.
        outcomes = [int(value) - 1 for value in data["data"][0]]
        by_learner = {}
        for learner, start, length in zip(learners, data["starts"], data["lengths"]):
            by_learner[str(learner)] = outcomes[int(start) - 1 : int(start) - 1 + int(length)]
        result[skill] = by_learner
    return result


def main():
    random = np.random.RandomState(SEED)
    assistments = []
    shop = []
    exploded = []
    order = 0
    for learner in range(1, LEARNERS + 1):
        ability = random.normal()
        for step in range(STEPS):
            order += 1
            skills = list(random.choice(SKILLS, size=1 + int(random.uniform() < 0.3), replace=False))
            correct = int(random.uniform() < 1 / (1 + np.exp(-(ability + 0.12 * step - 0.4))))
            problem = int(random.randint(1, 6))
            # ASSISTments has one row for each skill of a problem, which the recipe reads as separate attempts.
            for skill in skills:
                assistments.append({"order_id": 2 * order, "user_id": 1000 + learner, "problem_id": problem,
                                    "skill_name": str(skill), "correct": correct})
            outcome = "correct" if correct else str(random.choice(["incorrect", "hint"]))
            shop.append({"Row": order, "Anon Student Id": f"s{learner:02d}", "Problem Name": f"P{problem}",
                         "Step Name": f"step{step % 4 + 1}", "KC (Default)": "~~".join(skills),
                         "First Attempt": outcome})
            for skill in skills:
                exploded.append({"Row": order, "Anon Student Id": f"s{learner:02d}", "KC(Default)": str(skill),
                                 "Correct First Attempt": correct})
    # A step that no skill covers has an empty KC and is no attempt at any skill.
    for learner in range(1, 4):
        order += 1
        shop.append({"Row": order, "Anon Student Id": f"s{learner:02d}", "Problem Name": "P0", "Step Name": "step0",
                     "KC (Default)": "", "First Attempt": "correct"})
    fixture = {
        "assistments": {"rows": assistments,
                        "sequences": sequences(assistments, {"order_id": "order_id", "user_id": "user_id",
                                                             "skill_name": "skill_name", "correct": "correct"})},
        "dataShop": {"rows": shop,
                     "sequences": sequences(exploded, {"order_id": "Row", "user_id": "Anon Student Id",
                                                       "skill_name": "KC(Default)",
                                                       "correct": "Correct First Attempt"})},
    }
    path = f"{FIXTURES}/data-rows.json"
    with open(path, "w") as file:
        json.dump(fixture, file, separators=(",", ":"), allow_nan=False)
    print("Wrote", path)


main()
