"""Writes rater-automated-scoring.json: the evaluation of automated scores by RSMTool 12.0.0, the reference of the
recipe "Evaluating automated scores" in rater.md and of standardizedMeanDifference's population.

Data, both CC BY 4.0, fetched once into validation/psychometrics/.data, which git ignores, and checked by SHA-256:
- PERSUADE 2.0's training set: the holistic scores (1 to 6) and race or ethnicity of the 1,818 essays on "Does the
  electoral college work?", one row per essay;
- ELLIPSE's raw rater scores: the two raters' overall scores (1 to 5) of every essay both scored above 0, as RSMTool's
  exclude_zero_scores drops a score of 0.
Only scores and groups enter the fixture, never text. The machine scores are a stand-in for a scorer: the least-squares
line of the first human score on the essay's length in words, fitted by numpy on the same essays, then scaled to the
human scores' mean and SD, as RSMTool's scaled predictions are, and written unrounded; the recipe rounds them and trims
them to the scale. The script checks that no prediction lies within 1e-6 of a half,
so that rounding half to even, as numpy does, and half up, as JavaScript does, agree.

Expected values come from RSMTool's own functions: Analyzer.metrics_helper for each pair of scores, with the SMD pooled
over the whole sample and Williamson's for each group on the population's SDs; Analyzer.compute_metrics for the
human-human row and compute_degradation_and_disattenuated_correlations for the degradation and the disattenuated
correlation. RSMTool's degradation of the SMD takes the human-system SMD unpooled, its default, and the human-human SMD
pooled, so the fixture holds both pooled SMDs and the recipe's degradation is their difference.

Run from the repository root, in the environment of automated-scoring-requirements.txt:
validation/psychometrics/.venv-scoring/bin/python -I validation/psychometrics/automated-scoring.py
"""

import hashlib
import io
import json
import math
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from rsmtool.analyzer import Analyzer
from rsmtool.utils.metrics import standardized_mean_difference

DATA = Path("validation/psychometrics/.data")
FIXTURE = Path("packages/psychometrics/test/fixtures/rater-automated-scoring.json")
PERSUADE = {
    "url": "https://drive.usercontent.google.com/download?id=13phHyDzIsb0MHyJr6q-B-qIa9P2tM135&export=download&confirm=t",
    "file": "persuade_2.0_train.csv",
    "sha256": "f61319edd8bf16a982711ea0399fad59c05afaec05cdf0767f16a2c05c467e23",
}
ELLIPSE = {
    "url": "https://raw.githubusercontent.com/scrosseye/ELLIPSE-Corpus/dc3b8f0b3b4332fc9f64302c4ccfc4ed582f4b43/"
    "ellipsis_raw_rater_scores_anon_all_essay.zip",
    "file": "ellipsis_raw_rater_scores_anon_all_essay.zip",
    "sha256": "f9c3cede2d54144225e0651af6415137f7da07f2d2955e8f8587896c293d5e85",
}
# The password the ELLIPSE repository's README gives for the zip of raw rater scores.
ELLIPSE_PASSWORD = b"ellipse_raw_data"
PROMPT = "Does the electoral college work?"
# How far from a half a prediction must lie for rounding half to even and half up to agree.
HALF_MARGIN = 1e-6
# RSMTool's names of its metrics, and the fixture's; agreement stays in percent, as RSMTool gives it.
METRICS = {
    "kappa": "kappa",
    "wtkappa": "quadraticKappa",
    "exact_agr": "exactPercent",
    "adj_agr": "adjacentPercent",
    "SMD": "smd",
    "corr": "correlation",
    "R2": "rSquared",
    "RMSE": "rmse",
    "sys_sd": "machineSd",
    "h_sd": "humanSd",
}
DEGRADED = ["corr", "kappa", "wtkappa", "exact_agr", "adj_agr"]


def fetch(source: dict) -> Path:
    """Downloads a file once into DATA and checks its SHA-256."""
    path = DATA / source["file"]
    if not path.exists():
        DATA.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(source["url"]) as response, open(path, "wb") as out:
            while chunk := response.read(1 << 20):
                out.write(chunk)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == source["sha256"], f"{path} has SHA-256 {digest}, expected {source['sha256']}"
    return path


def machine_scores(human: np.ndarray, words: np.ndarray, low: int, high: int) -> tuple[np.ndarray, np.ndarray]:
    """Returns the scaled least-squares predictions of the human scores from the length, and those rounded and trimmed."""
    slope, intercept = np.polyfit(words, human, 1)
    fitted = intercept + slope * words
    predicted = np.mean(human) + (fitted - np.mean(fitted)) * np.std(human, ddof=1) / np.std(fitted, ddof=1)
    assert np.min(np.abs(predicted - np.floor(predicted) - 0.5)) > HALF_MARGIN
    return predicted, np.clip(np.rint(predicted), low, high)


def number(value: float):
    return None if value is None or (isinstance(value, float) and math.isnan(value)) else float(value)


def metrics(series: pd.Series) -> dict:
    return {key: number(series[name]) for name, key in METRICS.items()}


def persuade_case() -> dict:
    columns = ["essay_id_comp", "holistic_essay_score", "prompt_name", "race_ethnicity", "essay_word_count"]
    essays = pd.read_csv(fetch(PERSUADE), usecols=columns).drop_duplicates("essay_id_comp")
    essays = essays[essays.prompt_name == PROMPT].reset_index(drop=True)
    human = essays.holistic_essay_score.to_numpy(float)
    predicted, machine = machine_scores(human, essays.essay_word_count.to_numpy(float), 1, 6)
    overall = Analyzer.metrics_helper(pd.Series(human), pd.Series(machine), smd_method="pooled")
    assert abs(overall["SMD"] - standardized_mean_difference(human, machine, method="pooled")) < 1e-15
    human_sd = np.std(human, ddof=1)
    machine_sd = np.std(machine, ddof=1)
    groups = []
    for group in sorted(essays.race_ethnicity.unique()):
        rows = (essays.race_ethnicity == group).to_numpy()
        values = Analyzer.metrics_helper(
            pd.Series(human[rows]), pd.Series(machine[rows]), human_sd, machine_sd, smd_method="williamson"
        )
        groups.append({"group": group, "subjects": int(values["N"]), "smd": number(values["SMD"])})
    return {
        "scale": [1, 6],
        "human": human.astype(int).tolist(),
        "machine": predicted.tolist(),
        "groups": essays.race_ethnicity.tolist(),
        "overall": metrics(overall),
        "byGroup": groups,
    }


def ellipse_case() -> dict:
    with zipfile.ZipFile(fetch(ELLIPSE)) as archive:
        name = "ellipsis_raw_rater_scores_anon_all_essay.csv"
        raw = pd.read_csv(io.BytesIO(archive.read(name, pwd=ELLIPSE_PASSWORD)))
    raw = raw[(raw.Overall_1 > 0) & (raw.Overall_2 > 0)].reset_index(drop=True)
    first = raw.Overall_1.to_numpy(float)
    second = raw.Overall_2.to_numpy(float)
    predicted, machine = machine_scores(first, raw.Text.str.split().str.len().to_numpy(float), 1, 5)
    frame = pd.DataFrame({"sc1": first, "sc2": second, "raw_trim_round": machine})
    analyzer = Analyzer()
    degradation, correlations = analyzer.compute_degradation_and_disattenuated_correlations(frame)
    _, _, human_human = analyzer.compute_metrics(frame, include_second_score=True)
    human_system = Analyzer.metrics_helper(pd.Series(first), pd.Series(machine), smd_method="pooled")
    human_pair = Analyzer.metrics_helper(pd.Series(first), pd.Series(second), smd_method="pooled")
    for metric in DEGRADED:
        assert abs(human_pair[metric] - human_human[metric].iloc[0]) < 1e-15
    return {
        "scale": [1, 5],
        "first": first.astype(int).tolist(),
        "second": second.astype(int).tolist(),
        "machine": predicted.tolist(),
        "humanSystem": metrics(human_system),
        "humanHuman": metrics(human_pair),
        "degradation": {METRICS[metric]: number(degradation[metric].loc["raw_trim_round"]) for metric in DEGRADED},
        "disattenuated": number(correlations["corr_disattenuated"].loc["raw_trim_round"]),
    }


fixture = {
    "source": (
        "RSMTool 12.0.0's Analyzer on PERSUADE 2.0 (one prompt, by race or ethnicity) and ELLIPSE's raw rater scores, "
        "with machine scores from a least-squares line on essay length; agreement in percent, as RSMTool gives it."
    ),
    "cases": {"PERSUADE 2.0": persuade_case(), "ELLIPSE": ellipse_case()},
}
FIXTURE.write_text(json.dumps(fixture, separators=(",", ":")) + "\n")
print(f"Wrote {FIXTURE}")
