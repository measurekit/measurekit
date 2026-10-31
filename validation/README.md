# validation

Scripts that write the reference fixtures in `packages/<package>/test/fixtures/`. Run them from the repository root.
References, datasets, versions and tolerances are listed in
[psychometrics](../docs/psychometrics/validation.md) and [knowledge tracing](../docs/knowledge-tracing/validation.md).

## Psychometrics

R scripts in `validation/psychometrics/`:

```bash
Rscript validation/psychometrics/ctt.R               # classical test theory
Rscript validation/psychometrics/irt.R               # item response theory: calibration, JML, ability estimation, information
Rscript validation/psychometrics/cat.R               # adaptive testing: catR; diagnostic tests: cdcatR
Rscript validation/psychometrics/fit.R               # item and person fit: eRm's infit and outfit, mirt's S-X², Zh and Q3
Rscript validation/psychometrics/dif.R               # differential item functioning: difR's Mantel-Haenszel, logistic, Lord; Mantel and ordinal logistic: base R, lrm, lordif
Rscript validation/psychometrics/equating.R          # linking and equating: equateIRT, equate
Rscript validation/psychometrics/explanatory.R       # item parameters regressed on item features: mirt, metafor and base R
Rscript validation/psychometrics/generalizability.R  # variance components and decision studies of balanced designs: base R aov and lme4
Rscript validation/psychometrics/scaling.R           # score reporting and prediction: mirt, equate and base R
Rscript validation/psychometrics/standard-setting.R  # Angoff, Bookmark and Hofstee cut scores, classification accuracy: base R, mirt, cacIRT
Rscript validation/psychometrics/rater.R             # rater agreement: irr and psych; the many-facet Rasch model: base R, with TAM and sirt data
Rscript validation/psychometrics/response-time.R     # response times: nlme, LNIRT, mirt and base R
Rscript validation/psychometrics/cdm.R               # cognitive diagnosis: GDINA, NPCD and base R
Rscript validation/psychometrics/interop.R           # mirt's coefficient forms of every item type
Rscript validation/psychometrics/math.R              # normal, logistic and chi-square functions, Gauss-Hermite rules
```

They need R with the packages psych, mirt, ltm, catR, TAM, eRm, difR, lordif, MASS, rms, equate, equateIRT, irr, cacIRT,
sirt, nlme, LNIRT, numDeriv, lme4, metafor, GDINA, alabama, NPCD, CDM, cdcatR and jsonlite; `common.R` holds their shared helpers.

cdcatR imports cdmTools for its nonparametric rules only, and cdmTools needs a Rust compiler through CVXR and clarabel.
`cat.R` runs without them: download the cdcatR source with `download.packages`, remove cdmTools from `Imports` in
`DESCRIPTION` and `import(cdmTools)` from `NAMESPACE`, and run `R CMD INSTALL` on the directory.

## Knowledge tracing

R and Python scripts in `validation/knowledge-tracing/`:

```bash
Rscript validation/knowledge-tracing/math.R      # normal quantile, logistic function, forward-backward by enumeration, glm
Rscript validation/knowledge-tracing/evaluate.R  # AUC by pROC; RMSE, log-loss, accuracy, calibration and learning curves in base R
Rscript validation/knowledge-tracing/elo.R       # Elo updates written out, held-out predictions, the recovery simulation
Rscript validation/knowledge-tracing/logistic.R  # LKT's features; AFM, PFA and LKT fits by glm, lme4's Laplace approximation, LiblineaR
Rscript validation/knowledge-tracing/irt.R       # the grid posterior and exact mode of θ, against mirt's fscores
Rscript validation/knowledge-tracing/kst.R       # structures, fringes, BLIM, IITA and assessment by kst, pks and DAKS
validation/knowledge-tracing/.venv/bin/python validation/knowledge-tracing/bkt.py  # BKT by pyBKT: inference, EM from given and random starts, recovery
```

The R scripts need R with the packages pROC, lme4, LiblineaR, LKT, mirt, kst, pks, DAKS, sets and jsonlite; `common.R`
holds their shared helpers. `bkt.py` needs the Python environment below, and takes under a minute.

### Python

`bkt.py` runs in a Python 3.12 environment in `validation/knowledge-tracing/.venv`, which git ignores:

```bash
uv venv --python 3.12 validation/knowledge-tracing/.venv
uv pip install --python validation/knowledge-tracing/.venv -r validation/knowledge-tracing/requirements.txt
```

`requirements.txt` pins pyBKT with the NumPy, pandas and scikit-learn versions it works with.
