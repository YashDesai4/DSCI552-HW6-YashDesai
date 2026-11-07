# DSCI 552 - Homework 6

`Yash_Desai_HW6.ipynb` is the compact submission notebook. It keeps the required answers, model-fitting code, saved results and 11 figures. The longer study notebook is preserved separately, outside this submission ZIP.

## The notebook's own page-count check

`results/HW6_Notebook_Print.pdf` is a complete export of the current notebook: all 60 cells, 287 code lines, 31 saved output blocks and 11 figures. No code cells or output blocks are hidden in that export. It is **16 US Letter pages**, using 10-point body text and 8.5-point code.

This count is for the notebook export, not the separate study report. `results/notebook_export_check.json` records the exact notebook checksum, page count and included-content counts.

The `.ipynb` format has no intrinsic page count. GitHub, Jupyter and browser styles can paginate it differently. Local-file browser access is blocked in the preparation environment, so a native browser preview has not been verified here. Use the supplied notebook PDF or its fixed-page browser copy for the reproducible 16-page print layout. If your grader specifically prints the native GitHub/Jupyter view, check that preview on your computer before submission.

`Yash_Desai_HW6_print.html` mirrors the complete notebook export. Open it in a browser and print on US Letter at 100% scale with headers and footers off. It contains the same 16 fixed pages. It is image-based; the PDF provides selectable text and the notebook provides editable code.

## Reading and running the notebook

Extract the ZIP and open the project folder in VS Code. Keep the `data/` and `results/` folders alongside the notebook.

Leave `RUN_FULL_ANALYSIS=False` while reviewing the saved model results. The lighter preparation and reporting cells still run. Set it to `True` to repeat the full model fits and cross-validation. The training, CV, SMOTE and evaluation code remain inline. Only optional conceptual-figure helper definitions were moved to `learning_plots.py` to reduce repetition.

After saving notebook changes, regenerate its print export with:

```bash
python notebook_print.py
python export_notebook.py
```

## Repeating the full analysis

Use Python 3.11 or 3.12. In the VS Code terminal:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python analysis.py
```

`analysis.py` is an optional script version of the core analysis. The notebook does not import it. Full-data training and SMOTE can take time; run one training workflow at a time and keep the computer awake until it completes. The original CSVs are included.

## What the files contain

| File | Purpose |
| --- | --- |
| `Yash_Desai_HW6.ipynb` | Compact notebook with answers, core code and saved outputs |
| `results/HW6_Notebook_Print.pdf` | Complete, checked 16-page notebook export |
| `Yash_Desai_HW6_print.html` | Fixed-page browser copy of that notebook export |
| `notebook_print.py` | Exports every notebook cell and output, and checks the page count |
| `export_notebook.py` | Rebuilds the browser copy from the current notebook export |
| `analysis.py` | Optional core-analysis script |
| `learning_plots.py` | Source for the optional learning figures in Questions 2-4 |
| `report.py`, `theory.json`, `results/HW6_Results.pdf` | Separate study-report generator, written solutions and report |
| `data/` | Official CSVs, original dataset description and source checksums |
| `results/results.json` | Model metrics and selected hyperparameters |
| `results/feature_CV_all_170.csv` | CV calculations for every predictor |
| `results/correlations_all_170.csv` | Complete correlation matrix |
| Other files in `results/` | Figures, CV tables and prediction arrays |
| `ENVIRONMENT.md` | Package versions and execution notes |

## Methods and remaining qualification

All 60,000 training rows, 16,000 test rows and 170 predictors are used. The top 13 CV features are chosen only for exploratory plots. Missing values are filled using separate training and unlabeled validation/test batch medians, following the homework footnote. Scalers and models are training-fitted; SMOTE operates only on training folds. Final SMOTE produces 118,000 fitting examples, but training metrics use the original rows.

Positive alpha values 0.0001, 0.001, 0.01 and 0.1 are compared by shuffled, stratified five-fold CV (seed 552), using error first and AUC for ties. Both selected alphas are 0.0001. Selected-alpha CV is not nested CV, and forest OOB scores are conditional on the full-training imputer.

The model-definition issue in 1(e)-(f) remains explicit: XGBoost's `gblinear` fits a global L1-logistic model, not logistic models at individual tree nodes. The work does not relabel it as a literal node-wise model tree. Optional Question 6 is excluded.

Questions 2-3 have clearly labeled schematic illustrations. Question 4 uses the exact exercise probabilities. These learning figures do not add experimental APS results.
