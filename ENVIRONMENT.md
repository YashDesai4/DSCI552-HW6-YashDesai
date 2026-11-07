# About the environment used for the runs

These are the versions used to produce the saved results:

| Tool | Version |
| --- | --- |
| Python | 3.12 |
| NumPy | 2.3.5 |
| pandas | 2.2.3 |
| SciPy | 1.17.0 |
| Matplotlib | 3.10.8 |
| scikit-learn | 1.8.0 |
| XGBoost CPU | 3.0.5 |
| imbalanced-learn | 0.14.0 |

Two small import-compatibility changes were needed to use this imbalanced-learn version with scikit-learn 1.8. The temporary installed copy imported `is_pandas_df` under its earlier private name, and an unrelated ensemble class stopped passing the removed `algorithm="SAMME"` argument to AdaBoost. Neither change altered SMOTE's neighbor search, randomization, interpolation or resampling steps.

A small SMOTE check confirmed that the original rows were preserved and that the expected balanced class counts were produced. A small XGBoost fit/predict check passed too. For a new local installation, `requirements.txt` keeps scikit-learn below 1.8, so those temporary import changes are not needed. Dependency versions and hardware can still cause small numerical differences when rerunning.

The forests used four worker threads. The linear boosters used one, which avoided extra OpenMP overhead. Both linear comparisons used 100 rounds, a learning rate of 0.1, and the same five CV folds and alpha grid. Seed 552 makes the random choices repeatable.

For the linear models, the preprocessing order was median filling, `log1p`, then StandardScaler. The transform was added after training-only checks found probability saturation without it. The test set was not used to choose that fix.

The final checks matched the saved probabilities to the reported confusion matrices and metrics. The selected-alpha out-of-fold errors matched the CV errors, and the notebook reporting cells ran without errors. The PDF page count is checked again whenever its wording or layout is updated.

In the corrected runs, alpha is selected only from positive values (0.0001, 0.001, 0.01, 0.1). Training and unlabeled validation/test batches use separate median imputers to follow the homework footnote. The scaler and model remain training-fitted, and SMOTE is limited to training folds. All four models and both CV searches were rerun for this protocol. The primary PDF has a checked page count; the browser copy mirrors its fixed pages rather than claiming a verified native-notebook print preview.

The compact submission notebook has a direct, complete PDF export in HW6_Notebook_Print.pdf. The export includes all cells and saved outputs; its checksum and print profile are recorded in notebook_export_check.json. This is distinct from the separate 16-page study report. Native browser pagination has not been verified because the browser policy blocks local files.
