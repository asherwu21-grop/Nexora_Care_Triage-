# Ranked development shortlist



Validation-only development ranking; experimental output, not a clinical decision. Many comparisons reused validation, so these are selection results, not unbiased generalisation estimates. The reserved test set was not transformed/evaluated. Earlier full-cohort exploration remains a limitation.



Ranking is prespecified: .30 High recall + .30 Critical recall − .15 under-triage − .30 severe under-triage − .10 over-triage + .20 macro F1 − .10 multiclass Brier − .10 patient-bootstrap High/Critical recall SD. These weights are illustrative development preferences, not elicited clinical utilities.



1. **default_LightGBM_class_weights** — LightGBM, class_weights; High/Critical recall 67.0%/79.8%; under-triage 14.6%, severe 0.8%, over-triage 13.8%; macro F1 0.713; Brier 0.414; patient-bootstrap HC recall SD 0.026, conditional interval [0.682, 0.782]. Forward-CV HC recall SD 0.031 (Default uncalibrated base estimator). Its position reflects the stated recall/error trade-off, followed by F1, probability quality and conditional validation stability.

2. **calibrated_LogisticRegression_isotonic_cost2** — LogisticRegression, class_weights; High/Critical recall 67.8%/83.0%; under-triage 9.2%, severe 0.4%, over-triage 25.6%; macro F1 0.656; Brier 0.414; patient-bootstrap HC recall SD 0.026, conditional interval [0.707, 0.805]. Forward-CV HC recall SD 0.009 (Tuned base estimator before calibration/decision rules). Its position reflects the stated recall/error trade-off, followed by F1, probability quality and conditional validation stability.

3. **default_LightGBM_random_oversampling** — LightGBM, random_oversampling; High/Critical recall 67.8%/76.6%; under-triage 15.0%, severe 1.0%, over-triage 13.5%; macro F1 0.713; Brier 0.415; patient-bootstrap HC recall SD 0.028, conditional interval [0.674, 0.780]. This exact configuration was not forward-CV evaluated; stability here is conditional bootstrap only. Its position reflects the stated recall/error trade-off, followed by F1, probability quality and conditional validation stability.

4. **calibrated_LogisticRegression_sigmoid_threshold** — LogisticRegression, class_weights; High/Critical recall 61.2%/85.1%; under-triage 12.0%, severe 0.8%, over-triage 20.2%; macro F1 0.663; Brier 0.415; patient-bootstrap HC recall SD 0.025, conditional interval [0.685, 0.780]. Forward-CV HC recall SD 0.009 (Tuned base estimator before calibration/decision rules). Its position reflects the stated recall/error trade-off, followed by F1, probability quality and conditional validation stability.

5. **uncalibrated_ensemble_LogisticRegression** — LogisticRegression, class_weights; High/Critical recall 60.1%/83.0%; under-triage 13.3%, severe 0.9%, over-triage 15.8%; macro F1 0.694; Brier 0.427; patient-bootstrap HC recall SD 0.025, conditional interval [0.667, 0.766]. Forward-CV HC recall SD 0.009 (Tuned base estimator before calibration/decision rules). Its position reflects the stated recall/error trade-off, followed by F1, probability quality and conditional validation stability.



## Tuned versus default



- LightGBM: validation macro F1 0.713 default vs 0.681 tuned; HC recall 0.734 vs 0.719. Search selection used the training CV objective; validation improvements are not assumed.

- LogisticRegression: validation macro F1 0.689 default vs 0.687 tuned; HC recall 0.703 vs 0.699. Search selection used the training CV objective; validation improvements are not assumed.



## Selection limits



Grouped forward-CV means/SDs for tuned and matching default candidates are in tuning_comparison.csv. Bootstrap intervals resample validation patients for fixed predictions; they do not measure refit/seed instability or correct for model selection. Related calibrated and decision-rule rows share fitted models and are not independent replications. Cost/threshold decisions leave Brier unchanged because they do not change probabilities. Reported Brier sums squared class errors (range zero to two); per-class and mean-OVR scores are also logged. Sigmoid/isotonic ensembles use only forward-fold training/calibration data; matching uncalibrated ensemble controls isolate the calibration change.

SMOTE interpolates already encoded features, including ordinal and binary values; fractional synthetic values limit its interpretation. Resampling is always inside the fit pipeline. Class weighting uses fold-local multiclass sample weights, including XGBoost; no binary-only scale_pos_weight shortcut.

Isotonic can be unstable in sparse probability bins. Inspect the reliability curves and bin counts rather than interpreting one Brier value as proof of calibration. All costs/thresholds require independent domain validation before use.
