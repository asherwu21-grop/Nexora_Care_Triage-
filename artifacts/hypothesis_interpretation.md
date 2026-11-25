# Hypothesis-testing interpretation



Alpha: 0.05. Full cohort: 14,809 requests; sensitivity: 9,792 patients, one request each.



Full-cohort p-values are nominal because repeated requests are dependent. Kruskal-Wallis does not repair that violation. The sensitivity results are the more defensible comparison for within-patient repetition, conditional on this one seeded selection; patient-level confounding and other dependence remain.



## Nine primary comparisons



- **Age at request**: Kruskal-Wallis, n=9,792; Holm p=2.265e-09; rank_epsilon_squared=0.0049; reject_H0. Evidence of an association/difference after omnibus Holm correction. Observed rank_epsilon_squared=0.0049. This corresponds to 0.49% of pooled rank variation. Group medians span 31.92–39.80 years. One request per patient addresses within-patient repetition. Statistical significance does not establish practical/clinical importance; no practical threshold was prespecified. Association does not imply causation.

- **Recorded sex**: Chi-square of independence, n=9,792; Holm p=1; Cramers_V=0.0046; do_not_reject_H0. Insufficient evidence after omnibus Holm correction; equality is not established. Observed Cramers_V=0.0046. Cramer's V is on an association scale from zero toward one, not a risk difference. One request per patient addresses within-patient repetition. Statistical significance does not establish practical/clinical importance; no practical threshold was prespecified. Association does not imply causation.

- **Chief complaint**: Chi-square (conditional Monte Carlo), n=9,792; Holm p=0.00035; Cramers_V=0.2111; reject_H0. Evidence of an association/difference after omnibus Holm correction. Observed Cramers_V=0.2111. Cramer's V is on an association scale from zero toward one, not a risk difference. One request per patient addresses within-patient repetition. Statistical significance does not establish practical/clinical importance; no practical threshold was prespecified. Association does not imply causation.

- **Symptom severity**: Chi-square of independence, n=9,792; Holm p=0; Cramers_V=0.5547; reject_H0. Evidence of an association/difference after omnibus Holm correction. Observed Cramers_V=0.5547. Cramer's V is on an association scale from zero toward one, not a risk difference. One request per patient addresses within-patient repetition. Statistical significance does not establish practical/clinical importance; no practical threshold was prespecified. Association does not imply causation.

- **Any reported red flag**: Chi-square of independence, n=9,617; Holm p=0; Cramers_V=0.5628; reject_H0. Evidence of an association/difference after omnibus Holm correction. Observed Cramers_V=0.5628. Cramer's V is on an association scale from zero toward one, not a risk difference. One request per patient addresses within-patient repetition. Statistical significance does not establish practical/clinical importance; no practical threshold was prespecified. Association does not imply causation.

- **Any reported comorbidity**: Chi-square of independence, n=9,596; Holm p=4.576e-14; Cramers_V=0.0855; reject_H0. Evidence of an association/difference after omnibus Holm correction. Observed Cramers_V=0.0855. Cramer's V is on an association scale from zero toward one, not a risk difference. One request per patient addresses within-patient repetition. Statistical significance does not establish practical/clinical importance; no practical threshold was prespecified. Association does not imply causation.

- **Oxygen saturation**: Kruskal-Wallis, n=9,181; Holm p=0; rank_epsilon_squared=0.3356; reject_H0. Evidence of an association/difference after omnibus Holm correction. Observed rank_epsilon_squared=0.3356. This corresponds to 33.56% of pooled rank variation. Group medians span 94.00–98.00 percentage points. One request per patient addresses within-patient repetition. Statistical significance does not establish practical/clinical importance; no practical threshold was prespecified. Association does not imply causation.

- **Heart rate**: Kruskal-Wallis, n=9,541; Holm p=0; rank_epsilon_squared=0.3678; reject_H0. Evidence of an association/difference after omnibus Holm correction. Observed rank_epsilon_squared=0.3678. This corresponds to 36.78% of pooled rank variation. Group medians span 78.00–107.00 beats/min. One request per patient addresses within-patient repetition. Statistical significance does not establish practical/clinical importance; no practical threshold was prespecified. Association does not imply causation.

- **Symptom duration**: Kruskal-Wallis, n=9,792; Holm p=4.981e-117; rank_epsilon_squared=0.0559; reject_H0. Evidence of an association/difference after omnibus Holm correction. Observed rank_epsilon_squared=0.0559. This corresponds to 5.59% of pooled rank variation. Group medians span 0.70–3.10 days (seconds inferred). One request per patient addresses within-patient repetition. Statistical significance does not establish practical/clinical importance; no practical threshold was prespecified. Association does not imply causation.



Sensitivity comparisons detected after omnibus Holm: Age at request, Chief complaint, Symptom severity, Any reported red flag, Any reported comorbidity, Oxygen saturation, Heart rate, Symptom duration, Breathing difficulty, Chest pain, Confusion, Loss of consciousness, Seizure, Severe allergic reaction, Severe bleeding, Stroke signs, Thoughts of self-harm, Asthma, Diabetes, Heart disease, Hypertension, Sickle cell disease.

Adjusted conclusions differing across cohorts: none.



## Interpretation limits



Holm controls the defined omnibus family within each analysis; posthoc Holm is local to each parent hypothesis, not a claim of global error control across every selected posthoc contrast. Tukey uses additional conservative Holm.

Statistical detection and practical importance differ. Numeric comparisons report observed median ranges, and proportion posthocs report absolute differences, but no clinically meaningful threshold was specified. Non-rejection does not prove equality. These observational analyses do not establish causation.

Missing values are excluded per variable/target pair. Unknown multivalue responses are excluded, not relabelled absent. Duration is tested in days under the preparation's inferred-seconds convention; KW ranks are unchanged by positive unit conversion or log1p transformation.

Sparse tables use conditional Monte Carlo calibration; sparse posthoc proportion contrasts use Fisher exact tests. Complete grouped-outcome separation in the self-harm flag remains a reason to audit source label construction. A floating-point p=0 is underflow, not a literal zero probability.
