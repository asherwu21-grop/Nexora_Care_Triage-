# Frozen-model explanations



Experimental output. Not a clinical decision.



Development requests explained: 256; examples: 4. Maximum raw-margin additivity residual: 2.04e-14.



Top native gain features: ordinal__symptom_severity_ordinal, numeric__heart_rate, numeric__oxygen_saturation, numeric__red_flag_count, numeric__systolic_bp, ordinal__pain_level_ordinal, numeric__symptom_duration_log1p_hours, numeric__body_temperature, numeric__patient_age, numeric__diastolic_bp



Top grouped SHAP families: Symptom severity, Heart rate, Red flags, Oxygen saturation, Pain level, Respiratory rate, Chief complaint, Temperature, Systolic BP, Symptom onset



- Identifiers, dates, target and assessment fields are absent from transformed model features.

- Severity, pain and red flags can reflect how clinician urgency labels were constructed; verify source-label provenance even when legitimate intake fields dominate.

- The earlier EDA found complete High/Critical separation for a red-flag item. This model-derived importance does not establish independent clinical validity.

- Red-flag and comorbidity items/counts/none/unknown are grouped by summing signed per-request SHAP contributions before taking absolute means; cancellations are preserved.

- Numeric systolic and diastolic pressure remain separate; missingness indicators retain separate availability families to expose process proxies.

- Tree-path-dependent SHAP uses recorded training path counts and explains raw class margins, not probability changes, clinical recommendations, or causal effects.
