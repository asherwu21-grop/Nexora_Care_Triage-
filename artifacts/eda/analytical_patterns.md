## Analytical patterns

These observations describe the exported plots and their accompanying summary tables.
They nominate relationships for later testing; no significance or causal claims are made.
The cohort contains 14,809 labelled requests from
9,792 patients. Requests from the same patient are dependent.
No train/validation/test split exists; future holdouts from this explored cohort are
not fully blinded to these descriptive patterns.

- **Severity and pain:** The normalised bars show different urgency compositions.
  High/Critical shares by severity are Mild: 5.7%; Moderate: 30.8%; Severe: 84.1%;
  by pain they are No pain: 6.4%; Mild: 22.6%; Moderate: 45.9%; Severe: 78.6%. Test ordinal associations
  and check whether the two variables add overlapping information. These are
  associations with clinician-assigned labels, not evidence of clinical validity.
- **Chief complaint and onset:** High/Critical shares range from Skin Rash
  (16.5%) to Breathing Difficulty (73.9%). Onset shares are
  Sudden: 44.2%; Gradual: 23.8%; Recurring: 32.1%. Test differences in class composition,
  with adequate cell counts and adjustment for severity and related symptoms.
- **Duration and age:** The box/violin plots show per-urgency age medians of
  Low 33.65, Moderate 37.26, High 38.69, Critical 42.72 years and duration medians of
  Low 3.12, Moderate 2.77, High 1.93, Critical 0.77 days. The raw duration plot has a long
  right tail (maximum 120.00 days); the log-hours view
  helps compare the centre. Test distributional differences without assuming
  normality; account for possible confounding with complaint and onset.
- **Red flags and comorbidities:** Item bars overlap because responses may contain
  several items. High/Critical shares for red-flag responses are
  Breathing difficulty: 83.7%; Chest pain: 80.6%; Confusion: 81.9%; Loss of consciousness: 79.4%; Seizure: 83.5%; Severe allergic reaction: 68.9%; Severe bleeding: 80.0%; Stroke signs: 73.0%; Thoughts of self-harm: 100.0%; No red flags: 18.1%; Unknown (missing response): 29.0%.
  For comorbidity responses they are
  Asthma: 38.0%; Chronic kidney disease: 41.9%; Diabetes: 38.8%; HIV: 30.2%; Heart disease: 43.8%; Hypertension: 38.6%; Other chronic condition: 35.8%; Sickle cell disease: 42.8%; No known conditions: 28.7%; Unknown (missing response): 30.4%.
  Test item presence against absence among known responses; do not use unknowns
  as the absence group. Check sparse cells and multiple comparisons.
- **Vitals:** The violin and correlation plots nominate heart rate, respiration,
  temperature, and oxygen saturation for comparisons across urgency levels.
  Their Spearman correlations with urgency rank are heart rate
  0.597, respiration 0.392,
  temperature 0.273, and oxygen saturation
  -0.576. Systolic and diastolic correlations are
  0.071 and 0.064; compare
  their plots separately. Missing observations are excluded pairwise, not imputed.
- **Sex and pregnancy:** High/Critical shares by recorded sex are
  Female: 32.3%; Male: 32.3%; pregnancy-response shares are
  Yes: 34.9%; No: 30.2%; Not applicable: 33.6%. Compare composition and sample sizes
  before prioritising demographic hypotheses. Pregnancy response applicability
  differs by recorded sex, so avoid interpreting it as an isolated exposure.
- **Missingness and repeat requests:** The missingness heatmap motivates checking
  whether observation availability differs by urgency. The prior-request count
  distributions are also shown; test any history relationship with patient-level
  dependence accounted for. No plot here establishes that a feature predicts
  unseen patients or that an absent measurement has clinical meaning.

### Data quality surprises and limits

- **Complete grouped-outcome separation:** All 333 requests reporting "Thoughts of self-harm" are High or Critical (192 High; 141 Critical). Audit source label construction and triage-rule provenance before modelling. This is descriptive separation of a grouped outcome, and creates zero cells for later tests.
- Systolic/diastolic missingness patterns are identical:
  **True**. Their missing indicators therefore
  carry duplicate availability information; treat them as a coupled measurement.
- Unknown multi-value responses in this labelled cohort are
  **{'red_flags': 262, 'comorbidities': 293}**, compared with
  **{'red_flags': 267, 'comorbidities': 301}** in all raw intake rows. The denominators
  differ because unlabelled requests are excluded. Unknown remains separate from none.
- Symptom duration units are inferred, not confirmed by metadata. Raw-day skewness
  is 5.099; log-hours skewness is
  -0.229. Minute alignment and broad range support
  the seconds interpretation but cannot prove it.
- Observed ages span 16.03–89.96 years, with
  213 requests under eighteen. This cohort does not represent
  the entire age range of telemedicine users.
- 5,017 labelled requests are beyond the
  first request per patient; patient-level independence would be inappropriate.
  276 unlabelled intake requests are excluded from all
  target comparisons, so label availability could also affect representativeness.
- Preparation found 0 newly
  invalid vital values under its broad technical bounds, and the prepared data has
  0 duplicate request IDs and
  0 negative ages. Passing broad bounds does not confirm
  measurement accuracy; source confirmation of units remains outstanding.

Prioritise the visible severity/pain, red-flag, complaint/onset, duration, age, and
vital relationships for formal testing. Choose tests after checking assumptions,
report effect sizes, handle multiple comparisons and repeated patients, and preserve
patient-grouped time-aware validation for subsequent modelling.
