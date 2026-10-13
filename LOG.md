# Project Development Log

## Day 0 — Environment Setup

### Goal
Set up the development environment for the Predictive Maintenance Analyzer.

### Completed
- [ ] Python environment
- [ ] Virtual environment
- [ ] Project structure
- [ ] Dependencies
- [ ] Git repository
- [ ] Dataset

### What I learned

### Problems / Blockers

### Next Step
Begin Day 1: C-MAPSS dataset exploration.

## Day 1 — Dataset Understanding & Sensor Exploration

### Dataset
- Dataset: NASA C-MAPSS FD001
- Training engines: 100
- Columns: 26
- Features: 3 operating settings + 21 sensors

### Sensor Exploration

Visual inspection of multiple engines showed that sensor behavior varies considerably.

### Strong candidate degradation sensors
- Sensor 2
- Sensor 3
- Sensor 4
- Sensor 7
- Sensor 8
- Sensor 9
- Sensor 11
- Sensor 12
- Sensor 13
- Sensor 14
- Sensor 15
- Sensor 17
- Sensor 20
- Sensor 21

Some sensors showed increasing trends while others showed decreasing trends.

### Relatively flat sensors
- Sensor 1
- Sensor 5
- Sensor 6
- Sensor 10
- Sensor 16
- Sensor 18
- Sensor 19

These were identified as relatively uninformative based on visual inspection only. 
They will not be permanently removed until model-based feature selection is performed.

### Engine Lifetime

The engine lifetime distribution shows substantial variation between engines, with most engines concentrated in the lower-to-middle portion of the observed lifetime range and fewer engines surviving substantially longer.

### Key Learning

RUL cannot be calculated using a single global engine lifetime. 
Each engine has its own total lifetime, so RUL must be calculated separately for every engine.

### Engine-Level Sensor Analysis

For Engine 1, selected sensors were plotted across the complete operating trajectory.

Observations:
- Sensor 2 showed a noisy upward trend.
- Sensor 4 showed a stronger upward progression, especially later in life.
- Sensor 7 showed a downward trend.
- Sensor 9 was comparatively noisy for Engine 1 and showed a weaker downward tendency.
- Sensor 14 showed a noticeable downward trend.
- Sensor 20 showed a gradual downward trend.

### Key Insight

Degradation does not necessarily mean that a sensor value increases.
Different sensors can exhibit increasing, decreasing, nonlinear, or noisy behavior as the engine progresses.

Visual inspection identifies candidate signals but does not prove predictive usefulness. Final feature selection will be evaluated using model performance.


### Operating Settings Analysis

- Analyzed `setting_1`, `setting_2`, and `setting_3`.
- `setting_1` contains 158 unique values and fluctuates around zero.
- `setting_2` contains 13 unique values and behaves as a discrete operating condition.
- `setting_3` is constant at 100 across the FD001 training dataset.
- `setting_3` has zero variance and therefore provides no predictive variation.
- Initial decision: exclude `setting_3` from the modeling feature set.
- Pearson correlations between the operating settings and sensors were inspected.
- Correlation values for `setting_1` and `setting_2` were generally small in the inspected results.
- Correlation was treated as exploratory evidence rather than a final feature-selection method.

## Data Understanding & RUL Target Construction

### Completed
- Loaded and inspected the C-MAPSS FD001 training dataset.
- Identified 100 unique engines and 20,631 operational observations.
- Analyzed engine cycles, sensor trajectories, and operating settings.
- Identified `setting_3` as constant at 100 across FD001.
- Investigated sensor behavior across multiple engine trajectories.
- Calculated the maximum recorded cycle for each engine.
- Constructed engine-specific RUL using:

  `RUL = max_cycle - current_cycle`

### RUL Validation
- Minimum RUL: 0
- Maximum RUL: 361
- Negative RUL values: 0
- All final engine cycles have RUL = 0.
- RUL decreases by exactly 1 for each consecutive operating cycle.
- All automated RUL validation checks passed.

### Key ML Lessons
- RUL must be constructed separately for each engine.
- `engine_id` is important for grouping and leakage-safe splitting.
- `max_cycle` is an intermediate target-construction variable and should not be used as a model input.
- Sensor changes should not automatically be interpreted as degradation because operating conditions may also affect sensor readings.

## Day 3 — RUL Label Definition

- Constructed raw engine-specific RUL as:
  `RUL = max_cycle - current_cycle`
- Applied the handbook's piecewise-linear RUL cap:
  `RUL = min(RUL, 125)`
- Preserved the original target as `RUL_raw` for transparency.
- The final modeling target is stored in `RUL`.
- The capped target remains at 125 during early life and decreases linearly toward 0 as the engine approaches its final recorded cycle.
- Automated validation confirmed minimum RUL = 0 and maximum RUL = 125.


## Day 4 — Feature Engineering

### Completed
- Selected six informative sensors based on Day 2 EDA:
  `sensor_2`, `sensor_4`, `sensor_7`, `sensor_9`, `sensor_14`, `sensor_20`.
- Created 5-cycle rolling mean features.
- Created 5-cycle rolling standard deviation features.
- Created engine-specific difference-from-start features.
- Used `groupby("engine_id")` to ensure rolling calculations and baselines stay within each engine.
- Handled first-cycle rolling standard deviation NaN values by assigning 0.
- Created 24 sensor-derived features.
- Validated that engineered features contain 0 NaN values.
- Saved the engineered dataset to:
  `data/engineered_train_FD001.csv`.

### Validation
- Total NaN values in engineered features: 0.
- First observation of each engine correctly starts its own rolling history.
- Difference-from-start is 0 at each engine's first observation.

## Day 5 — Train/Validation Splitting & Official Test Set

### Completed
- Split the 100 training engines into 80 training engines and 20 validation engines.
- Performed the split at the engine level rather than the row level.
- Training set: 16,561 rows from 80 engines.
- Validation set: 4,070 rows from 20 engines.
- Verified zero engine overlap between training and validation.
- Loaded the official `test_FD001.txt`.
- Test set contains 13,096 rows, 26 columns, and 100 engines.
- Loaded `RUL_FD001.txt` containing 100 ground-truth RUL values.
- Verified the number of official RUL values matches the number of test engines.
- Kept the official test set separate from model development.

### Important Leakage Rule
Validation engines are completely unseen during training.
The official test set remains untouched for final evaluation.

## Day 6 — Baseline Modeling

### Models
- Linear Regression
- Random Forest Regressor
  - n_estimators: 100
  - random_state: 42

### Evaluation setup
- Training: 16,561 observations from 80 engines
- Validation: 4,070 observations from 20 unseen engines
- Input features: 24 engineered sensor features
- Target: capped RUL (0–125 cycles)
- Engine overlap: 0

### Validation Results

| Model | RMSE | MAE |
|---|---:|---:|
| Linear Regression | 16.15 | 12.95 |
| Random Forest | 13.08 | 9.20 |

### Findings
Random Forest outperformed Linear Regression on both validation
metrics. Its RMSE was approximately 19% lower and its MAE
approximately 29% lower.

Linear Regression produced predictions outside the target range.
Predictions were not clipped when calculating the baseline metrics.

### Limitations
These results apply to the current engine-level validation split
and capped RUL target. They are not official test-set results and
do not establish production reliability.

## Day 7 — XGBoost Baseline

### Configuration
- Library: XGBoost 3.4.1
- Model: XGBRegressor
- Objective: `reg:squarederror`
- Trees: 200
- Maximum depth: 4
- Learning rate: 0.05
- Subsample: 0.8
- Column sample per tree: 0.8
- Random state: 42

### Evaluation Setup
- Training: 16,561 observations from 80 engines
- Validation: 4,070 observations from 20 unseen engines
- Features: 24 engineered sensor-derived features
- Target: capped RUL (0–125 cycles)
- Official test set: not used

### Validation Results

| Model | RMSE | MAE |
|---|---:|---:|
| Linear Regression | 16.15 | 12.95 |
| Random Forest | 13.08 | 9.20 |
| XGBoost | 13.00 | 9.14 |

### Findings
XGBoost achieved the lowest validation RMSE and MAE. It improved substantially over Linear Regression, but only marginally over Random Forest on this split.

These are baseline results with fixed hyperparameters, not tuned results or official test-set metrics.

## Day 8 — Feature Importance & Iteration

### Feature Importance
- Extracted XGBoost feature importances for the 24 engineered features.
- `sensor_4_roll_mean` was the highest-ranked individual feature.
- Grouped feature importance by original sensor.
- Sensor 14 had the lowest combined importance among the six selected sensors.

### Feature Ablation Experiments

| Experiment | Features | RMSE | MAE |
|---|---:|---:|---:|
| Baseline | 24 | 12.999 | 9.145 |
| Without sensor_14 | 20 | 12.869 | 9.209 |
| Without sensor_9 | 20 | 12.886 | 9.277 |
| Without sensor_4 | 20 | 13.330 | 9.599 |
| Without sensor_7 | 20 | 13.437 | 9.690 |
| Without sensor_20 | 20 | 13.688 | 9.525 |
| Without sensor_2 | 20 | 13.693 | 9.618 |
| Without sensor_14 and sensor_9 | 16 | 13.203 | 9.715 |

### Decision
Retained all 24 engineered features for the current baseline. Removing Sensor 14 or Sensor 9 individually produced small, conflicting changes in RMSE and MAE, while removing both worsened both metrics. Removing Sensors 2, 4, 7, or 20 also worsened both metrics in their individual experiments.

The final feature set is provisional and may be revisited if later validation evidence justifies a change.

### Limitations
- Experiments used one engine-level validation split.
- Feature importance does not establish physical causation.
- Repeated experimentation on the same validation set can lead to selection bias.
- The official test set remains unused.


## Day 9 — Proper Evaluation Beyond RMSE

### Model
- XGBoost baseline using all 24 engineered features.
- Validation set: 4,070 observations from 20 unseen engines.
- Target: capped RUL, ranging from 0 to 125 cycles.

### Overall Validation Metrics
- RMSE: 12.999 cycles
- MAE: 9.145 cycles
- Mean signed error: +1.002 cycles

### Critical Low-RUL Evaluation

| Region | Observations | RMSE | MAE | Mean Signed Error |
|---|---:|---:|---:|---:|
| Critical: RUL < 30 | 600 | 7.553 | 5.321 | +3.214 |
| Other: RUL >= 30 | 3,470 | 13.723 | 9.806 | +0.619 |
| Overall | 4,070 | 12.999 | 9.145 | +1.002 |

### Findings
- The predicted-versus-actual plot shows a clear positive relationship with noticeable prediction spread.
- Absolute errors are smaller in the critical low-RUL subset than in the remaining observations.
- Positive mean signed error indicates that the model tends to overestimate RUL, particularly in the critical subset.
- The model's errors must be assessed for both magnitude and direction.

### Limitations
- These are validation metrics, not official held-out test results.
- The labels are capped at 125 cycles.
- Rows belonging to the same engine are correlated; observations are not independent machines.
- These metrics alone do not establish real-world industrial reliability.

## Day 10 — Hyperparameter Tuning & Official Test Evaluation

### Hyperparameter Search
- Evaluated 12 unique XGBoost configurations.
- Parameters varied: `max_depth`, `n_estimators`, and `learning_rate`.
- Selection criterion: validation RMSE, with MAE as a secondary metric.
- Selected configuration:
  - `max_depth = 5`
  - `n_estimators = 250`
  - `learning_rate = 0.03`

### Best Validation Results
- RMSE: 12.821 cycles
- MAE: 8.919 cycles

### Official Test Evaluation
- Test engines: 100
- Test feature matrix: 100 × 24
- Official labels: original, uncapped `RUL_FD001.txt`
- RMSE: 16.452 cycles
- MAE: 12.442 cycles

### Secondary Capped-Label Evaluation
- Test labels were capped at 125 for a separate diagnostic.
- RMSE: 15.390 cycles
- MAE: 11.372 cycles
- This result is not the primary official test metric.

### Prediction Diagnostics
- Official test labels above 125: 11
- Predictions below zero: 1
- Predictions above 125: 2

### Interpretation and Limitations
The tuned XGBoost model was evaluated on the official FD001 test engines for the first time. Test errors were higher than the best validation errors. Direct comparison is limited because validation metrics were calculated across all validation rows, whereas the official test metrics use one final observed row per engine, with uncapped ground-truth labels.

A matched partial-trajectory validation experiment is required before attributing the difference solely to overfitting. These results are benchmark evidence, not proof of real-world industrial reliability.

## Day 11 — Sequential Modeling & Sequence Preparation

### Objective
Prepare chronological sensor sequences for an LSTM-based Remaining Useful Life (RUL) prediction model.

### Environment
- PyTorch version: 2.14.0+cpu
- Training device: CPU
- CUDA available: No

### Input Data
- Dataset: NASA C-MAPSS FD001
- Selected sensors: `sensor_2`, `sensor_4`, `sensor_7`, `sensor_9`, `sensor_14`, `sensor_20`
- Sequence window: 30 consecutive operating cycles
- Features per time step: 6 raw sensor readings
- Target: capped RUL at the final cycle of each sequence

### Preprocessing
- Sorted observations by engine ID and cycle.
- Applied `StandardScaler` to the six selected sensor channels.
- Fitted the sensor scaler using training-engine observations only.
- Applied the fitted scaler to validation-engine observations.
- Generated overlapping 30-cycle windows separately for each engine.
- Excluded early observations that could not form a complete 30-cycle window.
- Assigned each sequence the capped RUL of its final observed cycle.
- Preserved the engine-level training/validation separation.

### Sequence Shapes

| Dataset | Input shape | Target shape |
|---|---|---|
| Training | `(14241, 30, 6)` | `(14241,)` |
| Validation | `(3490, 30, 6)` | `(3490,)` |

### Validation Checks
- All input sequences contain finite values.
- Training and validation target values remain within 0–125 cycles.
- Sequence inputs and targets have matching sample counts.
- No engine overlap between training and validation: 0.
- Forward-pass dimensions verified as `(batch_size, 30, 6)` inputs and `(batch_size,)` outputs.

### Outcome
Sequence preparation completed successfully. The data is ready for LSTM training.

---

## Day 12 — Building and Training an LSTM for RUL Prediction

### Objective
Build a small PyTorch LSTM that predicts RUL from chronological sensor sequences and evaluate its performance on unseen validation engines.

### Model Architecture
- Input size: 6 sensor channels
- Sequence length: 30 cycles
- LSTM layers: 1
- Hidden units: 32
- Output layer: Fully connected linear layer
- Output: One continuous RUL prediction per sequence
- Batch size: 128
- Optimizer: Adam
- Learning rate: 0.001
- Loss function: Mean Squared Error (MSE)
- Training device: CPU

### Experiment 1 — Initial Training

The first LSTM was trained for 10 epochs using the capped RUL targets without target standardization.

Training MSE decreased from 7809.408 to 3545.237, and validation MSE decreased from 7121.097 to 3318.292.

However, its validation predictions were concentrated in a narrow range of 1.76–38.92 cycles.

| Metric | Initial LSTM |
|---|---:|
| Validation RMSE | 57.605 cycles |
| Validation MAE | 46.498 cycles |
| Mean predicted RUL | 34.69 cycles |
| Mean actual RUL | 79.93 cycles |

A constant-prediction baseline achieved RMSE 41.763 cycles and MAE 37.002 cycles, outperforming the initial LSTM.

### Diagnosis
Although training loss decreased, the network severely underpredicted RUL across the validation set. Decreasing loss alone was not sufficient evidence of a useful predictive model.

### Experiment 2 — Target Standardization and Early Stopping

The training process was revised to improve optimization.

- Standardized the target using training-sequence target statistics only.
- Training target mean: 80.812 cycles.
- Training target standard deviation: 41.809 cycles.
- Used gradient clipping with a maximum gradient norm of 1.0.
- Increased the maximum training duration to 30 epochs.
- Implemented early stopping with a patience of 5 epochs.
- Restored the model checkpoint with the lowest validation MSE.
- Converted predictions back to cycles before calculating evaluation metrics.

The best checkpoint was selected at epoch 5, with standardized validation MSE of 0.1116.

### Retrained LSTM — Validation Results

| Metric | Result |
|---|---:|
| Validation RMSE | 13.966 cycles |
| Validation MAE | 9.980 cycles |
| Mean predicted RUL | 79.11 cycles |
| Mean actual RUL | 79.93 cycles |
| Minimum prediction | -4.66 cycles |
| Maximum prediction | 133.02 cycles |

### Experiment 3 — Fair Comparison Against Tuned XGBoost

The tuned XGBoost model was retrained using only the original 80 training engines and evaluated on the same 3490 validation-sequence endpoints as the LSTM.

| Model | Validation RMSE | Validation MAE |
|---|---:|---:|
| Tuned XGBoost | 13.570 cycles | 9.608 cycles |
| LSTM | 13.966 cycles | 9.980 cycles |

Both models were evaluated against the same capped RUL targets at the sequence endpoints.

### Findings
- Target standardization and early stopping substantially improved LSTM performance.
- The retrained LSTM substantially outperformed its initial run and the constant-prediction baseline.
- Tuned XGBoost currently performs slightly better than the LSTM on matched validation endpoints.
- The LSTM still produces predictions below zero and above the training target cap.
- Its performance does not yet justify replacing XGBoost as the preferred model.

### Limitations and Next Steps
- Current LSTM results are based on the engine-level validation split.
- Validation metrics are model-selection results and are not official test-set metrics.
- The official test set has not yet been evaluated using the LSTM.
- Final preprocessing must fit the sensor scaler using all labelled training engines and apply that same scaler to the official test sequences.
- A fair comparison on the official 100-engine test set is required before selecting a final model.
- Prediction-boundary handling and uncertainty estimation remain future tasks.

### Day 12 Outcome
A working LSTM pipeline has been implemented, trained, and evaluated. The model shows meaningful learning after the training procedure was revised, but tuned XGBoost remains the stronger model on the current matched validation comparison.


## Day 13 — XGBoost vs LSTM: Official Test Comparison

### Evaluation Setup
- Official test engines: 100
- Ground-truth labels: original, uncapped `RUL_FD001.txt`
- Both models evaluated on the same 100 engines in matching order.
- Test set remained excluded from model training and hyperparameter selection.

### Models
- Tuned XGBoost: `max_depth=5`, `n_estimators=250`, `learning_rate=0.03`
- LSTM: 30-cycle sequences, six sensors, one LSTM layer, 32 hidden units.
- Final LSTM trained on all labelled training engines for five epochs, based on the best validation epoch.

### Overall Official Test Results

| Model | RMSE | MAE |
|---|---:|---:|
| Tuned XGBoost | 16.452 | 12.442 |
| LSTM | 16.425 | 12.596 |

### Critical Low-RUL Results

Critical region: true RUL < 30 cycles.

| Model | Observations | RMSE | MAE |
|---|---:|---:|---:|
| Tuned XGBoost | 25 | 6.596 | 5.345 |
| LSTM | 25 | 5.405 | 4.438 |

### Findings
- Overall RMSE is nearly identical, with LSTM lower by 0.027 cycles.
- XGBoost has a slightly lower overall MAE.
- LSTM has lower RMSE and MAE in the critical low-RUL subset.
- The critical-region sample is small, so this result requires cautious interpretation.
- Both models were trained using capped RUL targets, while evaluation used the original uncapped test labels.

### Model Decision
Retain both models for further evaluation. XGBoost remains the primary baseline due to its simpler deployment characteristics and slightly lower overall MAE. LSTM remains a challenger because of its better critical-zone results. No ensemble will be adopted without further evidence.

### Limitations
These benchmark results are based on the C-MAPSS FD001 dataset and do not establish real-world industrial reliability. Further work is required on uncertainty estimation, asymmetric maintenance costs, prediction constraints, and deployment monitoring.


## Day 14 — Uncertainty Estimation with Quantile Regression

### Objective
Extend point predictions with lower and upper RUL prediction bounds.

### Initial Quantile Models
- Algorithm: XGBoost quantile regression
- Quantiles: 0.10, 0.50, and 0.90
- Features: 24 engineered sensor features
- Initial validation observations: 4,070
- Quantile-crossing observations: 143
- Initial empirical coverage: 91.6%
- Initial mean interval width: 43.16 cycles

### Calibration Experiment
- Split the original 80 training engines into 60 model-fitting engines and 20 calibration engines.
- Kept the original 20 validation engines separate from model fitting and calibration.
- Fitted lower and upper quantile models using the 60 fitting engines.
- Estimated a conformal-style interval correction from calibration residual scores.
- Estimated correction (`qhat`): -0.417 cycles.
- Calibrated validation coverage: 83.5%.
- Mean width of valid calibrated intervals: 41.94 cycles.
- Empty intervals: 0.

### Findings
The calibrated interval achieved empirical validation coverage of 83.5%, close to the intended 80% target. The estimated correction was negative, indicating that the raw intervals were wider than necessary under this calibration experiment.

### Limitations
- Calibration observations within an engine are correlated; the row-level coverage result is not a formal guarantee for future engines.
- The current intervals estimate the capped RUL target, not necessarily uncapped physical remaining life.
- The original training target was capped at 125 cycles.
- Additional engine-level coverage checks and evaluation are required before using these intervals in maintenance decisions.

### Outcome
A quantile-based RUL interval and calibration procedure have been implemented and evaluated on validation data. Further work is needed to assess coverage on independent engines and determine whether the interval width is practically useful.


## Day 15 — Cost-Based Evaluation

### Objective
Evaluate the XGBoost and LSTM models with an asymmetric scoring function that penalizes RUL overestimation more heavily than underestimation.

### Scoring Function

Define:

`error = predicted_RUL - actual_RUL`

The implemented score is:

- If `error < 0`: `exp(-error / 13) - 1`
- If `error >= 0`: `exp(error / 10) - 1`

The total score is the sum of the individual observation penalties. Lower scores are better.

### Asymmetry Validation
- Underestimation by 10 cycles: penalty = 1.158
- Overestimation by 10 cycles: penalty = 1.718
- Verified that the same-magnitude overestimation receives the larger penalty.

### Official Test Results

| Model | RMSE | MAE | Asymmetric Score |
|---|---:|---:|---:|
| Tuned XGBoost | 16.452 | 12.442 | 493.579 |
| LSTM | 16.425 | 12.596 | 423.567 |

### Findings
- LSTM achieved an approximately 14.2% lower asymmetric score than XGBoost.
- LSTM had slightly lower overall RMSE.
- XGBoost had slightly lower overall MAE.
- Day 13 evaluation also showed lower LSTM errors in the critical region where actual RUL < 30 cycles.
- The model ranking therefore depends on the evaluation objective: ordinary MAE slightly favours XGBoost, while the asymmetric score favours LSTM.

### Model Selection Decision
Carry LSTM forward as the provisional primary point-prediction model for Week 4 because of its better asymmetric score and critical-zone results. Retain XGBoost as a benchmark and as the current reference implementation for quantile-based uncertainty estimation.

The existing XGBoost prediction intervals must not be presented as calibrated LSTM intervals. Uncertainty bounds compatible with the selected LSTM must be developed and evaluated before using them for risk-based alerts.

### Limitations
- Evaluation was performed on 100 official FD001 test engines.
- The asymmetric score uses illustrative penalty constants and is not a calibrated model of actual maintenance costs.
- Test results do not establish industrial safety or deployment readiness.
- The training target was capped at 125 cycles, while the official test labels were uncapped.
- Further engine-level evaluation, uncertainty calibration, and operational validation are required.

## Day 16 — Risk-Based Alerting

### Objective
Convert RUL estimates and uncertainty intervals into actionable HIGH, MEDIUM, and LOW risk tiers.

### Alerting Model
- Model: LSTM
- Input window: 30 cycles
- Sensor channels: 6
- Fitting engines: 60
- Calibration engines: 20
- Calibration correction: 30.78 cycles
- Target: capped RUL (0–125 cycles)

### Initial Risk Rule
- HIGH if predicted RUL < 20 or lower bound < 20.
- MEDIUM if predicted RUL < 50.
- LOW otherwise.

The initial rule assigned 34 engines to HIGH, 0 to MEDIUM, and 66 to LOW. The broad interval made the MEDIUM condition unreachable.

### Revised Risk Rule
- HIGH if predicted RUL < 20 or lower bound < 20.
- MEDIUM if not HIGH and predicted RUL < 50 or lower bound < 50.
- LOW otherwise.

### Validation Against Known Test Outcomes
- Official test engines: 100.
- Initial inspection of three HIGH-risk engines found true RUL values of 8, 7, and 10 cycles.
- Record the revised HIGH/MEDIUM/LOW counts and measured high-alert precision and critical-engine recall after validation.

### Limitations
- The calibration estimate uses only 20 engines.
- The intervals concern capped RUL and are not guaranteed confidence intervals for uncapped physical remaining life.
- Risk thresholds are provisional demonstration settings, not certified maintenance limits.
- Test labels are available only for offline evaluation and must not be used by a deployed alerting function.
- Human engineering review remains necessary before any maintenance decision.