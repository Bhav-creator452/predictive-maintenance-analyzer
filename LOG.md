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