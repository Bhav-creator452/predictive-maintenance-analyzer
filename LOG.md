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