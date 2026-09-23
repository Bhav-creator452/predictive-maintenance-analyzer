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