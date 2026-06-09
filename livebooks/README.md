# Livebooks — build intuition for the math

One notebook per algorithm in the library. Each one builds the idea up
from scratch with synthetic device data, visualizes every step with
VegaLite, and ends by running the real library functions. No statistics
background assumed — terms are explained as they appear.

Open them in [Livebook](https://livebook.dev); the first cell installs
the library from this repo via a relative path, so run them from a
checkout.

Suggested order — 01–03 are a size/speed gradient over level changes
and build on each other; the rest stand alone:

| # | Notebook | Algorithm | The question it answers |
|---|---|---|---|
| 01 | [Jump — Shewhart charts](01_jump_shewhart_charts.livemd) | X̄ / S control charts | "Did it suddenly jump way off?" / "Did it get erratic?" |
| 02 | [Shift — EWMA](02_shift_ewma.livemd) | EWMA chart, exact limits | "Did it move and *stay* moved?" |
| 03 | [Drift — CUSUM](03_drift_cusum.livemd) | two-sided CUSUM | "Is it slowly creeping — since when?" |
| 04 | [Trend — Theil–Sen & Mann–Kendall](04_trend_theil_sen_mann_kendall.livemd) | robust slope, trend test | "Which way is it heading, is it real, when does it hit the wall?" |
| 05 | [Changepoint — binary segmentation](05_changepoint_binary_segmentation.livemd) | SSE cost + BIC penalty | "When exactly did behavior change, in hindsight?" |
| 06 | [Shape — distribution distances](06_shape_distribution_distances.livemd) | PSI, Jensen–Shannon, earth-mover | "The average is fine but the shape feels wrong" |
| 07 | [Novelty — Mahalanobis](07_novelty_mahalanobis.livemd) | Mahalanobis distance | "Each metric is fine but the *combination* is weird" |
| 08 | [Outlier — Isolation Forest](08_outlier_isolation_forest.livemd) | Isolation Forest | "Weird in a way no rule covers, by the fleet's standards" |
