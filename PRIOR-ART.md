# Prior art: how shipped systems do metric anomaly detection

Deep-research survey (2026-06-11) positioning MobiusSmarts against
systems that actually shipped. 25 sources fetched across five angles,
125 claims extracted, the top 25 adversarially verified (three votes
each, one killed), 11 findings surviving synthesis. Verbatim quotes
below were checked against the cited sources at survey time.

## Research question

> How do real-world systems actually do on-device / edge health and anomaly detection over locally stored metrics — the problem MobiusSmarts solves (an Elixir/Nerves library running statistical-process-control detectors — Shewhart X̄/S, CUSUM, EWMA, changepoint segmentation, PSI/JS distribution drift, Mahalanobis novelty — over a round-robin metrics store on the device itself, calibrated from a single false-alarm budget, learning baselines from the device's own history)?
>
> Cover, with concrete sources:
> 1. Prior art for statistical anomaly detection in infrastructure/device monitoring that shipped in real products: RRDtool's Holt-Winters aberrant behavior detection (Brutlag LISA 2000) and its adoption/abandonment in Cacti/Munin/Ganglia; Prometheus/Grafana practice (static thresholds vs z-score/MAD recording rules); commercial anomaly detection (Datadog's anomaly monitors — what algorithms: Basic/Agile/Robust; New Relic, Dynatrace baselining; Netflix Atlas/Kayenta; Twitter AnomalyDetection / Seasonal-Hybrid ESD; Meta Prophet).
> 2. IoT/embedded-fleet device health specifically: Memfault device vitals approach; AWS IoT Device Defender Detect (statistical/ML detect — how does it baseline, what does it alarm on); Azure IoT; Balena; Golioth; anything doing detection on-device rather than fleet-side, and why teams choose one or the other (compute, connectivity, privacy).
> 3. Whether SPC control charts (CUSUM/EWMA, western-electric rules) are actually used in IT/SRE monitoring practice — papers, blog posts, SRE books, or war stories about why they did or didn't work on computing telemetry (autocorrelation, seasonality, zero-inflation problems).
> 4. How shipped systems handle the practical UX problems we hit: baseline warm-up / "learning" periods (how long, how communicated), false-alarm budgets or sensitivity knobs (how products express them), alarm fatigue / severity saturation, flapping suppression and hysteresis, and presenting anomaly state on a CLI/dashboard.
> 5. The storage side: who else does multi-resolution RRD-style downsampling and how detection deals with resolution tiers (e.g., Graphite/Whisper aggregation vs detection granularity).
>
> The goal: position MobiusSmarts' design against actual practice — what's validated by precedent, what's unusual, and what known failure modes from the field we should design against.

## Summary

Field precedent strongly validates MobiusSmarts' core architecture: embedding statistical detectors directly in a round-robin metrics store dates to Brutlag/WebTV's Holt-Winters aberrant-behavior detection shipped inside RRDtool (LISA 2000), explicitly justified by in-memory efficiency, and every surveyed commercial system (Datadog, Dynatrace, Netflix Kayenta, Twitter) ships classical/robust statistics — rolling quantiles, SARIMA, seasonal-trend decomposition, Mann-Whitney U, ESD with median/MAD — not deep learning. Three patterns are near-universal in shipped systems and should be treated as table stakes: k-of-n windowed violation counting before alarming (RRDtool 7-of-9, Dynatrace 3-of-5, Datadog window-fraction trigger/recovery thresholds), explicitly communicated baseline warm-up periods (Datadog requires ≥3 seasonal cycles of history and warns that new metrics yield poor results), and seasonality handling as a first-class concern because plain SPC-style bands provably miss local anomalies inside the seasonal envelope. MobiusSmarts' single false-alarm budget is genuinely unusual: no surveyed product expresses calibration that way — all use band-width multipliers (RRDtool delta 2–3, Datadog bounds 2–3, Dynatrace n×IQR) — though RRDtool's implicit-RRA derived defaults and Netflix's deliberate removal of per-metric hand-tuning validate the one-knob aspiration. Documented failure modes to design against: baseline contamination by long-lasting anomalies and long-running-process drift (mitigated by Datadog Robust's stable predictions and Netflix's fresh baseline clusters), and configuration complexity killing adoption — Cacti's maintainer deliberately refused to expose RRDtool's free, already-shipped Holt-Winters detection for 20+ years because "it just makes a very complicated app all the more complicated."

## Findings

### 1. (high confidence)

Running statistical anomaly detection inside a round-robin metrics store on the monitored host is directly validated precedent: Brutlag's WebTV system (LISA 2000) shipped Holt-Winters aberrant-behavior detection as five new consolidation functions inside RRDtool itself (HWPREDICT, SEASONAL, DEVPREDICT, DEVSEASONAL, FAILURES), with in-store/in-memory efficiency cited first among the reasons for embedding detection in the store rather than an external program. The feature still ships in current RRDtool with two Holt-Winters variants (HWPREDICT additive, MHWPREDICT multiplicative seasonality) decomposing series into baseline, trend, and seasonal components.

**Evidence:** Paper (verbatim): "An external program would have fetch data from the RRD at the same frequency of update, while code within RRDtool is guaranteed to operate on this data already in-memory." Current man pages confirm the feature persists: "HWPREDICT and MHWPREDICT are actually two variations on the Holt-Winters method" that "decompose data into three components: a baseline, a trend, and a seasonal coefficient." Note: efficiency was listed first among four motivations, not the sole reason.

**Sources:**
- <https://www.usenix.org/legacy/event/lisa2000/full_papers/brutlag/brutlag.pdf>
- <https://manpages.debian.org/testing/rrdtool/rrdcreate.1.en.html>
- <https://oss.oetiker.ch/rrdtool/doc/rrdcreate.en.html>

### 2. (high confidence)

k-of-n windowed violation counting (never alarming on a single excursion) is the universal shipped flapping-suppression mechanism across 25 years of systems: RRDtool's FAILURES RRA alarms only when violations in a sliding window meet a threshold (defaults 7-of-9, window max 28); Datadog expresses alert, warning, and recovery thresholds each as the fraction (0–1) of an evaluation window that must be anomalous, with distinct trigger and recovery windows constrained so alert and recovery conditions cannot hold simultaneously; Dynatrace requires by default 3 of 5 sliding-window minutes (non-consecutive, window configurable to 60 min) to violate before raising an event.

**Evidence:** Brutlag: single-observation alarming "often yields a high number of false positives"; shipped default "window length of 9 and a default threshold value of 7." Datadog: "Thresholds are expressed as numbers from 0 to 1, and are interpreted as the fraction of the associated window that is anomalous." Dynatrace: "By default, any 3 minutes out of a sliding window of 5 minutes must violate your threshold to raise an event." Attribution nuance: Brutlag adopted the moving-window technique from Ward/Glynn/Richardson 1998 rather than originating it; strictly these are k-of-n persistence filters, with true enter/exit hysteresis only in Datadog's separate trigger/recovery windows.

**Sources:**
- <https://www.usenix.org/legacy/event/lisa2000/full_papers/brutlag/brutlag.pdf>
- <https://manpages.debian.org/testing/rrdtool/rrdcreate.1.en.html>
- <https://docs.datadoghq.com/monitors/types/anomaly/>
- <https://docs.dynatrace.com/docs/discover-dynatrace/platform/davis-ai/anomaly-detection/concepts/auto-adaptive-threshold>

### 3. (high confidence)

No surveyed shipped system expresses calibration as a statistical false-alarm budget — all expose sensitivity as a band-width multiplier knob: RRDtool uses a confidence-band scale factor delta (default 2, recommended 2–3, asymmetric via --deltapos/--deltaneg) and its paper concedes window/threshold parameters are "probably the most difficult to set a priori" with "no single optimal set of values"; Datadog exposes a single numeric bounds parameter ("a value of 2 or 3 should be large enough," interpretable as standard deviations); Dynatrace exposes n × signal-fluctuation where fluctuation is the 25th–75th percentile IQR. MobiusSmarts deriving all detector calibration from one false-alarm budget is therefore unusual relative to actual practice.

**Evidence:** Brutlag: "Choose 2 to detect more failures (which may just mean a higher rate of false positives)... These parameters are probably the most difficult to set a priori" — seven independent knobs with only heuristic guidance, no target-false-alarm-rate calibration anywhere. Datadog: "setting the bounds to 2 or 3 will capture most 'normal' points." Dynatrace: "you can control how many times the signal fluctuation is added to the baseline to produce the actual threshold." Qualification: Datadog's one knob covers band width only; the monitor product layers algorithm choice, seasonality, and window settings on top (this claim passed 2-1).

**Sources:**
- <https://www.usenix.org/legacy/event/lisa2000/full_papers/brutlag/brutlag.pdf>
- <https://www.datadoghq.com/blog/introducing-anomaly-detection-datadog/>
- <https://docs.datadoghq.com/dashboards/functions/algorithms/>
- <https://docs.dynatrace.com/docs/discover-dynatrace/platform/davis-ai/anomaly-detection/concepts/auto-adaptive-threshold>

### 4. (high confidence)

Collapsing many detector parameters into a single entry point has shipped precedent: RRDtool's create command, "to simplify the creation for the novice user," implicitly creates all four dependent RRAs (SEASONAL, DEVSEASONAL, DEVPREDICT, FAILURES) with derived defaults (gamma inherited from HWPREDICT's alpha; FAILURES threshold 7, window 9) when only HWPREDICT is specified — though that entry point is still an RRA spec with four arguments, not a single scalar like a false-alarm budget.

**Evidence:** Man page (verbatim): "In order to simplify the creation for the novice user... the RRDtool create command supports implicit creation of the other four when HWPREDICT is specified alone and the final argument rra-num is omitted." Implicit SEASONAL/DEVSEASONAL "will both have the same value for gamma: the value specified for the HWPREDICT alpha argument." Verified in current rrdtool 1.7.x docs (2026).

**Sources:**
- <https://manpages.debian.org/testing/rrdtool/rrdcreate.1.en.html>
- <https://oss.oetiker.ch/rrdtool/doc/rrdcreate.en.html>

### 5. (high confidence)

Configuration complexity is a documented adoption killer even when detection is free: Cacti never exposed RRDtool's built-in Holt-Winters aberrant-behavior detection despite it shipping in the storage engine Cacti sits on. The April 2020 enhancement request (issue #3398, citing Brutlag's paper and noting the GPLv2 code could be copied directly) drew core maintainer TheWitness's reply that the project had known about it for years but deliberately avoided it for complexity reasons; the issue was closed unimplemented on 2023-04-16 alongside other Holt-Winters requests (#5022: "requires a large architecture change") and is now locked — Brutlag-style detection never shipped in Cacti.

**Evidence:** Maintainer comment 2020-04-04 (verbatim, verified via GitHub API): "We have known about this for several years, it just makes a very complicated app all the more complicated, so we have been avoiding implementing." Closure 2023-04-16: "Closing the holt winters issues for now. We will reopen once we get the overall issue list reduced." Design implication for MobiusSmarts: the per-knob configuration burden, not the algorithm, was the barrier — supporting the calibrate-from-one-budget approach.

**Sources:**
- <https://github.com/Cacti/cacti/issues/3398>
- <https://github.com/Cacti/cacti/issues/5022>

### 6. (high confidence)

Datadog's flagship anomaly-detection product ships exactly three user-selectable algorithms, all classical time-series statistics rather than deep learning: Basic (non-seasonal lagging rolling quantile, adapts quickly but ignores seasonality and long-term trends), Agile (a robustified SARIMA variant, seasonal but quick to adjust to level shifts), and Robust (seasonal-trend decomposition for stable seasonal metrics). A deprecated fourth algorithm ("adaptive") can no longer be selected for new monitors.

**Evidence:** Current docs (fetched June 2026) match the 2016 launch blog: Basic "uses a simple lagging rolling quantile computation... has no knowledge of seasonality or long-term trends"; Agile is "a robust version of the seasonal autoregressive integrated moving average (SARIMA) algorithm"; Robust is "a seasonal-trend decomposition algorithm." Scope note: Datadog's separate Watchdog/AIOps features use ML but expose no named user-selectable algorithms; the claim holds for the anomaly-monitor product.

**Sources:**
- <https://docs.datadoghq.com/monitors/types/anomaly/>
- <https://www.datadoghq.com/blog/introducing-anomaly-detection-datadog/>

### 7. (high confidence)

Baseline warm-up periods are explicit, quantified, and communicated in shipped products: Datadog's seasonal algorithms (Agile/Robust) require at least three times the chosen seasonality period of historical data before computing a baseline (e.g., ~3 weeks for weekly seasonality), the docs warn that anomaly detection on a new metric "may yield poor results," and the UI shows no bounds during the first seasons — directly validating MobiusSmarts' need for a learning-period concept and for surfacing it to users.

**Evidence:** Docs (verbatim, including their typo): "Machine learning algorithms require at least three time as much historical data time as the chosen seasonality time to compute the baseline"; "the anomalies function uses the past to predict what is expected in the future, so using it on a new metric may yield poor results." The algorithms page ties the 3x requirement explicitly to robust/agile with weekly default seasonality requiring three weeks of history.

**Sources:**
- <https://docs.datadoghq.com/monitors/types/anomaly/>
- <https://docs.datadoghq.com/dashboards/functions/algorithms/>

### 8. (high confidence)

Baseline contamination is a recognized field failure mode with two distinct shipped mitigations: (a) anomaly contamination — Datadog's Robust algorithm is deliberately stable so "predictions remain constant even through long-lasting anomalies," at the documented cost of slow adaptation to intended level shifts; (b) history contamination — Netflix Kayenta refuses to compare against the long-lived production cluster's own history, instead spinning up a fresh ~3-instance baseline cluster on current code, because long-running-process effects (JIT, caches, memory) make long-lived baselines unreliable. Both are cautionary precedents for MobiusSmarts learning baselines from a device's own long-running history.

**Evidence:** Datadog: "Its predictions are very stable, so its forecast won't be unduly influenced by long-lasting anomalies." Netflix (verbatim): "comparing a newly created canary cluster to a long-lived production cluster could produce unreliable results. Creating a brand new baseline cluster ensures that the metrics produced are free of any effects caused by long-running processes." Framing nuance: Netflix's failure mode is process-age mismatch in concurrent comparison, applied here analogically to learning from history.

**Sources:**
- <https://www.datadoghq.com/blog/introducing-anomaly-detection-datadog/>
- <https://docs.datadoghq.com/monitors/types/anomaly/>
- <https://netflixtechblog.com/automated-canary-analysis-at-netflix-with-kayenta-3260bc7acc69>

### 9. (high confidence)

Plain SPC-style detection on seasonal computing telemetry is a documented failure mode, and production systems handle seasonality before applying statistical tests: Twitter shipped the open-source AnomalyDetection R package whose Seasonal Hybrid ESD (S-H-ESD) algorithm extends the Generalized ESD test with time-series (STL-style) decomposition and robust statistics (median/MAD replacing mean/std-dev), explicitly because "many of the anomalies... are local anomalies within the bounds of the time series' seasonality (hence, cannot be detected using the traditional approaches)." This warns that MobiusSmarts' Shewhart/CUSUM/EWMA detectors will miss within-seasonal-envelope anomalies on any device metric with daily/weekly cycles unless seasonality is removed first.

**Evidence:** README (verbatim): S-H-ESD "builds upon the Generalized ESD test... [employing] time series decomposition and using robust statistical metrics, viz., median together with ESD" and "can be used to detect both global as well as local anomalies." Production use corroborated by Twitter's 2015 engineering blog (Tweets-per-second, CPU utilization) and the authors' arXiv paper. Caveats: repo archived 2021-11-13 and unmaintained; residual-normality assumption and middling Numenta benchmark scores bound its quality as a model to copy.

**Sources:**
- <https://github.com/twitter/AnomalyDetection>
- <https://arxiv.org/abs/1704.07706>

### 10. (high confidence)

Netflix Kayenta demonstrates replacing many hand-tuned sensitivity knobs with algorithmic statistical decisions in production: its primary metric-comparison algorithm classifies each metric Pass/High/Low using confidence intervals from the nonparametric Mann-Whitney U test (98% CI in the default judge), and Netflix explicitly states it "removed much of the complexity of setting proper thresholds and other hand-tuning" in favor of the statistical algorithms — precedent for MobiusSmarts replacing per-detector tuning with principled calibration.

**Evidence:** Blog (verbatim): "The primary metric comparison algorithm in Kayenta uses confidence intervals, computed by the Mann-Whitney U test"; "We have removed much of the complexity of setting proper thresholds and other hand-tuning, and instead rely on superior algorithms." Corroborated by Spinnaker judge docs and MannWhitneyClassifier.scala in the open-source code. Qualifications: "much," not all — aggregate score thresholds, group weights, and optional per-metric settings remain; practitioners later documented Mann-Whitney judge shortcomings (spinnaker/kayenta#780).

**Sources:**
- <https://netflixtechblog.com/automated-canary-analysis-at-netflix-with-kayenta-3260bc7acc69>
- <https://spinnaker.io/docs/guides/user/canary/judge/>
- <https://github.com/spinnaker/kayenta>

### 11. (high confidence)

Commercial baselining favors robust/quantile statistics over SPC mean-and-sigma control limits: Dynatrace's auto-adaptive thresholds compute the baseline as the 99th percentile of per-minute measurements (over a 7-day lookback per the same doc page) and derive the alert margin as n × the interquartile range (25th–75th percentile) — no standard deviation appears anywhere in the mechanism. This supports MobiusSmarts' use of robust estimators but shows a major vendor chose quantiles over Shewhart-style X-bar/S limits for noisy IT telemetry.

**Evidence:** Docs (verbatim, verified live June 2026): "Measurements for each minute are used to calculate the 99th percentile of all the measurements. This determines the appropriate baseline"; "The interquartile range between the 25th and 75th percentiles is then used as the signal fluctuation, which can be added to the baseline." Scope: this describes the auto-adaptive threshold feature specifically, not all Dynatrace baselining (topology-driven response-time baselines are separate).

**Sources:**
- <https://docs.dynatrace.com/docs/discover-dynatrace/platform/davis-ai/anomaly-detection/concepts/auto-adaptive-threshold>
