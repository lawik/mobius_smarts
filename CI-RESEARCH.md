# CI-RESEARCH — statistical regression gates and device-in-the-loop CI

Research notes (2026-06-11) on how mature projects detect performance
regressions in CI, how hardware/embedded projects run device-in-the-loop
pipelines, and how the two combine into a Nerves + MobiusSmarts flow for
validating firmware (e.g. new Nerves systems) on real boards.

Method note: the core statistical findings below went through a
fact-checking pass (multi-source fetch, 3-vote adversarial verification per
claim; 24 of 25 top claims confirmed, 1 refuted). The device-farm and
SBC-noise sections come from a follow-up targeted pass that is sourced but
not adversarially verified — treat those as well-sourced reportage rather
than cross-checked fact. Sources are linked inline; a caveats section at
the end lists what was refuted or left unverified.

## TL;DR

1. **Fixed thresholds ("fail if >10% slower than baseline") are a
   documented failure mode.** MongoDB measured up to 99% false positives
   under that scheme while still missing real sub-threshold regressions.
   Mature projects converged on change-point detection over the per-commit
   metric series instead.
2. **Nobody hard-gates merges on performance.** MongoDB, Firefox, and
   Chromium all run detection as an advisory post-commit trend monitor
   feeding a triage loop (human sheriff or automated bisection). Small-team
   tools (Bencher, Nyrkiö, Conbench, rustc-perf) all default to advisory
   too; hard-fail is always an opt-in flag.
3. **Change-point detection is retroactive** — it needs 3–12 post-change
   points, so it flags regressions days after merge. Per-PR verdicts use a
   different tool: outlier-test the candidate run against the *stable
   region* of main's history (MongoDB uses GESD for exactly this).
4. **Statistics cannot fix unstable hardware.** Hunter saw ~3 false change
   points/month on noisy hardware and zero after moving to repeatable
   machines. The proven mitigation for device farms is *canary
   workloads* — tests that exercise the rig, not the software — plus
   automatic board quarantine when the canary shifts.
5. **Device farms separate infra failure from regression structurally**,
   not statistically: an explicit error taxonomy raised at fault time
   (LAVA's InfrastructureError vs JobError vs TestError), bounded
   per-action retries for flaky flash/boot steps, automatic offlining on
   health-check failure with *manual* recovery, and explicit quarantine
   lists for known-flaky tests.
6. **MobiusSmarts already contains the right detector stack** — the gap is
   plumbing: pulling the device's stored metric history off per run (the
   exact Mobius export mechanism is in flux), a per-commit metric store,
   and a thin verdict layer that applies the detectors along the *commit*
   axis instead of the *wall-clock* axis.

---

## 1. The statistics: what actually works on noisy benchmark series

### Thresholds fail measurably

MongoDB originally compared each result against the previous run, a
week-ago run, and the last stable release with a static threshold (usually
10%). Result: ~2,393 auto-filed tickets over 5 months of which ~24 were
useful — up to 99% false positives, roughly 100 notifications per
actionable issue — while simultaneously *missing* real regressions smaller
than the threshold on quiet tests. The Hunter paper adds that thresholds
must be set per-test and re-tuned every time an improvement establishes a
new baseline. ([MongoDB, ICPE 2020](https://arxiv.org/abs/2003.00584);
[Hunter, ICPE 2023](https://arxiv.org/abs/2301.03034);
[Daly, ICPE 2021](https://www.researchgate.net/publication/350795655_Creating_a_Virtuous_Cycle_in_Performance_Testing_at_MongoDB))

This matters for MobiusSmarts because the library's whole calibration
philosophy (one false-alarm budget, thresholds derived per detector via
ARL math, baselines fitted from history) is the *answer* to the same
disease — per-metric hand-tuned thresholds — appearing in a different
host body.

### The convergent method: E-Divisive change-point detection

MongoDB, DataStax's Hunter, and its Apache successor
[Otava](https://github.com/apache/otava) all use the E-Divisive family
(Matteson & James): nonparametric, no distributional assumption beyond a
finite mean, divergence between segment distributions with recursive
segmentation — so it catches variance and shape changes, not just mean
shifts. Adopting it took MongoDB from ~100 notifications per tracked issue
to 9 notifications yielding 6 worth follow-up and 4 tracked issues, and
turned triage from more than a full-time job into a part-time rotation.
([MongoDB](https://arxiv.org/abs/2003.00584);
[Otava math docs](https://otava.apache.org/docs/math/))

Hunter's refinements for CI practicality:

- Replaced E-Divisive's randomized-permutation significance test with a
  **Student's t-test** — deterministic output run-to-run and an
  order-of-magnitude faster (authors' internal profiling).
- **Rejected Mann-Whitney U** because it needs ~30 points to be
  conclusive; the t-test variant finds changes in series of only **4–7
  points** — critical when each commit contributes one data point.
- A related trap: MongoDB's original E-Divisive implementation was
  non-deterministic in isolation and worked partly because their CI layer
  permanently remembered triaged points. Relevant if implementing
  E-Divisive from scratch rather than the t-test variant.
  ([Hunter](https://arxiv.org/abs/2301.03034))

### False-alarm control is a small, layered knob set

Hunter/Otava control alert volume with, in order
([Hunter](https://arxiv.org/abs/2301.03034);
[Otava](https://github.com/apache/otava)):

1. **p-value threshold** — typically 0.05; Otava notes 0.01 ≈ at most ~1%
   false positives. (CLI: `--p-value`.)
2. **Minimum relative-change filter** — ~5%; discards statistically
   significant but non-actionable changes. (CLI: `--magnitude`.) Nyrkiö's
   hosted defaults: p = 0.001, magnitude floor 5%.
3. **Alert recency cap** — notifications limited to change points from the
   last 7 days (built-in dedup: a change alerts once, then becomes part of
   the baseline).
4. **Window sizing** — widening the analysis window from ~14 to ~30 daily
   points measurably reduced false positives.

False-*negative* rates are acknowledged as data-dependent and hard to
estimate.

The MobiusSmarts analogue is direct: these four knobs are the ad-hoc
version of what `false_alarm_every:` + `resolution:` already derive
principally through ARL math. A CI integration should expose the same
single budget ("this pipeline may cry wolf about once a month") and derive
p-value/magnitude equivalents from it, rather than re-importing per-metric
tuning.

### Changepoint is for trends; PR verdicts need an outlier test

Change-point detection is inherently retroactive. MongoDB's minimum
cluster size N=3 means ≥3 post-change points before detection — in
practice alerts arrive **5–6 days after commit**. Mozilla's Perfherder
requires 12 future data points before alerting (up to 10 days on low-push
branches, mitigated by manually retriggering jobs to mint points faster).
For the "is this one new result a regression?" question, MongoDB
explicitly uses **outlier detection (GESD) against the stable region
between the last change points** instead.
([MongoDB](https://arxiv.org/pdf/2003.00584);
[Perfherder docs](https://firefox-source-docs.mozilla.org/testing/perfdocs/perf-sheriffing.html))

The pattern to copy:

- **On main:** maintain a changepoint-segmented history per metric; the
  current stable segment's mean/variance *is* the baseline.
- **On a PR:** outlier-test the candidate run(s) against that stable
  segment. No changepoint math on a single point.
- **Comparing releases** (e.g. new Nerves system vs previous): compare
  *stable-region statistics*, not individual runs. MongoDB filters changes
  under 2 standard deviations and sorts by percent change; this cut
  monthly release review from hours to 30–60 minutes. (They name Welch's
  t-test or Mann-Whitney U as the upgrade path from the simple variance
  rule.)

Baseline selection for PRs (consensus across Conbench and Bencher): compare
against the **fork point** of the branch, not current main HEAD, so
unrelated drift on main isn't attributed to the PR.
([Conbench](https://conbench.github.io/conbench/pages/lookback_zscore.html);
[Bencher](https://bencher.dev/docs/how-to/github-actions/))

### Advisory + triage beats gating

All three mature systems run detection as advisory-plus-triage
([MongoDB](https://arxiv.org/abs/2003.00584);
[Firefox](https://firefox-source-docs.mozilla.org/testing/perfdocs/perf-sheriffing.html);
[Chromium](https://github.com/chromium/chromium/blob/master/docs/speed/addressing_performance_regressions.md)):

- **MongoDB:** a rotating human ("Build Baron") reviews every change point
  and classifies it real / insignificant / noise before any ticket.
- **Firefox:** performance sheriffs backfill data, investigate alerts,
  file bugs against the culprit commit, escalate after 1 business day of
  silence, ultimately request backout — all post-landing.
- **Chromium:** the chrome.perf waterfall continuously benchmarks
  post-commit on real Android/Windows/Mac/Linux hardware; the per-commit
  verdict comes from **automated bisection (Pinpoint)** which auto-files a
  bug assigned to the culprit CL's author. Perf tests are explicitly too
  long-running to gate every revision.
- **rustc-perf:** post-merge run per bors commit, summary posted as a PR
  comment, `perf-regression` label applied (auto-removed if neutral),
  weekly human triage rotation; significance per test from historical
  variance via IQR fencing (`result > Q3 + 3×IQR`), headline metric is
  instruction count *because* it's the most stable — a deliberate
  noise-management choice.
  ([comparison-analysis.md](https://github.com/rust-lang/rustc-perf/blob/master/docs/comparison-analysis.md))

Operational noise hygiene from Firefox: known spurious-alert sources are
catalogued and handled operationally, not statistically — weekend/holiday
low-push periods produce less-noisy points that trigger small (<3%)
Monday-evening alerts; regressions smaller than a test's noise range are
closed as low-value because blame can't be pinned; suites too noisy to
monitor are removed outright. Principle: **prune or annotate known-noisy
metrics rather than letting them erode trust in the alert stream.**
([Mozilla Noise FAQ](https://wiki.mozilla.org/TestEngineering/Performance/Sheriffing/Noise_FAQ) — parts are Talos-era/dated)

The legacy pattern — a checked-in JSON of per-metric improve/regress
bounds with a direction flag (ChromeOS `perf_expectations.json`, baselines
hand-derived from a known-good run ±15%) — is the cheapest possible start
and embodies exactly the manual tuning burden everyone else abandoned.
Fine for week one; do not let it grow.
([ChromeOS, ~2011-13 era](https://www.chromium.org/chromium-os/testing/perf-regression-detection/))

---

## 2. Hardware-in-the-loop: how device farms actually run

*(Targeted-pass findings; sourced, not adversarially verified.)*

### LAVA (Linaro) — the canonical quarantine design

- A **health check** is a special job validating the device *and* the
  infrastructure around it: deploy + boot a frozen known-good ("gold
  standard") image plus a minimal sanity suite — deliberately small, and
  deliberately testing the rig rather than any software under test.
- **Failure ⇒ automatic offlining**: device health goes to *Bad* and the
  scheduler stops using the board "until an admin sets the health to
  either unknown or good." **Quarantine is automatic; un-quarantine is
  manual.** Health checks run per device-type on a time or job-count
  cadence (24 h is common on live instances). A *Looping* admin mode
  soak-tests a board by running health checks back to back.
  ([LAVA health checks](https://docs.lavasoftware.org/lava/healthchecks.html))
- **Three-way error taxonomy raised at fault time**, not parsed from logs
  afterwards: `InfrastructureError` (dispatcher/lab problem — triggers a
  health check so a sick board gets quarantined instead of failing user
  jobs repeatedly), `JobError` (bad job/user parameters), `TestError`
  (failure in the test itself).
  ([LAVA dispatcher docs](https://docs.lavasoftware.org/lava/dispatcher-testing.html))
- **Retries are bounded and scoped to the flaky layer**: job YAML supports
  `failure_retry: N` on deploy/boot/test actions individually. Whole-job
  resubmission is not automatic; failure triage is human-assisted via
  failure tags/comments.
- Distinguishing bootloader-infra failures from kernel-didn't-boot remains
  a known hard problem even for LAVA/KernelCI.

### KernelCI — bisection with a human gate

Automated bisection runs per boot regression with platform + lab + config
held constant. Build failures and boot-test false positives can mislead
it, so **every bisection result is manually verified before being
reported** — "false positives in this area can be very harmful."
([Collabora](https://www.collabora.com/news-and-blog/blog/2018/01/16/kernelci-automated-bisection/);
[Linaro/Maestro](https://www.linaro.org/blog/scaling-linux-kernel-quality-automating-bisection-in-kernelci/))
Boards known broken on some kernel/config combos are blacklisted at
onboarding, and Collabora's 2024 retrospective makes consistency a formal
test-quality criterion after a boot-test suite "generated many false
positives."

### balena — the closest analogue to a Nerves flow

- **Autokit**: a USB-hub kit of off-the-shelf parts per device-under-test —
  SD-card multiplexer (flash from host, then flip the card to the DUT; no
  hands), mains relay for power cycling, USB-UART for serial capture, HDMI
  capture, Ethernet. Non-SD targets use alternate strategies
  (`generic-flasher` for eMMC, a relay for boot-mode pins).
  ([autokit](https://github.com/balena-io-hardware/autokit-info-doc);
  [Leviathan quickstart](https://balena-os.github.io/leviathan/documents/quickstart.quickstart_autokit.html))
- **PR vs release split**: per-PR runs on QEMU (cheap, horizontal,
  deterministic); after merge, the *same suites* run on physical hardware
  per device type, and those runs gate the production release of that
  device type's OS version. Same-tests-both-ways is a deliberate design
  property of their Leviathan framework.
  ([meta-balena tests](https://github.com/balena-os/meta-balena/tree/master/tests))
- The test rigs are themselves balena devices in one balenaCloud fleet
  across multiple geographic locations — the farm is managed by the
  product it tests.
- Artifacts land in a `workspace/reports` directory per run. No documented
  automatic retry policy.

### Zephyr twister — quarantine lists as code

- A YAML **hardware map** inventories attached boards (probe serial, port,
  platform, runner); twister builds, flashes (`west flash` by default),
  and parses serial output for verdicts. `--device-flash-timeout` bounds
  flashing; `flash_before: True` works around USB-CDC boards that drop
  serial during flash — a concrete example of flash-vs-test failure
  entanglement at the USB layer.
- `--retry-failed N` / `--retry-interval` for bounded re-runs;
  `--retry-build-errors` extends to builds.
- **`--quarantine-list <yaml>`**: known-flaky tests/platforms are skipped
  and *marked quarantined* in results, each entry carrying a comment (link
  to the tracking issue). **`--quarantine-verify` inverts the list** —
  runs only quarantined tests to check whether they've recovered.
  ([twister docs](https://docs.zephyrproject.org/latest/develop/test/twister.html))
- Real-world small-scale shape (Golioth): self-hosted GitHub Actions
  runners with boards attached, stable `/dev/serial/by-id/...` paths,
  credentials in a local env file on the runner rather than cloud secrets.
  ([golioth/zephyr_twister_hil_testing](https://github.com/golioth/zephyr_twister_hil_testing))

### Small-scale patterns worth stealing

- **Ferrous Systems** run HIL on a self-hosted runner (an RPi4) in a
  *private mirror repo* because fork PRs can run arbitrary code on
  self-hosted runners; a proxy action reposts logs to the public PR.
  On-target test failure propagates to `cargo test` exit status, so the
  workflow fails naturally.
  ([Ferrous blog](https://ferrous-systems.com/blog/gha-hil-tests/))
- **Memfault/Interrupt's pragmatic split**: emulation (Renode + Robot
  Framework) for every-PR coverage; scarce real hardware reserved for a
  few high-value cases — firmware update, factory reset, power, basic
  hardware function.
  ([Interrupt](https://interrupt.memfault.com/blog/test-automation-renode))
- A recurring war story: unstable USB hub power resets boards or
  interrupts flashing mid-process, and the CI log just says "connection
  lost" — power delivery is part of the rig's reliability budget.

### The unanimous hardware lesson

Hunter's case study (section literally titled "Change Point Detection
Cannot Fix Noisy Data"): on untuned data-center hardware with ±10% median
latency swings, Hunter flagged ~3 change points/month — **all false
positives**; after migrating to repeatable hardware, zero.
MongoDB's structural answer is **canary tests** — workloads that exercise
the testbed rather than the software, whose results should never change;
a canary shift indicts the infrastructure. They built GESD outlier
detection on canaries to auto-rerun suspect runs and then **disabled the
automated rerun** ("not solving our problem, but costing us money") while
keeping the canary computation for triage.
([Hunter](https://arxiv.org/abs/2301.03034);
[Daly 2021](https://www.researchgate.net/publication/350795655_Creating_a_Virtuous_Cycle_in_Performance_Testing_at_MongoDB))

---

## 3. Noise control on Raspberry-Pi-class hardware

*(Targeted-pass findings; sourced, not adversarially verified.)*

Three complementary layers appear in every mature setup:

**Prevent** (control the clocks and heat):

- Fix the CPU frequency: `performance` governor
  (`cpupower frequency-set --governor performance`) or on Pi
  `force_turbo=1` + explicit `arm_freq` in config.txt — constant clock,
  still subject to the 85 °C hard ceiling.
- Pi thermal profile: soft throttle at **60 °C** (Pi 3B+/4 firmware drops
  to a sustained mid-clock to avoid the 80–85 °C hard throttle); Pi 4
  steps 1.5 GHz → 1.0 GHz → 750 MHz worst case.
- **Firmware version is a benchmark variable**: time-to-throttle under
  synthetic load on the same Pi 4 board went 65 s (launch firmware) →
  177 s (beta firmware) across 2019's updates; record firmware/EEPROM
  version in run metadata. Realistic bursty loads (a kernel compile) may
  never throttle at all — synthetic stress overstates the risk.
  ([Pi 4 thermal testing](https://www.raspberrypi.com/news/thermal-testing-raspberry-pi-4/))
- Heatsink/airflow, decent PSU (under-voltage throttles too), network
  isolation, disable background services. Android's equivalents:
  `lockClocks` (root-only, microbenchmarks) and sustained performance
  mode as the non-root fallback; dynamic clocks are named their #1 noise
  source. ([Android microbenchmark docs](https://developer.android.com/topic/performance/benchmarking/microbenchmark-overview))

**Detect and discard** (invalidate contaminated samples):

- **`vcgencmd get_throttled`** is a ready-made contamination detector:
  bits 0–3 = under-voltage / freq-capped / throttled / soft-temp-limit
  *now*; sticky bits 16–19 = *has occurred since boot*. Check sticky bits
  after a run; if set, discard the sample and re-run rather than feeding
  it to the detector. Companions: `vcgencmd measure_clock arm`,
  `vcgencmd measure_temp`.
- Android's library does precisely this pattern: a `ThrottleDetector`
  runs a known workload between benchmarks; on detection it sleeps 90 s
  and retries, up to 3 times, and reports the sleep time in the output.
- Record the control state with every result: Android's output JSON
  carries `cpuLocked`, `sustainedPerformanceModeEnabled`, max clock, core
  count, device fingerprint alongside min/median/max **and the raw runs**.

**Absorb** (statistics over what remains):

- Prefer median (or minimum, for latency-like measures — noise is
  one-sided additive: results come in slower than truth, never faster).
- Android trades per-commit sample count for **temporal aggregation**:
  ~5–10 macro iterations per commit, then step-fitting across commits
  (Skia Perf approach, WIDTH = 5 commits before/after, THRESHOLD = 25)
  with error-weighted comparison so nanosecond microbenchmarks and
  high-variance macro benchmarks share one system. Presubmit guidance:
  5+ runs with and without the patch — "single runs don't give us
  sufficient confidence."
  ([Android Medium post](https://medium.com/androiddevelopers/fighting-regressions-with-benchmarks-in-ci-6ea9a14b5c71))
- Sample-count reference points: hyperfine defaults to ≥10 runs and ≥3 s
  total with warmup/prepare options and built-in outlier warnings; Go
  practice is `-count=10` minimum, ideally 20, compared with benchstat
  (Mann-Whitney U, α = 0.05, insignificant deltas rendered `~`);
  Kalibera & Jones' rigorous-benchmarking guidance is 20–30 repetitions
  at the top level, dimensioned from an initial variance-estimation
  experiment.
- Quantified stakes (CodSpeed): with a 2% regression gate, a 2.66%
  coefficient of variation (GitHub-hosted runners) gives ~45% false-alarm
  probability per run; 0.56% CV (tuned bare metal) gives ~1 in 2500.
  Variance reduction *is* false-alarm reduction.
  ([CodSpeed](https://codspeed.io/blog/benchmarks-in-ci-without-noise))

---

## 4. Artifacts and reports per run

What the surveyed systems persist per run, converging on a common shape:

| Artifact | Who | Why |
|---|---|---|
| Raw samples **and** summary stats (min/median/max + runs array) | Android, Conbench | re-analysis without re-running; outlier exclusion later |
| Environment/control metadata (clocks locked?, max freq, device fingerprint, firmware) | Android, (Pi: `get_throttled`, firmware ver) | classify runs as clean/contaminated; explain variance |
| Serial/boot logs | LAVA, Zephyr twister, balena (UART via autokit) | the only evidence when boot fails; infra-vs-test triage |
| Job/infra logs with error class | LAVA | structural infra-vs-regression separation |
| Canary metric results | MongoDB | indict the rig, not the code |
| Metric time series keyed by commit | everyone doing detection | the input to changepoint/outlier analysis |

Storage can be trivially simple: Otava reads **CSV files**, PostgreSQL,
BigQuery, or Graphite and is "launched by a Jenkins/Cron job to analyze
the recorded metrics regularly"; github-action-benchmark commits a JSON
series to `gh-pages` (which doubles as the dashboard). A small team does
not need a metrics platform to start — a CSV per metric in a repo is a
legitimate v1 that the real tools literally support.

Reporting postures, weakest to strongest:

- **Single-point ratio vs previous run** (github-action-benchmark, default
  alert at 200%) — no statistics; fine as a tripwire, noisy as a gate.
- **PR comment + optional fail flag** (Bencher `--error-on-alert`,
  github-action-benchmark `fail-on-alert` with a separate higher
  `fail-threshold`) — comment at one severity, fail at another.
- **Check/comment from a lookback distribution** (Conbench z-score over
  ≤100 ancestor commits, best-of-repetitions, outliers excluded).
- **Label + triage rotation** (rustc-perf `perf-regression` label,
  auto-removed when neutral; weekly human triage).
- **Changepoint trend monitor + capped notifications** (Otava/Nyrkiö —
  Slack/email/GitHub issues, only persistent shifts, magnitude floor,
  7-day recency cap). Alert dedup is the weakest area across all tools;
  changepoint's "a change alerts once, then joins the baseline" is the
  only structural dedup found.

---

## 5. Mapping onto MobiusSmarts + Nerves

### Two time axes, both already supported

The CI problem has two distinct series, and the library serves both
because the detectors are pure and take plain lists:

1. **Within-run axis** (wall-clock windows during one device session) —
   what MobiusSmarts ships for. In CI this answers: "during the soak run
   on this firmware, did memory drift, did latency go bimodal, did a
   ceiling-ETA appear?" Run the normal runtime on-device during the test
   window, or replay the pulled data off-device.
2. **Per-commit axis** (one summary point per firmware build) — the
   regression-detection series. Treat *runs* as windows: each CI run
   yields scalars (boot time, steady-state memory mean, p99 latency from
   the DDSketch, CPU temp at idle), appended to a per-metric history.
   `Detect.Changepoint` (binary segmentation, BIC penalty) plays the
   Otava role on this series; `Detect.Jump`'s Shewhart limits over the
   current stable segment are the GESD-analogue outlier test for PR
   verdicts; `Detect.Drift`/`Shift` catch the slow degradations
   changepoint segmentation dates after the fact.

The role split mirrors the verified industry pattern exactly:
changepoint = retroactive trend monitor on main; outlier-vs-stable-segment
= per-PR verdict; stable-segment statistics = release-vs-release
comparison for a new Nerves system.

### The artifact is the device's metric history, replayed off-device

The replay half is already designed: `MobiusSmarts.Source.from_metrics/1`
/ `from_summary_windows/1` are pure converters explicitly intended for
tests and replays — hand them the stored windows in whatever form they
came off the device. The extraction half is in flux on the Mobius side
(the old `Mobius.Exports`/MBF module is being removed; a replacement
data-off-device path is planned), so treat "pull the full stored metric
history as one artifact" as the requirement and the wire format as a
detail to settle later. The per-run flow:

```
flash → boot → settle → workload/soak → pull artifacts → power off
artifacts: metric-history dump (format TBD), ring-logger/dmesg dump,
           MobiusSmarts.report/0 text, run metadata (firmware rev,
           Nerves system rev, board id, get_throttled sticky bits,
           measured clock, fw/EEPROM version)
```

Analysis then runs **off-device in plain CI** (`Nx.BinaryBackend` — the
README already names it the right CI backend): replay the dump through
`Source.from_*`, compute the run's scalar summaries, append to the
per-commit store (CSV is fine), run the per-commit detectors, emit the
verdict.

A Nerves-specific trap for boot-time metrics: the system clock warps at
boot until NTP sync, so early-boot Mobius timestamps are unreliable.
Measure boot time from the host side (power-on to first serial heartbeat
/ SSH up) or against the monotonic clock, not from on-device wall-clock
timestamps.

### Proposed flow (small team, low ceremony)

**Per PR (advisory, fast):**
- Build firmware, flash one board (or QEMU first if available — balena's
  QEMU-for-PRs / hardware-for-release split is the proven economizer).
- Boot + smoke: did it boot, within the outlier limits of main's stable
  segment for boot time? 3–5 boots, take the median (hyperfine/Android
  floor is ~5–10 samples; boot iterations are cheap).
- Short soak (10–30 min) with the standard workload; pull the metric
  history; outlier-test each watched scalar against main's current
  stable segment.
- Report as a **PR comment with a table** (every tool surveyed converged
  on this), advisory by default; reserve hard-fail for "did not boot" and
  egregious static ceilings (the one place static thresholds are
  appropriate — Bencher recommends `static` only for should-be-constant
  values).
- Baseline = the PR's **fork point** segment, not main HEAD.

**On main (the real detector, nightly or per-merge):**
- Longer soak per merge/nightly on the board farm; append scalars to the
  store; run `Detect.Changepoint` + `Drift` over the per-commit series.
- Notify (PR comment on culprit range / Slack / issue) only for new change
  points with magnitude above a floor (~5% is the industry default) within
  a recency window (~7 days). One false-alarm budget for the whole
  pipeline, derived the way the runtime already derives detector
  thresholds.
- Expect 3+ points of latency before a changepoint confirms; that is the
  design, not a bug — the PR-level outlier check is the fast path.

**Board health (the canary layer):**
- Each board runs a fixed canary workload per session (or per day, LAVA
  health-check style) on a *frozen known-good firmware* — gold-standard
  image, never updated casually.
- Track canary scalars per board with the same detectors. A canary change
  point ⇒ **automatically quarantine the board** (stop scheduling it),
  un-quarantine manually after inspection — LAVA's asymmetry is the proven
  policy. Do not auto-rerun suspect runs in a loop (MongoDB tried;
  disabled it as cost without benefit). Keep a Zephyr-style quarantine
  list with issue links, and a `--quarantine-verify`-style mode to test
  recovery.
- Classify failures at fault time, LAVA-style: flash/serial/power failure
  (infra — retry bounded, e.g. `failure_retry: 2`, then quarantine) vs
  firmware didn't boot (regression — never auto-retry into silence) vs
  test failure. Never let an infra failure file a regression.

**Per-run hygiene:**
- Pin clocks (`force_turbo=1` + `arm_freq`, or performance governor) and
  cool the boards; record firmware/EEPROM versions.
- After every run read `vcgencmd get_throttled` sticky bits; a throttled
  or under-volted run is discarded and re-run once, not fed to detectors.
- Store raw windows (the metric-history dump *is* this) + scalar
  summaries + environment metadata; never just the verdict.

### What MobiusSmarts might grow (gaps this research exposes)

- A thin **per-commit verdict helper**: given a metric history (list of
  per-run scalars) and a candidate value, return
  outlier-vs-stable-segment verdict + the segment stats — composing
  `Detect.Changepoint` (find segments) with `Detect.Jump` limits (test the
  point). All ingredients exist; the composition and its calibration
  (budget → per-pipeline thresholds across few-point series) do not.
  Note Hunter's floor: ~4–7 points minimum for the t-test variant;
  segment-based outlier tests have a similar warm-up — the doc should say
  what happens for a 3-run-old metric (answer: nothing fires; learning,
  exactly like the runtime's learning state).
- A **metric-dump → run-summary** extraction recipe (livebook or mix
  task): boot-time, steady-state means, sketch percentiles,
  flatline/anomaly flags from the within-run detectors. Blocked on /
  shaped by whatever replaces `Mobius.Exports` as the data-off-device
  path.
- A documented **canary recipe**: which metrics, frozen firmware, and the
  quarantine policy, as a guide rather than code.

## Caveats and refuted claims

- One claim was killed in verification (0–3): that Perfherder alerts via
  per-framework percentage-magnitude thresholds (2% Talos, 0.25% AWSY,
  etc.). **Do not cite those numbers.** Treeherder source suggests a
  t-statistic cutoff (`detect_changes(..., fore_window=12,
  t_threshold=7)`), but Perfherder's actual defaults remain unverified,
  and Mozilla flags the 12-point rule as under re-evaluation.
- The quantitative wins (99% FP, 100:1→9:6:4, hours→30–60 min) are
  first-party experience reports, peer-reviewed at ICPE but
  self-measured.
- Sections 2 and 3 (device farms, SBC noise) come from the targeted
  follow-up pass: sourced but not adversarially cross-verified; the
  ChromeOS threshold doc is ~2011–2013 legacy, parts of Mozilla's noise
  FAQ are Talos-era, and docs.lavasoftware.org/blog.balena.io block
  fetchers, so some LAVA/balena quotes came via mirrors and search
  snippets.
- Not covered anywhere in depth: Memfault's internal HIL practices,
  Android device-lab quarantine policy, and LuaJIT benchmarking practice.

## Primary sources

- MongoDB change-point detection in CI (ICPE 2020): https://arxiv.org/abs/2003.00584
- Daly, *Creating a Virtuous Cycle in Performance Testing at MongoDB* (ICPE 2021): https://www.researchgate.net/publication/350795655
- Hunter (DataStax, ICPE 2023): https://arxiv.org/abs/2301.03034
- Apache Otava: https://github.com/apache/otava · https://otava.apache.org/docs/math/
- Chromium perf regression workflow: https://github.com/chromium/chromium/blob/master/docs/speed/addressing_performance_regressions.md
- Firefox perf sheriffing: https://firefox-source-docs.mozilla.org/testing/perfdocs/perf-sheriffing.html
- rustc-perf comparison analysis: https://github.com/rust-lang/rustc-perf/blob/master/docs/comparison-analysis.md
- LAVA health checks: https://docs.lavasoftware.org/lava/healthchecks.html
- KernelCI automated bisection: https://www.collabora.com/news-and-blog/blog/2018/01/16/kernelci-automated-bisection/
- balena OS testing / autokit: https://blog.balena.io/from-pr-to-release-os-testing-at-balena/ · https://github.com/balena-io-hardware/autokit-info-doc
- Zephyr twister: https://docs.zephyrproject.org/latest/develop/test/twister.html
- Android benchmarking in CI: https://developer.android.com/topic/performance/benchmarking/benchmarking-in-ci · https://medium.com/androiddevelopers/fighting-regressions-with-benchmarks-in-ci-6ea9a14b5c71
- Pi 4 thermal testing: https://www.raspberrypi.com/news/thermal-testing-raspberry-pi-4/
- CodSpeed on CI noise: https://codspeed.io/blog/benchmarks-in-ci-without-noise
- Bencher thresholds: https://bencher.dev/docs/explanation/thresholds/
- Conbench lookback z-score: https://conbench.github.io/conbench/pages/lookback_zscore.html
- Nyrkiö change-detection action: https://github.com/nyrkio/change-detection
- Kalibera & Jones, *Rigorous Benchmarking in Reasonable Time*: https://kar.kent.ac.uk/33611/
