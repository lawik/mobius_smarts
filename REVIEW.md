# REVIEW.md — full-codebase review

Reviewed 2026-06-09, at b964cdc. Full read of `lib/` (~2.9k lines) and the
project files, suite run, plus numerical probes for the new findings
(reproduction snippets inline below). Third review pass on this codebase:
CRITIQUE.md (2026-06-05) audited the detector math, QUALITY.md (2026-06-06)
audited the runtime and hygiene. This pass re-verified both against current
code and hunted for what they missed.

**Checks at review time:** `mix test` — 48 doctests, 16 properties, 138
tests, 0 failures.

## Verdict

The math that the two earlier reviews pinned down is genuinely correct — I
re-derived the core formulas independently and found no errors in: Jump's
X̄/S limits and the `c4` approximation, the dof-weighted pooled sigma, the
CUSUM reflection identity and its onset logic (batch and streaming forms
agree, including the off-by-one), Siegmund's ARL numbers, the EWMA
time-varying variance, Mann–Kendall's `S = pairs − ties − 2·inversions`
identity, its null variance and continuity correction, the merge-sort
inversion counter, Theil–Sen's Conover intercept, discrete W1/JSD/PSI, the
Mahalanobis Cholesky-solve, Isolation Forest's `c(n)` and score, and all of
Calibrate (Bonferroni split, two-sided tail inversion, Siegmund bisection,
Wilson–Hilferty). The test suite really does pin these to textbook
constants.

There is, however, **one genuine crime against math that both prior
reviews missed** (N1 below): `Changepoint` violates the float-precision
discipline its sibling modules document and defend against, and
independently uses the cancellation-prone textbook SSE formula — the same
"naive sum-of-squares" failure CRITIQUE R5 flagged *upstream* in Mobius
while declaring the in-library code clean. It silently corrupts results
inside the documented use envelope. The rest of the new findings are
smaller: one real bug in a recipe kernel, doc drift from this week's
dependency change, and a handful of robustness gaps.

Most prior findings are still open (status table at the end). The repo is
accumulating review documents faster than fixes — CRITIQUE O1 (the sklearn
exporter that produces models `load!/1` rejects) has now survived two
reviews with the fix being a one-line list comprehension in a docstring.

---

## New findings

### N1. `Changepoint` silently misses real changes and reports phantom ones for large-mean series — **High**

Two compounding numerical failures in `best_split/2`
(`lib/mobius_smarts/detect/changepoint.ex:126-162`), both invisible to the
test suite because every test uses small-magnitude values.

**(a) The f32 scalar-wrap bug.** `total1` and `total2` are pulled out as
Elixir floats (`changepoint.ex:131-132`) and then fed back into tensor
expressions as bare scalars (`changepoint.ex:143-146`). Nx wraps bare
floats as **f32** before the binary op, so both totals lose everything
below ~2⁻²⁴ relative — `Nx.subtract(total2, left_sq)` injects an absolute
error of `total2 · ~6e-8` into every per-split cost. Jump, Drift, and
Shift all carry explicit `f64/1` wrapper helpers with comments naming this
exact trap ("bare floats are wrapped as f32 by Nx and silently cost
precision" — `drift.ex:137-138`); Changepoint is the one module that
forgot.

**(b) Textbook-formula cancellation.** Even with the scalar bug fixed,
`SSE = Σx² − (Σx)²/len` over prefix sums cancels catastrophically once
`mean²/variance` approaches f64 precision: at values around 1e9 (bytes),
`Σx² ≈ 6e19` and the cumulative-sum rounding (~ulp 1.6e4 per element)
swamps gains of order `n·Δ²`.

**Demonstrated** (the module's own doctest series, shifted by a constant —
SSE gains are shift-invariant in exact arithmetic, so the answer must not
change):

```elixir
series = List.duplicate(10.0 + offset, 30) ++ List.duplicate(20.0 + offset, 30)
Changepoint.detect(series)
# offset 0.0    => [30]        correct
# offset 1.0e6  => []          10-unit step MISSED (f32 wrap: cost error 406376)
# offset 1.0e8  => [30, 45, 55]  found, plus two phantoms
# offset 1.0e9  => [50, 55]    true changepoint LOST, two phantoms reported
```

With realistic noise (σ=1, 3-unit step at index 60, n=120): offset 0
detects `[60]`; offset 1e9 returns `[33, 110, 115]` — pure fiction.

The break-even is roughly `mean/Δ ≳ 4000` for the f32 path: a step smaller
than ~0.025% of the level is invisible, and the phantom regime starts well
before that. Percent-scale metrics (the README examples) are safe;
bytes-scale gauges and rate-converted counters (`mean_rate` on `bytes_rx`
— a documented data shape in SPEC.md) are squarely in the failure zone.

**Blast radius beyond `detect/2`:** `Analysis.settled_segment/1` uses
Changepoint to avoid "averaging across a regime change" when fitting
baselines (`analysis.ex:111-116`) — for large-magnitude metrics that guard
is quietly inert (missed changes) or shreds the baseline pool (phantoms →
`:unsettled` forever). `Analysis.changepoint_candidates/1` emits the
phantoms as `:regime_change` observations with confident timestamps and
before/after means.

**Fix (small):** SSE gains are shift-invariant, so center once on entry —
`values = Nx.subtract(values, Nx.mean(values))` at the top of `detect/2`
(or `best_split/segment`) kills both failure modes for any realistic
magnitude; verified that this restores `[30]` at offsets up to 1e12. Keep
`total1`/`total2` as f64 tensor slices (or wrap with the same `f64/1`
helper the siblings use) rather than round-tripping through Elixir floats.
Then add a regression test that asserts shift-invariance,
e.g. `detect(xs) == detect(xs .+ 1.0e9)`.

### N2. `Recipes.DefnKernels.run_lengths/2` swallows a leading on-run — **Medium**

`run_ids/1` (`defn_kernels.ex:99-109`) labels runs by counting *rising
edges of the diff*, so a series that **starts** in state 1 has no rising
edge for its first run — it gets id 0, the same id as 0-samples, and
`run_lengths/2` (`defn_kernels.ex:126-140`) then merges it into bucket 0,
which the docs say "holds the count of 0-samples and is usually ignored":

```elixir
state = Nx.tensor([1, 1, 0, 1, 1, 1, 0], type: :u8)
DefnKernels.run_ids(state)            #=> [0, 0, 0, 1, 1, 1, 1]
DefnKernels.run_lengths(state, 2)     #=> [4, 3, 0]
# Reality: two on-runs of lengths 2 and 3. The leading run vanished into
# bucket 0 (4 = 2 on-samples + 2 off-samples) and bucket 2 is empty.
```

For the advertised use ("time-in-state for a binary signal" — relay duty,
link-up runs) a signal that is on at the start of the window is the common
case, not the corner. Fix: prepend a virtual 0 before diffing
(`Nx.concatenate([Nx.tensor([0]), state])` then diff), so a leading 1-run
gets a real rising edge; the doctest example happens to start at 0, which
is why this was never caught — add a leading-1 doctest.

### N3. `run_ids/1` docstring overclaims — **Low**

"Per-sample tensor where each entry is the index of the run it belongs to"
(`defn_kernels.ex:88-97`) is false for 0-samples: they share the id of the
*preceding* 1-run (`[0,1,1,0,...]` → sample 3 gets id 1). The actual
semantics is "number of 1-runs started so far," which is fine for the
documented mask-and-tally recipe but will burn anyone who uses it for
0-runs, which the moduledoc explicitly suggests ("mask out the
`0`-runs"). Document the real semantics or compute true run ids (count
both edge directions).

### N4. README installation section contradicts `mix.exs` since this week — **Low**

`README.md:50-52` says the project "tracks the unreleased Mobius
`histograms` branch via a path dependency (`{:mobius, path: "../mobius"}`)"
— but 1ce7a2c and b964cdc (both 2026-06-09) moved the dep to
`{:mobius, github: "mobius-home/mobius", branch: "main"}` (`mix.exs:124`)
precisely because histograms merged. The commit messages updated; the
README didn't. One sentence to fix, and it's the *installation* section —
the first thing a new user acts on.

### N5. Theil–Sen materializes all O(n²) pair slopes as a BEAM list — **Low** (device memory)

QUALITY M3 covers the redundant *CPU* passes; the memory side is
unrecorded: `theil_sen/2` builds the full slope list in one comprehension
(`trend.ex:96-101`) — at the default runtime wiring (`trend_window:
{24, :hour}` at minute resolution, n=1440) that is ~1.04M boxed floats,
roughly 40 MB of process heap per pass, allocated up to 4× per metric per
sweep (M3) — on a Nerves device. Computing the slope count and using a
selection algorithm, or sampled Theil–Sen (CRITIQUE R4 already names it),
fixes both; short of that, the moduledoc's cost section should mention
memory next to the milliseconds.

### N6. `Outlier.load!/1` accepts cyclic trees; `score/2` then loops forever — **Low**

`validate_internal!` checks child indices for *range* but not for
acyclicity or even self-reference (`outlier.ex:350-365`); a node with
`left == node` passes validation and `walk/4` (`outlier.ex:246-258`)
recurses infinitely at score time. The moduledoc promises `load!/1`
"raises … on anything malformed". A real sklearn export can't produce a
cycle, but the loader's whole job is distrusting hand-rolled JSON. Cheap
fix without graph analysis: bound `walk` depth by the node count (a valid
tree path can't be longer) and raise past it.

### N7. Calibration silently assumes tick interval == RRD window cadence — **Low**

`Calibrate.for_config/1` converts the false-alarm budget to windows via
`budget / interval` (`calibrate.ex:59`). The Config docs say `:interval`
"should match the Mobius RRD resolution you want to monitor at"
(`config.ex:22-24`) — but nothing checks, and the actual number of *new
windows per budget* is set by the RRD resolution, not the tick. Monitor a
10-second-resolution metric with the default 1-minute interval and every
threshold is calibrated 6× too permissive a budget (false alarms ~6× the
promise). The data to verify is already in hand: `Analysis.gaps/2`
measures the series' median cadence every tick — comparing it against
`config.interval` and logging once on mismatch would close the gap.

### N8. Wobble-collapsed-to-zero produces absurd `concern` values — **Low**

When the latest window's `std_dev` falls below the wobble LCL, concern is
`lcl / max(std, 1.0e-12)` (`analysis.ex:240-241`) — a window with zero
internal spread (quantized gauge gone flat, sensor stuck) yields concern
~1e12. It's "correctly" critical, but it poisons `status.concern` (a max
across findings, `board.ex:234`) and the cross-detector comparability
story Calibrate exists to provide. Cap it (e.g. at `lcl/wobble-band-width`
or a fixed ceiling) so the field stays meaningful in dashboards.

### N9. `eta_to_threshold/3` anchors the projection on the last raw sample — **Info**

`trend.ex:198-210` projects from `List.last(values)`, not from the fitted
line's value at the last timestamp. The slope is outlier-proof by
construction; the *anchor* is whatever the final window happened to read,
so one noisy window swings the ETA (and with it the warn/critical horizon
classification in `Analysis.trend_candidates/3`). Using
`slope * t_last + intercept` keeps the whole projection on the robust fit.
Deliberate-looking, but the moduledoc sells robustness and this is the
one non-robust link in the chain — worth a sentence either way.

### N10. Hygiene singles — **Low**

- `CHANGELOG.md` is a `TODO: write changelog` stub (fine unreleased, on
  the same publish-day list as the `TODO` GitHub link, L7/O2).
- `Jump.scan/4` with a `counts` entry of 0 produces NaN wobble limits and
  ±inf jump limits in the returned tensors (`jump.ex:210-213` — `c4` =
  4/3, `sqrt(1 − c4²)` of a negative). Unreachable through `Source`
  (zero-report windows contribute no point) but reachable through the
  public detector API; the `counts >= 2` mask hides it from `wobbles` but
  not from the returned `wobble_ucl`/`wobble_lcl`.

---

## Status of prior findings (verified against b964cdc)

| Finding | Status |
|---|---|
| CRITIQUE O1 — sklearn exporter emits `feature = -2` at leaves, `load!/1` rejects | **Open** — `outlier.ex:98` still annotates `t.feature.tolist()` with "`-1 at leaves`" |
| CRITIQUE O2 / QUALITY L7 — `TODO` GitHub link in Hex metadata | **Open** — `mix.exs:73`; `source_url` also still absent from `docs()` |
| CRITIQUE R1 — i.i.d. caveat has no executable tool (no `lag1_autocorrelation/1`) | **Open** — still doc-only |
| CRITIQUE R2 — Drift/Shift don't accept the baseline map; sigma footgun | **Open** — `Drift.scan/2`, `Shift.chart/2` still require manual `:sigma` |
| CRITIQUE R3 — Changepoint sd-fallback sensitivity regression | **Open** — and see N1, which is worse and in the main path |
| CRITIQUE R5 — upstream Mobius naive std_dev accumulator | **Confirmed still upstream** — `deps/mobius/lib/mobius/summary.ex:62`; no caveat in `Source.summary_series/3` docs yet |
| QUALITY M1 — stale/skewed values feed the novelty vector | **Open** — `watcher.ex:113-114` returns `{last_ts, avg}` in the stale branch too |
| QUALITY M2 — `Config.new!/1` validates keys, not values | **Open** — `config.ex:147-157` |
| QUALITY M3 — up to 4 redundant O(n²) Theil–Sen passes per sweep | **Open** — `analysis.ex:405`; see also N5 (memory) |
| QUALITY M4 — refit guard matches name only, ignores tags | **Open** — `sweeper.ex:161` |
| QUALITY M5 — README/SPEC contradict the runtime layer | **Open** — `README.md:45-46` ("the caller's problem"), `SPEC.md` Principles ("No GenServers"); see also N4 (new drift, same document) |
| QUALITY M6 — unused alias warning in test suite | **Fixed** — `analysis_test.exs:4` clean |
| QUALITY L1 — stale `Tensor.from_column` doc reference | **Fixed** — no matches in `lib/` |
| QUALITY L2 — NaN on zero-mass distributions (`mean_with_validity`, `Shape.proportions`) | **Open** |
| QUALITY L3 — `>=` vs `>` alarm inconsistency (Analysis vs Drift) | **Open** — `analysis.ex:322` vs `drift.ex:243` |
| QUALITY L4 — `Jump.scan/4` missing friendly empty-series error; broad `rescue ArgumentError` in `do_fit` | **Open** |
| QUALITY L5 — `Board.status/1` fallback unreachable-or-raising | **Open** — `board.ex:61-64` |
| QUALITY L6 — `nstandard` as a prod dependency | **Open** — `mix.exs:125` |
| QUALITY L7 — `.DS_Store` hygiene | **Partially fixed** — now gitignored; TODO link and `source_url` remain (above) |
| QUALITY L8 — loop cadence drifts by work time | **Open** — `watcher.ex:50`, `sweeper.ex:42` |

QUALITY's four "design observations" (hardcoded-critical `:jumped`, Trend
concern scale, conditions clearing during silence, duplicate watch
entries) all still describe the code accurately.

## What's good

QUALITY.md's "genuinely good" list still stands: the
one-budget-to-all-thresholds calibration story, the honest-limitations
documentation, the conformance/property/scenario test architecture, and
the per-metric failure isolation in the runtime. This pass adds: the
streaming and batch forms of Drift/Shift are *provably* consistent (the
`step/2` onset bookkeeping matches the batch reflection-identity scan
including edge cases), the inversion-counting Mann–Kendall is both correct
and the right algorithm, and Source's narrow `{:error, :unavailable}`
matching is exactly right per `Mobius.Data`'s specs — checked, not
assumed.

## Reproduction

All N-finding probes are one-file `mix run` scripts; the key ones:

```elixir
# N1 — shift-invariance violation
series = List.duplicate(10.0 + 1.0e9, 30) ++ List.duplicate(20.0 + 1.0e9, 30)
MobiusSmarts.Detect.Changepoint.detect(series)   #=> [50, 55] — should be [30]

# N1 — root cause, isolated
t = Nx.tensor([1.0], type: :f64)
Nx.to_number(Nx.subtract(60_001_800_015_000.0, t)) - 60_001_800_014_999.0
#=> 406376.0  (bare-float operand wrapped via f32)

# N2 — leading on-run
MobiusSmarts.Recipes.DefnKernels.run_lengths(Nx.tensor([1, 1, 0, 1, 1, 1, 0], type: :u8), 2)
#=> [4, 3, 0] — should report on-runs of 2 and 3
```
