defmodule MobiusSmarts.Finding do
  @moduledoc """
  One detected issue, with a lifecycle.

  Findings come in two classes:

  - **Conditions** are stateful: raised when a detector confirms
    trouble, updated every tick while it persists, escalated when it
    worsens, and cleared (with hysteresis) once the detector stops
    confirming it. Conditions drive the aggregate health level.
  - **Observations** are point-in-time annotations — a spike that came
    and went, a regime change dated in hindsight, a reporting gap.
    They are recorded and emitted once, never cleared, and never
    affect the health level.

  Every finding carries its *diagnosis*, not just a boolean: the
  `onset` (when the trouble started, not when it was detected), a
  `concern` ratio (distance to/past the calibrated alarm threshold —
  comparable across detectors because all thresholds derive from one
  false-alarm budget), detector-specific `evidence`, and a one-line
  human `message` ready for a log or an alert.

  ## Kinds

  | Kind | Class | Detector | Meaning |
  |---|---|---|---|
  | `:jumped` | condition | Jump | latest window far outside the band right now |
  | `:spiked` | observation | Jump | an earlier window jumped, then returned |
  | `:wobbling` | condition | Jump (S side) | within-window spread above its band — erratic |
  | `:flatlined` | condition | Jump (S side) | spread collapsed below a normally-noisy metric's floor — stuck signal |
  | `:departed` | condition | Departure | a constant metric left its learned constant |
  | `:baseline_stale` | observation | Drift | level moved on both sides of the target; baseline dropped, relearning |
  | `:shifted_up/_down` | condition | Shift | level moved and stayed moved |
  | `:drifting_up/_down` | condition | Drift | slow creep confirmed; onset dated |
  | `:approaching_limit` | condition | Trend | heading for a ceiling/floor; ETA attached |
  | `:shape_drift` | condition | Shape | distribution shape moved off baseline |
  | `:novel_behavior` | condition | Novelty | metric combination outside device habits |
  | `:silent` | condition | — | metric stopped reporting |
  | `:reporting_gap` | observation | — | a past stretch with no windows |
  | `:regime_change` | observation | Changepoint | a dated character change, in hindsight |
  """

  @type kind() ::
          :jumped
          | :spiked
          | :wobbling
          | :flatlined
          | :departed
          | :baseline_stale
          | :shifted_up
          | :shifted_down
          | :drifting_up
          | :drifting_down
          | :approaching_limit
          | :shape_drift
          | :novel_behavior
          | :silent
          | :reporting_gap
          | :regime_change

  @type class() :: :condition | :observation
  @type severity() :: :info | :warning | :critical
  @type status() :: :active | :cleared | :noted

  @type t() :: %__MODULE__{
          metric: String.t(),
          tags: map(),
          detector: atom(),
          kind: kind(),
          class: class(),
          severity: severity(),
          status: status(),
          onset: integer() | nil,
          raised_at: integer(),
          last_seen_at: integer(),
          cleared_at: integer() | nil,
          concern: float(),
          evidence: map(),
          message: String.t()
        }

  @enforce_keys [:metric, :tags, :kind, :class, :severity, :raised_at, :last_seen_at]
  defstruct [
    :metric,
    :tags,
    :detector,
    :kind,
    :class,
    :severity,
    :onset,
    :raised_at,
    :last_seen_at,
    :cleared_at,
    :message,
    status: :active,
    concern: 0.0,
    evidence: %{}
  ]

  @doc """
  The identity of a finding, used for dedup and lifecycle.

  Conditions are identified by what is wrong (`{metric, tags, kind}`):
  the same trouble re-confirmed updates one finding rather than raising
  a stream of duplicates. Observations additionally carry their onset
  (`{metric, tags, kind, onset}`): the spike at 10:32 and the spike at
  11:07 are two records.
  """
  @spec id(t()) :: tuple()
  def id(%__MODULE__{class: :condition} = f), do: {f.metric, f.tags, f.kind}
  def id(%__MODULE__{class: :observation} = f), do: {f.metric, f.tags, f.kind, f.onset}
end
