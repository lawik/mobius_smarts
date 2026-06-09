defmodule MobiusSmarts do
  @moduledoc """
  On-device health monitoring over
  [Mobius](https://github.com/mobius-home/mobius) metric storage.

  Three layers, each usable without the ones above it:

  - **`MobiusSmarts`** (this module) — the opinionated runtime: a
    supervision tree that polls Mobius on an interval, runs the
    detector stack over what it finds, and maintains *findings* and an
    aggregate *health level*. One config list of metrics, one
    false-alarm budget; everything else is calibrated from the
    device's own stored history.
  - **`MobiusSmarts.Detect`** — the pure detector stack: control
    charts, CUSUM, EWMA, robust trend + time-to-threshold,
    change-point segmentation, distribution-shape drift, multivariate
    self-novelty. Each detector's blind spot is another's specialty;
    the `Detect` moduledoc maps them out.
  - **`MobiusSmarts.Source`** — Mobius data (via `Mobius.Data`) as Nx
    tensors.

  General-purpose tensor recipes live under `MobiusSmarts.Recipes`.
  This library does not pick an Nx backend; `nx_eigen` is recommended
  on Nerves targets, and `Nx.BinaryBackend` works everywhere at RRD
  scale.

  ## Getting started

  Add the runtime to your supervision tree next to Mobius:

      children = [
        {Mobius, metrics: metrics()},
        {MobiusSmarts,
         config: [
           watch: [
             "vm.memory.used_percent",
             [metric: "disk.used_percent", ceiling: 95.0],
             [metric: "cpu.temp_c", ceiling: 85.0]
           ],
           false_alarm_budget: {1, :week}
         ]}
      ]

  With no `:config` option, configuration is read from the
  `:mobius_smarts` application environment (same keys). Pass `:name`
  and an explicit `:config` to run several instances side by side.
  See `MobiusSmarts.Config` for every key and its default.

  ## What happens, automatically

  Each watched metric starts in **learning**: once enough healthy
  windows exist (changepoint-checked so a regime change is never
  averaged into a baseline, outlier-trimmed so a spike never inflates
  one), a baseline is fitted from the device's own history. From then
  on, every `:interval` the fast detectors run (sudden jumps, rising
  within-window wobble, sustained shifts, slow drifts — plus the
  cross-metric novelty score), and every `:sweep_interval` the slow
  ones do (ETA-to-ceiling projections, retrospective change points,
  distribution-shape drift), with quiet-period baseline refits.

  Reporting gaps are first-class: a metric going quiet raises a
  `:silent` condition, a past gap is recorded as a `:reporting_gap`
  observation, and detectors re-anchor after long gaps rather than
  hallucinate continuity across a reboot.

  Detection thresholds are not configured per detector — they derive
  from the single `:false_alarm_budget` via each detector's
  average-run-length math (`MobiusSmarts.Calibrate`).

  ## Findings and health

      MobiusSmarts.status()
      #=> %{level: :watch, since: 1759300000, concern: 1.2,
      #     findings: [%MobiusSmarts.Finding{kind: :drifting_up, ...}],
      #     learning: [], updated_at: ...}

      MobiusSmarts.findings()      # active conditions, worst first
      MobiusSmarts.observations()  # recent annotations (spikes, gaps, regime changes)

  Each `MobiusSmarts.Finding` carries its diagnosis: the onset (when
  the trouble *began*, not when it was caught), a `concern` ratio
  comparable across detectors, detector evidence, and a one-line
  human message. See `MobiusSmarts.Finding` for the full kind table
  and lifecycle.

  The aggregate level is a deliberate enum, not a score:

  - `:ok` — no active conditions.
  - `:watch` — warning-level conditions active.
  - `:degraded` — a critical condition, or 3+ concurrent warnings.
  - `:critical` — resource exhaustion has a date (a critical
    `:approaching_limit`).

  Levels rise immediately and fall with hysteresis (`:clear_after`).

  ## Telemetry

  - `[:mobius_smarts, :finding, :raised | :escalated | :cleared]` —
    measurements `%{concern: float}`, metadata
    `%{finding: %MobiusSmarts.Finding{}, instance: name}`.
  - `[:mobius_smarts, :health, :level_changed]` — measurements
    `%{concern: float}`, metadata
    `%{level: atom, previous: atom, instance: name}`.

  Attach handlers to alert, persist, or forward findings wherever they
  should go — e.g. set/clear [Alarmist](https://hex.pm/packages/alarmist)
  alarms keyed by finding kind.
  """

  use Supervisor

  alias MobiusSmarts.{Board, Config, Sweeper, Watcher}

  @doc """
  Start a monitoring instance.

  Options:

  - `:name` — instance name (default `MobiusSmarts`); also the handle
    for `status/1`, `findings/1`, `observations/1`.
  - `:config` — keyword configuration (see `MobiusSmarts.Config`).
    Defaults to the `:mobius_smarts` application environment.
  """
  @spec start_link(keyword()) :: Supervisor.on_start()
  def start_link(opts \\ []) do
    name = Keyword.get(opts, :name, __MODULE__)

    config =
      opts
      |> Keyword.get_lazy(:config, fn -> Application.get_all_env(:mobius_smarts) end)
      |> Config.new!()

    Supervisor.start_link(__MODULE__, {name, config}, name: :"#{name}.Supervisor")
  end

  @impl Supervisor
  def init({name, config}) do
    children = [
      {Board, name: name, config: config},
      {Watcher, name: name, config: config},
      {Sweeper, name: name, config: config}
    ]

    # The Board owns the ETS table; if it goes, its dependents restart
    # too and the world is rebuilt from the Mobius-persisted RRD.
    Supervisor.init(children, strategy: :rest_for_one)
  end

  @doc """
  The instance's aggregate health: `%{level: :ok | :watch | :degraded
  | :critical, since: unix_seconds, concern: float, findings: [...],
  learning: [metric names still baselining], updated_at: ...}`.
  """
  @spec status(atom()) :: map()
  def status(name \\ __MODULE__), do: Board.status(name)

  @doc "Active conditions, worst first."
  @spec findings(atom()) :: [MobiusSmarts.Finding.t()]
  def findings(name \\ __MODULE__), do: Board.findings(name)

  @doc "Recent observations (spikes, reporting gaps, regime changes), newest first."
  @spec observations(atom(), pos_integer()) :: [MobiusSmarts.Finding.t()]
  def observations(name \\ __MODULE__, limit \\ 50), do: Board.observations(name, limit)

  @doc "The fitted baseline for a watched metric, or `nil` while learning."
  @spec baseline(atom(), String.t(), map()) :: map() | nil
  def baseline(name \\ __MODULE__, metric, tags \\ %{}), do: Board.baseline(name, {metric, tags})
end
