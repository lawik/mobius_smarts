# Backend shoot-out for the Nx-shaped detector paths.
#
#     elixir bench/nx_backends_bench.exs
#
# The first run downloads/compiles EXLA, EMLX, and NxEigen — give it minutes.
# Knobs:
#
#     BENCH_SKIP="emlx,nx_eigen"  skip deps that won't build on this machine
#     BENCH_N=4096                series length (default 1024)
#     BENCH_TIME=2                seconds of measurement per scenario
#
# What runs: every detector path that is actually Nx — Jump's chart kernel,
# Drift's cumulative-op CUSUM, Shift's band kernel, Changepoint's prefix-sum
# split scan, Shape's distance kernels, Novelty's Cholesky fit/solve.
# Trend is deliberately plain Elixir (see SPEC's Nx strategy) and is not here.
#
# Each algorithm × backend combo is smoke-tested first; combos that raise
# (missing backend callbacks, no f64 on MLX GPU, ...) are reported and
# skipped instead of crashing the run. "EXLA jit" routes the defn kernels
# through the EXLA compiler; note Changepoint's split scan is plain Nx ops
# (not defn), so its jit row should match its eager row.

skip =
  System.get_env("BENCH_SKIP", "")
  |> String.split([",", " "], trim: true)
  |> MapSet.new()

repo = Path.expand("..", __DIR__)

optional = [
  {:exla, "~> 0.7"},
  {:emlx, "~> 0.1"},
  {:nx_eigen, "~> 0.1"}
]

Mix.install(
  [
    {:mobius_smarts, path: repo},
    {:benchee, "~> 1.3"}
  ] ++ Enum.reject(optional, fn {name, _} -> MapSet.member?(skip, to_string(name)) end)
)

defmodule BackendBench do
  alias MobiusSmarts.Detect.{Changepoint, Drift, Jump, Novelty, Shape, Shift}

  def run do
    n = String.to_integer(System.get_env("BENCH_N", "1024"))
    {time, _} = Float.parse(System.get_env("BENCH_TIME", "2"))

    IO.puts("Elixir #{System.version()}, Nx #{Application.spec(:nx, :vsn)}, n = #{n}")

    data = build_data(n)
    scenarios = scenarios()

    IO.puts("candidate scenarios: #{Enum.map_join(scenarios, ", ", &elem(&1, 0))}\n")

    for {algo, input, fun} <- algos(data) do
      IO.puts("\n#{String.duplicate("=", 60)}\n  #{algo}\n#{String.duplicate("=", 60)}")

      usable =
        Enum.filter(scenarios, fn scenario ->
          case smoke(scenario, input, fun) do
            :ok ->
              true

            {:error, message} ->
              IO.puts("  skipping #{elem(scenario, 0)}: #{message}")
              false
          end
        end)

      jobs =
        for {name, backend, compiler} <- usable, into: %{} do
          {name,
           {fun,
            before_scenario: fn _ ->
              Nx.default_backend(backend)
              Nx.Defn.default_options(compiler: compiler)
              transfer(input, backend)
            end}}
        end

      if jobs == %{} do
        IO.puts("  no scenario survived the smoke test — nothing to run")
      else
        Benchee.run(jobs,
          warmup: 1.0,
          time: time,
          print: [configuration: false]
        )
      end
    end

    :ok
  end

  # One candidate row per installed backend, plus EXLA as a defn compiler.
  defp scenarios do
    [
      {"BinaryBackend", Nx.BinaryBackend, Nx.Defn.Evaluator},
      {"NxEigen", NxEigen.Backend, Nx.Defn.Evaluator},
      {"EMLX", EMLX.Backend, Nx.Defn.Evaluator},
      {"EMLX (cpu)", {EMLX.Backend, device: :cpu}, Nx.Defn.Evaluator},
      {"EXLA eager", EXLA.Backend, Nx.Defn.Evaluator},
      {"EXLA jit", EXLA.Backend, EXLA}
    ]
    |> Enum.filter(fn {_name, backend, compiler} ->
      loaded? = fn
        {mod, _opts} -> Code.ensure_loaded?(mod)
        mod -> Code.ensure_loaded?(mod)
      end

      loaded?.(backend) and loaded?.(compiler)
    end)
  end

  # Run the algo once in an isolated, unlinked process with the scenario's
  # backend/compiler; anything that raises disqualifies the combo. Also
  # pre-warms EXLA's compilation cache for the jit scenario.
  defp smoke({_name, backend, compiler}, input, fun) do
    {pid, ref} =
      spawn_monitor(fn ->
        Nx.default_backend(backend)
        Nx.Defn.default_options(compiler: compiler)
        fun.(transfer(input, backend))
      end)

    receive do
      {:DOWN, ^ref, :process, ^pid, :normal} ->
        :ok

      {:DOWN, ^ref, :process, ^pid, {error, _stack}} when is_exception(error) ->
        {:error, error |> Exception.message() |> String.slice(0, 200)}

      {:DOWN, ^ref, :process, ^pid, reason} ->
        {:error, reason |> inspect() |> String.slice(0, 200)}
    after
      120_000 ->
        Process.exit(pid, :kill)
        {:error, "timed out after 120s"}
    end
  end

  defp algos(data) do
    [
      {"Jump.scan — X̄/S chart kernel", {data.avgs, data.stds, data.counts},
       fn {avgs, stds, counts} ->
         Jump.scan(avgs, stds, counts, baseline: data.baseline)
       end},
      {"Drift.scan — CUSUM via cumulative ops", data.avgs,
       fn avgs -> Drift.scan(avgs, baseline: data.baseline) end},
      {"Shift.chart — EWMA fold + Nx band", data.avgs,
       fn avgs -> Shift.chart(avgs, baseline: data.baseline) end},
      {"Changepoint.detect — prefix-sum split scan", data.steps,
       fn steps -> Changepoint.detect(steps) end},
      {"Shape — PSI + JSD + W1 + mean shift", {data.p_counts, data.q_counts, data.bin_values},
       fn {p, q, v} ->
         Shape.psi(p, q)
         Shape.js_divergence(p, q)
         Shape.moved_by(p, q, v)
         Shape.mean_shift(p, q, v)
       end},
      {"Novelty.fit — covariance + Cholesky", data.history,
       fn history -> Novelty.fit(history) end},
      {"Novelty.score — batched triangular solve", {data.model, data.batch},
       fn {model, batch} -> Novelty.score(model, batch) end}
    ]
  end

  defp build_data(n) do
    :rand.seed(:exsss, {11, 22, 33})
    target = 70.0
    sigma = 2.0
    drift_start = div(n * 3, 5)

    avgs =
      for i <- 0..(n - 1) do
        creep = if i >= drift_start, do: (i - drift_start) * 0.02 * sigma, else: 0.0
        target + creep + :rand.normal() * sigma
      end

    stds = for _ <- 1..n, do: abs(sigma + :rand.normal() * 0.3)
    counts = for _ <- 1..n, do: 50 + :rand.uniform(20)

    # Three regimes for the changepoint scan.
    steps =
      Enum.flat_map([{10.0, 0.4}, {14.0, 0.3}, {9.0, 0.3}], fn {level, share} ->
        for _ <- 1..round(n * share), do: level + :rand.normal()
      end)

    # Two binned distributions on a shared ascending axis (128 bins).
    bins = 128
    bump = fn center, i -> 1000.0 * :math.exp(-((i - center) / 12.0) ** 2) end
    p_counts = for i <- 0..(bins - 1), do: bump.(40, i) + 5.0
    q_counts = for i <- 0..(bins - 1), do: 0.8 * bump.(55, i) + 0.4 * bump.(90, i) + 5.0
    bin_values = for i <- 0..(bins - 1), do: 10.0 * 1.04 ** i

    # Correlated 8-metric history + batch for Novelty.
    m = 8
    rows = max(512, 5 * m)

    correlated = fn ->
      base = :rand.normal()
      for _ <- 1..m, do: 0.6 * base + :rand.normal()
    end

    history = Nx.tensor(for(_ <- 1..rows, do: correlated.()), type: :f64)
    batch = Nx.tensor(for(_ <- 1..rows, do: correlated.()), type: :f64)

    avgs_t = Nx.tensor(avgs, type: :f64)
    stds_t = Nx.tensor(stds, type: :f64)
    counts_t = Nx.tensor(counts, type: :f64)

    %{
      avgs: avgs_t,
      stds: stds_t,
      counts: counts_t,
      baseline: Jump.baseline(avgs_t, stds_t, counts_t),
      steps: Nx.tensor(steps, type: :f64),
      p_counts: Nx.tensor(p_counts, type: :f64),
      q_counts: Nx.tensor(q_counts, type: :f64),
      bin_values: Nx.tensor(bin_values, type: :f64),
      history: history,
      batch: batch,
      model: Novelty.fit(history)
    }
  end

  # Deep-copy any tensors in the input onto the scenario's backend.
  defp transfer(%Nx.Tensor{} = t, backend), do: Nx.backend_copy(t, backend)
  defp transfer(%{} = map, backend), do: Map.new(map, fn {k, v} -> {k, transfer(v, backend)} end)
  defp transfer(tuple, backend) when is_tuple(tuple),
    do: tuple |> Tuple.to_list() |> Enum.map(&transfer(&1, backend)) |> List.to_tuple()

  defp transfer(list, backend) when is_list(list), do: Enum.map(list, &transfer(&1, backend))
  defp transfer(other, _backend), do: other
end

BackendBench.run()
