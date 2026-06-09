defmodule MobiusSmarts.Recipes.DefnKernels do
  @moduledoc """
  Hand-written kernels for shapes that don't have a one-call form in Nx
  but are short enough to write inline.

  Each function is framed around a plausible question for the kinds of
  metrics a Nerves device tends to emit. The data names are illustrative.

  ## Events per hour from a sparse timestamp stream

  `events_per_hour/1` takes a tensor of UNIX-second timestamps and
  returns a `{24}` count tensor — the histogram of events per hour-of-day.

  ## Transitions and time-in-state for a binary signal

  - `transition_count/1` — diff → abs → sum. Counts every 0→1 and 1→0.
  - `seconds_on/2` — sum of the `1`s, multiplied by the sample interval.

  ## Run-length structure of a state signal

  `run_ids/1` labels every sample with the index of the run it belongs
  to, so you can mask out the `0`-runs and tally per-run lengths with
  `run_lengths/2`.
  """

  import Nx.Defn

  @doc """
  Returns a `{24}` tensor of counts. Bucket `h` is the number of input
  timestamps that fell in hour `h` of the day, modulo 24h.

  ## Examples

      iex> ts = Nx.tensor([0, 3600, 18000, 18100, 93600], type: :s64)
      iex> MobiusSmarts.Recipes.DefnKernels.events_per_hour(ts)
      ...> |> Nx.to_flat_list()
      [1, 1, 1, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  """
  defn events_per_hour(timestamps_unix_sec) do
    bucket =
      timestamps_unix_sec
      |> Nx.remainder(86_400)
      |> Nx.quotient(3_600)
      |> Nx.as_type(:s32)

    zeros = Nx.broadcast(0, {24})
    Nx.indexed_add(zeros, Nx.new_axis(bucket, 1), Nx.broadcast(1, Nx.shape(bucket)))
  end

  @doc """
  Number of state transitions (both rising and falling edges) in a 0/1
  series. Divide by 2 to get the number of complete on-off episodes.

  ## Examples

      iex> state = Nx.tensor([0, 0, 1, 1, 0, 1, 1], type: :u8)
      iex> MobiusSmarts.Recipes.DefnKernels.transition_count(state)
      ...> |> Nx.to_number()
      3
  """
  defn transition_count(state_u8) do
    state_u8
    |> Nx.as_type(:s32)
    |> Nx.diff()
    |> Nx.abs()
    |> Nx.sum()
  end

  @doc """
  Total time the state was `1`, in the same unit as `dt_sec`.
  `Nx.mean(state_u8)` gives the duty cycle directly.

  ## Examples

      iex> state = Nx.tensor([1, 1, 0, 0, 1, 1, 1], type: :u8)
      iex> MobiusSmarts.Recipes.DefnKernels.seconds_on(state, 2.0)
      ...> |> Nx.to_number()
      10.0
  """
  defn seconds_on(state_u8, dt_sec) do
    state_u8
    |> Nx.as_type(:f32)
    |> Nx.sum()
    |> Nx.multiply(dt_sec)
  end

  @doc """
  Per-sample tensor where each entry is the index of the run it belongs
  to. Combine with `Nx.equal(state_u8, 1)` to keep only the `1`-runs
  and `run_lengths/2` to get the duration of each.

  ## Examples

      iex> state = Nx.tensor([0, 1, 1, 0, 1, 1, 1, 0], type: :u8)
      iex> MobiusSmarts.Recipes.DefnKernels.run_ids(state)
      ...> |> Nx.to_flat_list()
      [0, 1, 1, 1, 2, 2, 2, 2]
  """
  defn run_ids(state_u8) do
    rising_edge =
      state_u8
      |> Nx.as_type(:s32)
      |> Nx.diff()
      |> Nx.greater(0)
      |> Nx.as_type(:s32)

    edges = Nx.concatenate([Nx.tensor([0], type: :s32), rising_edge])
    Nx.cumulative_sum(edges)
  end

  @doc """
  Returns a `{max_runs + 1}` tensor where bucket `i` is the duration (in
  samples) of run number `i` for runs where `state_u8` was `1`. Bucket
  `0` holds the count of `0`-samples and is usually ignored.

  ## Examples

      iex> state = Nx.tensor([0, 1, 1, 0, 1, 1, 1, 0], type: :u8)
      iex> # Two on-runs: length 2 then length 3. Run id 1 = 2 samples,
      iex> # run id 2 = 3 samples. The `0`-samples land in bucket 0.
      iex> MobiusSmarts.Recipes.DefnKernels.run_lengths(state, 2)
      ...> |> Nx.to_flat_list()
      [3, 2, 3]
  """
  @spec run_lengths(Nx.Tensor.t(), pos_integer()) :: Nx.Tensor.t()
  def run_lengths(state_u8, max_runs) when is_integer(max_runs) and max_runs > 0 do
    ids = run_ids(state_u8)
    actual_runs = ids |> Nx.reduce_max() |> Nx.to_number()

    # Out-of-range indexed_add indices are backend-defined behavior —
    # fail loudly instead of corrupting counts silently.
    if actual_runs > max_runs do
      raise ArgumentError,
            "series contains #{actual_runs} runs but max_runs is #{max_runs}; " <>
              "raise max_runs to cover them"
    end

    masked = Nx.select(Nx.equal(state_u8, 1), ids, 0)
    bincount(masked, max_runs + 1)
  end

  ## ---------------------------------------------------------------------
  ## Internals
  ## ---------------------------------------------------------------------

  defp bincount(indices, n_bins) do
    zeros = Nx.broadcast(0, {n_bins})
    Nx.indexed_add(zeros, Nx.new_axis(indices, 1), Nx.broadcast(1, Nx.shape(indices)))
  end
end
