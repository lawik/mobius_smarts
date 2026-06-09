defmodule MobiusSmarts.Detect.Novelty do
  @moduledoc """
  Detects when a *combination* of metric values is one this device has
  never produced — even though every individual metric is inside its
  band. CPU pinned while network sits idle is alarming precisely
  because the device's own history says those two move together.

  Implements: Mahalanobis distance (1936) against the device's own
  mean and covariance.

  A device's metrics have habits — when load rises, CPU, temperature,
  and network rise together. Plot a fortnight of windows as points in
  one-axis-per-metric space and they don't fill the space; they hug a
  characteristic cloud whose *shape* encodes the habits. The score is
  simply "how far outside the cloud is this new point?" — but measured
  in the cloud's own stretched coordinates: a direction in which the
  device routinely varies counts as cheap distance, while a direction
  the device has never moved in counts as expensive. One unit ≈
  ordinary; the score answers "how many typical variations away from
  its own habits is the device right now?" — which is why a point that
  violates the *correlations* gets a big score even when each
  coordinate alone is unremarkable.

  And because we know how far points wander under normal conditions,
  the alert threshold can be picked from a target false-alarm rate
  instead of guessed: for well-behaved data the squared score is
  approximately chi-square distributed with `n_metrics` degrees of
  freedom, so threshold at `:math.sqrt/1` of the chi-square quantile
  for your false-alarm rate.

  This is the only multivariate, cross-metric detector in the stack —
  every other module watches one metric at a time.

  **Honest caveats:** it learns straight-line habits (linear
  correlations); it needs a healthy history to fit (a few hundred
  windows); and a fitted model goes stale when the device's legitimate
  behavior changes — refit on a schedule, or after an accepted change
  point from `MobiusSmarts.Detect.Changepoint`.

  Cost is `O(n_metrics²)` per score — trivial for the tens of metrics a
  device watches.
  """

  import Nx.Defn

  @type model() :: %{
          mean: Nx.Tensor.t(),
          chol: Nx.Tensor.t(),
          n_metrics: pos_integer()
        }

  @doc """
  Fit a model from history: an `{n_windows, n_metrics}` tensor (or list
  of row lists) of per-window metric averages.

  Options:

  - `:ridge` — *relative* stabilizer, default `1.0e-9`: the covariance
    diagonal gets `ridge * mean(diagonal)` added, so the knob is
    unit-free (an absolute ridge would be meaningless for bytes-scale
    metrics and overwhelming for micro-scale ones). Raise it if you
    feed near-duplicate metrics.

  Requires at least `n_metrics + 1` windows for a full-rank covariance
  and raises `ArgumentError` below that; practically, fit on a few
  hundred healthy windows. The model factors the covariance (Cholesky)
  rather than inverting it — scoring solves triangular systems, which
  stays accurate where an explicit inverse of nearly-collinear metrics
  would not.
  """
  @spec fit(Nx.Tensor.t() | [[number()]], keyword()) :: model()
  def fit(history, opts \\ []) do
    ridge = Keyword.get(opts, :ridge, 1.0e-9)
    history = to_f64(history)
    {n, m} = Nx.shape(history)

    if n < m + 1 do
      raise ArgumentError,
            "history has #{n} windows for #{m} metrics — a full-rank covariance " <>
              "needs at least n_metrics + 1 windows (practically: a few hundred)"
    end

    {mean, chol} = fit_kernel(history, f64(ridge))

    %{mean: mean, chol: chol, n_metrics: m}
  end

  @doc """
  Novelty score of one window vector (`{n_metrics}`) or a batch
  (`{n_windows, n_metrics}`) against the fitted history.

  Returns a float for a single vector, a `{n_windows}` tensor for a
  batch. `0.0` means "at the center of the device's habits"; the units
  are "typical variations".

  ## Examples

      iex> alias MobiusSmarts.Detect.Novelty
      iex> history = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0], [0.5, 0.5]]
      iex> model = Novelty.fit(history)
      iex> Novelty.score(model, [0.5, 0.5])
      0.0
  """
  @spec score(model(), Nx.Tensor.t() | [number()] | [[number()]]) ::
          float() | Nx.Tensor.t()
  def score(model, x) do
    x = to_f64(x)
    centered = Nx.subtract(x, model.mean)

    case Nx.rank(centered) do
      1 ->
        centered
        |> Nx.new_axis(1)
        |> solve_distance(model.chol)
        |> Nx.squeeze()
        |> Nx.to_number()

      2 ->
        centered
        |> Nx.transpose()
        |> solve_distance(model.chol)
    end
  end

  defnp fit_kernel(history, ridge) do
    n = Nx.axis_size(history, 0)
    m = Nx.axis_size(history, 1)

    mean = Nx.mean(history, axes: [0])
    centered = history - mean
    cov = Nx.dot(Nx.transpose(centered), centered) / (n - 1)

    # Unit-free stabilizer: ridge is relative to the covariance scale.
    scaled_ridge = ridge * Nx.max(Nx.mean(Nx.take_diagonal(cov)), 1.0e-30)
    cov = cov + Nx.eye(m, type: :f64) * scaled_ridge

    {mean, Nx.LinAlg.cholesky(cov)}
  end

  # With cov = L·Lᵀ, the distance is ‖L⁻¹ d‖ — one triangular solve per
  # batch, no explicit inverse. `centered_t` is {m, batch}.
  defnp solve_distance(centered_t, chol) do
    chol
    |> Nx.LinAlg.triangular_solve(centered_t, lower: true)
    |> Nx.pow(2)
    |> Nx.sum(axes: [0])
    |> Nx.sqrt()
  end

  defp to_f64(values) when is_list(values), do: Nx.tensor(values, type: :f64)
  defp to_f64(values), do: Nx.as_type(values, :f64)

  defp f64(scalar), do: Nx.tensor(scalar * 1.0, type: :f64)
end
