defmodule MobiusProcessing.Recipes.Scholar do
  @moduledoc """
  Recipes that need Scholar — the "compare metrics" and "classify
  behavior" tools. Optional dep:

      {:scholar, "~> 0.3"}

  Each function is framed around a question you might ask of one or more
  device metrics. The data names are illustrative.

  ## Are CPU temperature and CPU load actually related?

  `correlation/2` returns the Pearson coefficient between two
  same-length tensors, between -1 and 1.

  ## Putting unlike metrics on a common scale

  `zscore_columns/1` standardises an `{n_samples, n_metrics}` matrix
  per-column. Use when you care about *deviation from typical*.

  ## Estimating "days until disk is full"

  `days_to_threshold/2` is Scholar's `LinearRegression.fit/3` over the
  index axis. Same semantics as `MobiusProcessing.Recipes.CoreNx.disk_days_to_threshold/2`
  but using the Scholar model object so you can keep predicting without
  re-fitting.

  ## Clustering device state

  `cluster_states/3` runs k-means on per-row standardised state vectors
  and returns the cluster label for each row.

  ## Device health score from many BEAM metrics

  `pca_first_component/1` PCAs an `{n, k}` matrix and returns the first
  principal-component coordinate of every row — usually a "general
  activity" axis.

  ## Anomaly detection by nearest-neighbor distance

  `nn_distance/3` indexes a historical state matrix with a KDTree and
  returns the mean distance of a candidate point to its `k` nearest
  historical neighbors.

  ## Was my threshold rule any good?

  `binary_precision_recall/2` reports `{precision, recall}` of a 0/1
  predicted-vs-ground-truth pair.
  """

  alias Scholar.Cluster.KMeans
  alias Scholar.Decomposition.PCA
  alias Scholar.Linear.LinearRegression
  alias Scholar.Metrics.Classification
  alias Scholar.Neighbors.KDTree
  alias Scholar.Preprocessing.StandardScaler

  @doc """
  Pearson correlation between two 1D tensors of equal length.

  ## Examples

      iex> a = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
      iex> b = Nx.tensor([2.0, 4.0, 6.0, 8.0, 10.0])
      iex> MobiusProcessing.Recipes.Scholar.correlation(a, b)
      ...> |> Float.round(2)
      1.0

      iex> a = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
      iex> b = Nx.tensor([5.0, 4.0, 3.0, 2.0, 1.0])
      iex> MobiusProcessing.Recipes.Scholar.correlation(a, b)
      ...> |> Float.round(2)
      -1.0
  """
  @spec correlation(Nx.Tensor.t(), Nx.Tensor.t()) :: float()
  def correlation(a, b) do
    xs = Nx.stack([a, b], axis: 1)
    matrix = Scholar.Stats.correlation_matrix(xs)
    matrix[0][1] |> Nx.to_number()
  end

  @doc """
  Standardises each column of an `{n_samples, n_metrics}` matrix to
  mean 0, stddev 1. Useful when comparing metrics on wildly different
  scales (CPU% vs. MB vs. raw counter).

  ## Examples

      iex> raw = Nx.tensor([
      ...>   [10.0, 1000.0],
      ...>   [20.0, 2000.0],
      ...>   [30.0, 3000.0]
      ...> ])
      iex> z = MobiusProcessing.Recipes.Scholar.zscore_columns(raw)
      iex> z |> Nx.mean(axes: [0]) |> Nx.to_flat_list() |> Enum.map(&Float.round(&1, 4))
      [0.0, 0.0]
  """
  @spec zscore_columns(Nx.Tensor.t()) :: Nx.Tensor.t()
  def zscore_columns(matrix) do
    StandardScaler.fit_transform(matrix, axes: [0])
  end

  @doc """
  Linear projection of "how many days until `values` crosses `threshold`",
  fitting `values ~ a + b·day` with Scholar's `LinearRegression`. Returns
  `:infinity` if the slope is non-positive.

  ## Examples

      iex> disk = Nx.tensor([50.0, 55.0, 60.0, 65.0, 70.0])
      iex> MobiusProcessing.Recipes.Scholar.days_to_threshold(disk, 95.0)
      ...> |> Float.round(1)
      5.0

      iex> declining = Nx.tensor([60.0, 55.0, 50.0, 45.0])
      iex> MobiusProcessing.Recipes.Scholar.days_to_threshold(declining, 95.0)
      :infinity
  """
  @spec days_to_threshold(Nx.Tensor.t(), number()) :: float() | :infinity
  def days_to_threshold(values, threshold) do
    n = Nx.size(values)
    days = Nx.iota({n, 1}, type: :f32)
    model = LinearRegression.fit(days, Nx.as_type(values, :f32))

    [slope] = Nx.to_flat_list(model.coefficients)
    current = Nx.to_number(values[-1])

    if slope > 0 do
      (threshold - current) / slope
    else
      :infinity
    end
  end

  @doc """
  K-means cluster labels for each row of an `{n_samples, n_metrics}`
  matrix, after per-column standardisation.

  Requires a fixed `Nx.Random.key/1` for reproducibility.

  ## Examples

      iex> states = Nx.tensor([
      ...>   [0.1, 0.1, 30.0],
      ...>   [0.1, 0.1, 31.0],
      ...>   [0.9, 0.8, 70.0],
      ...>   [0.9, 0.8, 71.0]
      ...> ])
      iex> key = Nx.Random.key(42)
      iex> labels = MobiusProcessing.Recipes.Scholar.cluster_states(states, 2, key: key)
      iex> # Two distinct clusters of two rows each; same label within a cluster
      iex> labels = Nx.to_flat_list(labels)
      iex> [a, b, c, d] = labels
      iex> a == b and c == d and a != c
      true
  """
  @spec cluster_states(Nx.Tensor.t(), pos_integer(), keyword()) :: Nx.Tensor.t()
  def cluster_states(matrix, num_clusters, opts \\ []) do
    standardised = StandardScaler.fit_transform(matrix, axes: [0])

    fit_opts =
      Keyword.merge([num_clusters: num_clusters], Keyword.take(opts, [:key]))

    model = KMeans.fit(standardised, fit_opts)
    KMeans.predict(model, standardised)
  end

  @doc """
  Returns the first principal-component coordinate of every row of
  `matrix`. The first PC usually carries the "general activity" axis
  when input metrics are correlated.

  ## Examples

      iex> # Three metrics that all rise together — first PC should
      iex> # separate row 0 from row 4 strongly.
      iex> matrix = Nx.tensor([
      ...>   [1.0, 10.0, 100.0],
      ...>   [2.0, 20.0, 200.0],
      ...>   [3.0, 30.0, 300.0],
      ...>   [4.0, 40.0, 400.0],
      ...>   [5.0, 50.0, 500.0]
      ...> ])
      iex> axis = MobiusProcessing.Recipes.Scholar.pca_first_component(matrix)
      iex> Nx.shape(axis)
      {5}
      iex> # Strict monotonic along the activity axis (direction may be either sign)
      iex> [a, b, c, d, e] = Nx.to_flat_list(axis)
      iex> (a < b and b < c and c < d and d < e) or (a > b and b > c and c > d and d > e)
      true
  """
  @spec pca_first_component(Nx.Tensor.t()) :: Nx.Tensor.t()
  def pca_first_component(matrix) do
    standardised = StandardScaler.fit_transform(matrix, axes: [0])
    model = PCA.fit(standardised, num_components: 2)
    transformed = PCA.transform(model, standardised)
    transformed[[.., 0]]
  end

  @doc """
  Mean Euclidean distance from `candidate` to its `k` nearest neighbors
  in `corpus`. A growing value over time is "this device is drifting
  away from how it usually behaves".

  Both `corpus` and `candidate` are `{n, n_features}` tensors.

  ## Examples

      iex> corpus = Nx.tensor([
      ...>   [0.0, 0.0],
      ...>   [0.1, 0.1],
      ...>   [0.0, 0.2],
      ...>   [0.2, 0.0]
      ...> ])
      iex> # Far from anything in the corpus
      iex> candidate = Nx.tensor([[10.0, 10.0]])
      iex> score = MobiusProcessing.Recipes.Scholar.nn_distance(corpus, candidate, 2)
      iex> score |> Nx.to_number() |> Kernel.>(10.0)
      true
  """
  @spec nn_distance(Nx.Tensor.t(), Nx.Tensor.t(), pos_integer()) :: Nx.Tensor.t()
  def nn_distance(corpus, candidate, k) do
    tree = KDTree.fit(corpus, num_neighbors: k)
    {_indices, distances} = KDTree.predict(tree, candidate)
    Nx.mean(distances)
  end

  @doc """
  `{precision, recall}` of a 0/1 prediction against 0/1 ground truth.
  Both tensors must be the same length and dtype.

  ## Examples

      iex> truth = Nx.tensor([0, 1, 1, 0, 1, 0, 1, 1, 0, 0], type: :u32)
      iex> pred  = Nx.tensor([0, 1, 0, 0, 1, 1, 1, 1, 0, 0], type: :u32)
      iex> {p, r} = MobiusProcessing.Recipes.Scholar.binary_precision_recall(truth, pred)
      iex> {Float.round(Nx.to_number(p), 2), Float.round(Nx.to_number(r), 2)}
      {0.8, 0.8}
  """
  @spec binary_precision_recall(Nx.Tensor.t(), Nx.Tensor.t()) :: {Nx.Tensor.t(), Nx.Tensor.t()}
  def binary_precision_recall(truth, pred) do
    {Classification.binary_precision(truth, pred), Classification.binary_recall(truth, pred)}
  end
end
