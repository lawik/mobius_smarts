defmodule MobiusSmarts.Detect.Outlier do
  @moduledoc """
  Detects windows that are weird in ways nobody programmed a rule for,
  scored against what the *fleet's* training data said normal looks
  like.

  Implements: Isolation Forest (Liu, Ting & Zhou, 2008), inference
  only.

  Outliers are *few and different*, so random splits isolate them in
  very few cuts: a forest of trees grown by picking a random feature
  and a random threshold corners an anomaly almost immediately — it
  sits alone near the root — while a normal point, buried in the
  crowd, needs many cuts before it ends up in a leaf of its own. The
  score is simply *how quickly did the forest's random cuts corner
  this point*: a short average path means anomalous, a long average
  path means ordinary.

  This is the only detector in the stack that carries *fleet-learned*
  normality down to the device. Its sibling
  `MobiusSmarts.Detect.Novelty` (Mahalanobis) learns the device's
  *own* linear habits on-device and needs no infrastructure; `Outlier`
  carries the fleet's learned *nonlinear* normality down to the device
  and needs a model trained somewhere with the whole fleet's feature
  vectors in front of it. Both are multivariate; reach for `Novelty`
  when each device is its own normal, and `Outlier` when "normal" is a
  fleet-wide notion (and the fleet has the data to learn its odd,
  curved shape that a single covariance can't).

  **This module is inference only.** Training happens fleet-side — for
  example a `sklearn.ensemble.IsolationForest` fitted over per-device
  feature vectors pulled from ClickHouse — and the fitted forest is
  exported to the JSON shape below, shipped to the device, loaded with
  `load!/1`, and used to `score/2` the device's own window vectors.

  **Honest caveats:** the score only means something relative to the
  training distribution baked into the model — feed it features built
  the same way training built them, in the same order and units. A
  model goes stale when the fleet's notion of normal moves; re-train
  and re-ship. Like every tree method it is axis-aligned and scale-free
  per feature, so it shrugs off monotone rescaling but is blind to a
  rotated correlation that `Novelty` would catch — run both.

  ## The score

  `s(x) = 2 ^ (-E[h(x)] / c(psi))` where `E[h(x)]` is the mean path
  length of `x` over the trees and `c(psi)` normalizes by the average
  path length of an unsuccessful BST search over `psi` points (the
  training subsample size). A path that ends in a leaf still holding
  `n` training samples gets `depth + c(n)` charged to it — the `c(n)`
  estimates the cuts that *would* have been needed to keep splitting
  that crowd. With

      c(n) = 2 * (ln(n - 1) + 0.5772156649) - 2 * (n - 1) / n   for n > 2
      c(2) = 1.0
      c(n) = 0.0                                                 for n <= 1

  Scores land in `(0, 1]`. The conventional reading: `~0.5` is
  ordinary, `> ~0.6`–`0.7` is anomalous, close to `1.0` is a near-certain
  outlier, and uniformly low scores across all points mean the model
  saw no clear anomalies.

  ## Model JSON format

  Array-of-columns per tree; node `0` is the root; `feature == -1`
  marks a leaf; `size[i]` is how many training samples reached node
  `i`. At an internal node go **left** when `x[feature] <= threshold`
  (the sklearn convention).

      {
        "psi": 256,
        "n_features": 6,
        "trees": [
          {"feature":   [0,   2,   -1,  -1],
           "threshold": [0.5, 1.2, 0.0, 0.0],
           "left":      [1,   3,   -1,  -1],
           "right":     [2,   4,   -1,  -1],
           "size":      [256, 120, 80,  40]}
        ]
      }

  This library takes no JSON dependency. Decode the bytes at the call
  site and hand the resulting maps/lists to `load!/1`:

      json |> :json.decode() |> MobiusSmarts.Detect.Outlier.load!()   # OTP 27+
      json |> Jason.decode!() |> MobiusSmarts.Detect.Outlier.load!()  # or Jason

  ### Exporting a fitted scikit-learn model

  ```python
  import json

  def export_iforest(est):
      trees = []
      for e in est.estimators_:
          t = e.tree_
          trees.append({
              "feature":   t.feature.tolist(),         # -1 at leaves
              "threshold": t.threshold.tolist(),
              "left":      t.children_left.tolist(),    # -1 at leaves
              "right":     t.children_right.tolist(),   # -1 at leaves
              "size":      t.n_node_samples.tolist(),
          })
      return {
          "psi": int(est.max_samples_),
          "n_features": int(est.n_features_in_),
          "trees": trees,
      }

  with open("iforest.json", "w") as f:
      json.dump(export_iforest(fitted), f)
  ```
  """

  @euler_gamma 0.5772156649

  @typedoc """
  A loaded forest. `trees` are pre-converted to tuple-of-tuples for
  O(1) node access during traversal; `psi` is the training subsample
  size that sets the path-length normalizer; `n_features` is the
  expected vector width.
  """
  @type model() :: %{
          trees: [tree()],
          psi: pos_integer(),
          n_features: pos_integer()
        }

  @typep tree() :: %{
           feature: tuple(),
           threshold: tuple(),
           left: tuple(),
           right: tuple(),
           size: tuple()
         }

  @doc """
  Validate and load a decoded model (maps/lists, as produced by a JSON
  decoder) into the fast `t:model/0` form.

  Raises `ArgumentError` with a pointed message on anything malformed:
  missing keys, ragged columns within a tree, a non-positive `psi`, or
  a feature index that reaches past `n_features`.

  ## Examples

      iex> alias MobiusSmarts.Detect.Outlier
      iex> decoded = %{
      ...>   "psi" => 10,
      ...>   "n_features" => 1,
      ...>   "trees" => [
      ...>     %{"feature" => [0, -1, -1], "threshold" => [10.0, 0.0, 0.0],
      ...>       "left" => [1, -1, -1], "right" => [2, -1, -1], "size" => [10, 1, 9]}
      ...>   ]
      ...> }
      iex> model = Outlier.load!(decoded)
      iex> model.n_features
      1
  """
  @spec load!(map()) :: model()
  def load!(%{} = decoded) do
    psi = fetch!(decoded, "psi")
    n_features = fetch!(decoded, "n_features")
    trees = fetch!(decoded, "trees")

    unless is_integer(psi) and psi > 0 do
      raise ArgumentError, "psi must be a positive integer, got: #{inspect(psi)}"
    end

    unless is_integer(n_features) and n_features > 0 do
      raise ArgumentError,
            "n_features must be a positive integer, got: #{inspect(n_features)}"
    end

    unless is_list(trees) and trees != [] do
      raise ArgumentError, "trees must be a non-empty list, got: #{inspect(trees)}"
    end

    %{
      psi: psi,
      n_features: n_features,
      trees: Enum.with_index(trees, fn tree, i -> load_tree!(tree, i, n_features) end)
    }
  end

  def load!(other) do
    raise ArgumentError,
          ~s(expected a decoded model map with "psi", "n_features", "trees", ) <>
            "got: #{inspect(other)}"
  end

  @doc """
  Anomaly score in `(0, 1]` for one feature vector, or a list of scores
  for a batch of vectors.

  A single vector may be a list of numbers or a 1D `Nx.Tensor` (it is
  flattened with `Nx.to_flat_list/1`); a batch is a list of such
  vectors and returns a list of scores in the same order. Higher means
  more anomalous; see the moduledoc for the conventional reading.

  ## Examples

      iex> alias MobiusSmarts.Detect.Outlier
      iex> decoded = %{
      ...>   "psi" => 10,
      ...>   "n_features" => 1,
      ...>   "trees" => [
      ...>     %{"feature" => [0, -1, -1], "threshold" => [10.0, 0.0, 0.0],
      ...>       "left" => [1, -1, -1], "right" => [2, -1, -1], "size" => [10, 9, 1]}
      ...>   ]
      ...> }
      iex> model = Outlier.load!(decoded)
      iex> # x[0] = 100 > 10 -> right leaf, size 1, depth 1, h = 1 + c(1) = 1.
      iex> # c(psi=10) = 2*(ln 9 + gamma) - 2*9/10 = 3.748880...
      iex> # s = 2 ^ (-1 / 3.748880...)
      iex> Float.round(Outlier.score(model, [100.0]), 6)
      0.831192
  """
  @spec score(model(), [number()] | Nx.Tensor.t()) :: float()
  @spec score(model(), [[number()]]) :: [float()]
  def score(%{} = model, [first | _] = batch) when is_list(first) do
    Enum.map(batch, &score(model, &1))
  end

  def score(%{} = model, vector) do
    x = to_vector(vector)
    norm = c(model.psi)

    mean_path =
      model.trees
      |> Enum.reduce(0.0, fn tree, acc -> acc + path_length(tree, x) end)
      |> Kernel./(length(model.trees))

    :math.pow(2.0, -mean_path / norm)
  end

  # --- traversal -----------------------------------------------------

  # Walk from the root, charging depth at each cut, then add c(leaf_size)
  # for the cuts that would have separated the crowd still in the leaf.
  defp path_length(tree, x) do
    {leaf, depth} = walk(tree, x, 0, 0)
    depth + c(elem(tree.size, leaf))
  end

  defp walk(tree, x, node, depth) do
    case elem(tree.feature, node) do
      -1 ->
        {node, depth}

      feature ->
        next =
          if elem(x, feature) <= elem(tree.threshold, node),
            do: elem(tree.left, node),
            else: elem(tree.right, node)

        walk(tree, x, next, depth + 1)
    end
  end

  # --- path-length normalizer c(n) -----------------------------------

  @doc false
  # Average path length of an unsuccessful search in a BST over n points
  # — the harmonic-number estimate, with the small-n edge cases pinned.
  @spec c(integer()) :: float()
  def c(n) when n > 2 do
    2.0 * (:math.log(n - 1) + @euler_gamma) - 2.0 * (n - 1) / n
  end

  def c(2), do: 1.0
  def c(_n), do: 0.0

  # --- loading helpers -----------------------------------------------

  defp fetch!(map, key) do
    case Map.fetch(map, key) do
      {:ok, value} -> value
      :error -> raise ArgumentError, "model is missing required key #{inspect(key)}"
    end
  end

  defp load_tree!(%{} = tree, index, n_features) do
    cols =
      Map.new(~w(feature threshold left right size), fn key ->
        {key, column!(tree, key, index)}
      end)

    n = length(cols["feature"])

    for {key, col} <- cols, length(col) != n do
      raise ArgumentError,
            "tree #{index}: column #{inspect(key)} has #{length(col)} entries, " <>
              "expected #{n} to match \"feature\""
    end

    validate_nodes!(cols, index, n, n_features)

    %{
      feature: List.to_tuple(cols["feature"]),
      threshold: List.to_tuple(Enum.map(cols["threshold"], &(&1 * 1.0))),
      left: List.to_tuple(cols["left"]),
      right: List.to_tuple(cols["right"]),
      size: List.to_tuple(cols["size"])
    }
  end

  defp load_tree!(other, index, _n_features) do
    raise ArgumentError, "tree #{index} must be a map of columns, got: #{inspect(other)}"
  end

  defp column!(tree, key, index) do
    case Map.fetch(tree, key) do
      {:ok, list} when is_list(list) and list != [] ->
        list

      {:ok, other} ->
        raise ArgumentError,
              "tree #{index}: column #{inspect(key)} must be a non-empty list, " <>
                "got: #{inspect(other)}"

      :error ->
        raise ArgumentError, "tree #{index} is missing column #{inspect(key)}"
    end
  end

  defp validate_nodes!(cols, index, n, n_features) do
    cols["feature"]
    |> Enum.zip([cols["left"], cols["right"], cols["size"]] |> Enum.zip())
    |> Enum.with_index()
    |> Enum.each(fn {{feature, {left, right, size}}, node} ->
      validate_node!(index, node, n, n_features, feature, left, right, size)
    end)
  end

  defp validate_node!(index, node, n, n_features, feature, left, right, size) do
    unless is_integer(size) and size >= 0 do
      raise ArgumentError,
            "tree #{index}, node #{node}: size must be a non-negative integer, " <>
              "got: #{inspect(size)}"
    end

    if feature != -1 do
      validate_internal!(index, node, n, n_features, feature, left, right)
    end

    :ok
  end

  defp validate_internal!(index, node, n, n_features, feature, left, right) do
    cond do
      not is_integer(feature) or feature < 0 or feature >= n_features ->
        raise ArgumentError,
              "tree #{index}, node #{node}: feature index #{inspect(feature)} out of " <>
                "range for n_features=#{n_features} (leaves use -1)"

      child_out_of_range?(left, n) or child_out_of_range?(right, n) ->
        raise ArgumentError,
              "tree #{index}, node #{node}: child index out of range " <>
                "(left=#{inspect(left)}, right=#{inspect(right)}, nodes=#{n})"

      true ->
        :ok
    end
  end

  defp child_out_of_range?(child, n), do: not is_integer(child) or child < 0 or child >= n

  defp to_vector(%Nx.Tensor{} = t), do: List.to_tuple(Nx.to_flat_list(t))
  defp to_vector(list) when is_list(list), do: List.to_tuple(list)
end
