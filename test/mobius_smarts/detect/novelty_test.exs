defmodule MobiusSmarts.Detect.NoveltyTest do
  use ExUnit.Case, async: true

  alias MobiusSmarts.Detect.Novelty

  doctest Novelty

  describe "conformance to theory" do
    test "hand-computed case with diagonal covariance" do
      # History: mean [0, 0], sample covariance diag(2/3, 2/3)
      # (each column sums squares to 2 over n - 1 = 3 dof).
      history = [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]
      model = Novelty.fit(history)

      # d([1, 0]) = sqrt(1 / (2/3)) = sqrt(1.5)
      assert_in_delta Novelty.score(model, [1.0, 0.0]), :math.sqrt(1.5), 1.0e-6
    end

    test "reduces to scaled Euclidean distance for uncorrelated history" do
      history = [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]
      model = Novelty.fit(history)

      # Equal variance per axis: distances depend only on the radius.
      d1 = Novelty.score(model, [3.0, 4.0])
      d2 = Novelty.score(model, [5.0, 0.0])
      assert_in_delta d1, d2, 1.0e-6
    end

    test "invariant under invertible linear transformation of the space" do
      history = [
        [1.0, 2.0],
        [2.0, 4.5],
        [3.0, 5.5],
        [4.0, 8.2],
        [5.0, 9.7],
        [6.0, 12.1]
      ]

      x = [3.5, 9.0]

      # Transform: [a, b] -> [2a + b, a - 3b]
      transform = fn [a, b] -> [2.0 * a + b, a - 3.0 * b] end

      original = Novelty.fit(history, ridge: 0.0) |> Novelty.score(x)

      transformed =
        history
        |> Enum.map(transform)
        |> Novelty.fit(ridge: 0.0)
        |> Novelty.score(transform.(x))

      assert_in_delta original, transformed, 1.0e-6
    end

    test "distance from the history mean is zero" do
      history = [[10.0, 5.0], [12.0, 6.0], [14.0, 7.0], [16.0, 9.0]]
      model = Novelty.fit(history)

      mean = Nx.to_flat_list(model.mean)
      assert_in_delta Novelty.score(model, mean), 0.0, 1.0e-9
    end
  end

  describe "fit validation and conditioning" do
    test "too few windows for the metric count raises instead of scoring garbage" do
      # 4 windows for 5 metrics: rank-deficient covariance (CRITIQUE §8).
      history = for _ <- 1..4, do: Enum.map(1..5, fn m -> m * 1.0 end)

      assert_raise ArgumentError, ~r/n_metrics \+ 1/, fn ->
        Novelty.fit(history)
      end
    end

    test "ridge is relative to the covariance scale" do
      # Identical correlation structure at wildly different scales must
      # produce (nearly) identical distances under the default ridge —
      # an absolute ridge would crush the small-scale model.
      base = [[1.0, 2.0], [2.0, 4.5], [3.0, 5.5], [4.0, 8.2], [5.0, 9.7], [6.0, 12.1]]
      tiny = Enum.map(base, fn row -> Enum.map(row, &(&1 * 1.0e-6)) end)

      d_base = base |> Novelty.fit() |> Novelty.score([3.5, 9.0])
      d_tiny = tiny |> Novelty.fit() |> Novelty.score([3.5e-6, 9.0e-6])

      # A percent, not machine epsilon: the columns are nearly
      # collinear, so elementwise rounding of the scaled history
      # legitimately perturbs the smallest eigenvalue a little. An
      # absolute ridge would be off by orders of magnitude here.
      assert_in_delta d_base, d_tiny, 0.01 * d_base
    end
  end

  describe "batch scoring" do
    test "returns a tensor of distances for a batch of windows" do
      history = [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]
      model = Novelty.fit(history)

      distances = Novelty.score(model, [[1.0, 0.0], [0.0, 0.0], [2.0, 0.0]])

      assert Nx.shape(distances) == {3}
      [d1, d0, d2] = Nx.to_flat_list(distances)
      assert_in_delta d1, :math.sqrt(1.5), 1.0e-6
      assert_in_delta d0, 0.0, 1.0e-9
      assert_in_delta d2, 2.0 * :math.sqrt(1.5), 1.0e-6
    end
  end
end
