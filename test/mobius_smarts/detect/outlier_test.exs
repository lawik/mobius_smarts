defmodule MobiusSmarts.Detect.OutlierTest do
  use ExUnit.Case, async: true
  use ExUnitProperties

  alias MobiusSmarts.Detect.Outlier

  doctest Outlier

  @gamma 0.5772156649

  # c(n) written out longhand, independent of the implementation, so the
  # conformance assertions pin the theory rather than the code.
  defp c_ref(n) when n > 2, do: 2.0 * (:math.log(n - 1) + @gamma) - 2.0 * (n - 1) / n
  defp c_ref(2), do: 1.0
  defp c_ref(_), do: 0.0

  # A single tree: root splits x[0] <= 10. Left leaf holds 9 training
  # samples, right leaf holds 1, both at depth 1.
  defp one_split_model(psi) do
    Outlier.load!(%{
      "psi" => psi,
      "n_features" => 1,
      "trees" => [
        %{
          "feature" => [0, -1, -1],
          "threshold" => [10.0, 0.0, 0.0],
          "left" => [1, -1, -1],
          "right" => [2, -1, -1],
          "size" => [10, 9, 1]
        }
      ]
    })
  end

  describe "conformance to theory" do
    test "single split: scores match the closed-form 2^(-h/c(psi))" do
      psi = 10
      model = one_split_model(psi)
      cpsi = c_ref(psi)

      # x[0] = 100 > 10 -> right leaf, size 1, depth 1.
      # h = 1 + c(1) = 1 + 0 = 1.
      assert_in_delta Outlier.score(model, [100.0]),
                      :math.pow(2.0, -1.0 / cpsi),
                      1.0e-9

      # x[0] = 5 <= 10 -> left leaf, size 9, depth 1.
      # h = 1 + c(9).
      h_in = 1.0 + c_ref(9)

      assert_in_delta Outlier.score(model, [5.0]),
                      :math.pow(2.0, -h_in / cpsi),
                      1.0e-9
    end

    test "the anomaly scores higher than the inlier" do
      model = one_split_model(10)
      assert Outlier.score(model, [100.0]) > Outlier.score(model, [5.0])
    end

    test "c(n) edge cases via leaf sizes: size 1 adds 0, size 2 adds 1" do
      psi = 10
      cpsi = c_ref(psi)

      # Right leaf size 1: h = depth(1) + c(1)=0 -> 1.
      size1 = one_split_model(psi)
      assert_in_delta Outlier.score(size1, [100.0]), :math.pow(2.0, -1.0 / cpsi), 1.0e-9

      # Right leaf size 2: h = depth(1) + c(2)=1.0 -> 2.
      size2 =
        Outlier.load!(%{
          "psi" => psi,
          "n_features" => 1,
          "trees" => [
            %{
              "feature" => [0, -1, -1],
              "threshold" => [10.0, 0.0, 0.0],
              "left" => [1, -1, -1],
              "right" => [2, -1, -1],
              "size" => [12, 10, 2]
            }
          ]
        })

      assert_in_delta Outlier.score(size2, [100.0]), :math.pow(2.0, -2.0 / cpsi), 1.0e-9
    end

    test "two-tree forest: mean path length is hand-computable" do
      psi = 10
      cpsi = c_ref(psi)

      # Tree A: x[0] <= 10 -> left size 9, right size 1.
      # Tree B: x[1] <= 5  -> left size 1, right size 9.
      model =
        Outlier.load!(%{
          "psi" => psi,
          "n_features" => 2,
          "trees" => [
            %{
              "feature" => [0, -1, -1],
              "threshold" => [10.0, 0.0, 0.0],
              "left" => [1, -1, -1],
              "right" => [2, -1, -1],
              "size" => [10, 9, 1]
            },
            %{
              "feature" => [1, -1, -1],
              "threshold" => [5.0, 0.0, 0.0],
              "left" => [1, -1, -1],
              "right" => [2, -1, -1],
              "size" => [10, 1, 9]
            }
          ]
        })

      # Point [100, 100]:
      #   Tree A -> right, size 1, h = 1 + c(1) = 1.
      #   Tree B -> right, size 9, h = 1 + c(9).
      mean = (1.0 + (1.0 + c_ref(9))) / 2.0

      assert_in_delta Outlier.score(model, [100.0, 100.0]),
                      :math.pow(2.0, -mean / cpsi),
                      1.0e-9
    end
  end

  describe "behavior on a synthetic forest" do
    # "Train" a trivial forest by recursively splitting a generated
    # normal cluster with random axis-aligned cuts, depth-limited at
    # ceil(log2(psi)). Each tree is grown over its own bootstrap-ish
    # subsample of the cluster. Deterministic via a seeded :rand state.
    defp synthetic_forest(n_trees, psi, n_features) do
      :rand.seed(:exsss, {17, 23, 42})
      cluster = for _ <- 1..psi, do: for(_ <- 1..n_features, do: :rand.normal())
      max_depth = ceil(:math.log2(psi))

      trees = for _ <- 1..n_trees, do: build_tree(cluster, n_features, max_depth)

      Outlier.load!(%{"psi" => psi, "n_features" => n_features, "trees" => trees})
    end

    # Returns a column-oriented tree map. Builds the node list by a
    # depth-first grow, threading the next free node index.
    defp build_tree(points, n_features, max_depth) do
      {nodes, _next} = grow(points, n_features, 0, max_depth, 0)

      %{
        "feature" => Enum.map(nodes, & &1.feature),
        "threshold" => Enum.map(nodes, & &1.threshold),
        "left" => Enum.map(nodes, & &1.left),
        "right" => Enum.map(nodes, & &1.right),
        "size" => Enum.map(nodes, & &1.size)
      }
    end

    defp grow(points, _n_features, depth, max_depth, id)
         when depth >= max_depth or length(points) <= 1 do
      {[%{feature: -1, threshold: 0.0, left: -1, right: -1, size: length(points)}], id + 1}
    end

    defp grow(points, n_features, depth, max_depth, id) do
      feature = :rand.uniform(n_features) - 1
      vals = Enum.map(points, &Enum.at(&1, feature))
      lo = Enum.min(vals)
      hi = Enum.max(vals)

      if hi - lo < 1.0e-9 do
        {[%{feature: -1, threshold: 0.0, left: -1, right: -1, size: length(points)}], id + 1}
      else
        threshold = lo + :rand.uniform() * (hi - lo)
        {left_pts, right_pts} = Enum.split_with(points, &(Enum.at(&1, feature) <= threshold))

        left_id = id + 1
        {left_nodes, after_left} = grow(left_pts, n_features, depth + 1, max_depth, left_id)
        {right_nodes, after_right} = grow(right_pts, n_features, depth + 1, max_depth, after_left)

        root = %{
          feature: feature,
          threshold: threshold,
          left: left_id,
          right: after_left,
          size: length(points)
        }

        {[root | left_nodes ++ right_nodes], after_right}
      end
    end

    test "a far-outside point scores higher than every in-cluster point" do
      psi = 128
      model = synthetic_forest(100, psi, 3)

      # Reuse the same seed lineage to draw fresh in-cluster points.
      :rand.seed(:exsss, {99, 7, 3})
      in_points = for _ <- 1..50, do: for(_ <- 1..3, do: :rand.normal())
      outlier = [50.0, -50.0, 50.0]

      out_score = Outlier.score(model, outlier)
      in_scores = Outlier.score(model, in_points)

      assert Enum.all?(in_scores, &(out_score > &1))
      # In-cluster points are ordinary: scores hover below the ~0.6 line.
      assert Enum.all?(in_scores, &(&1 < 0.6))
      # ...and the far outlier reads as anomalous.
      assert out_score > 0.6
    end
  end

  describe "load! validation" do
    defp base do
      %{
        "psi" => 10,
        "n_features" => 1,
        "trees" => [
          %{
            "feature" => [0, -1, -1],
            "threshold" => [10.0, 0.0, 0.0],
            "left" => [1, -1, -1],
            "right" => [2, -1, -1],
            "size" => [10, 9, 1]
          }
        ]
      }
    end

    test "round-trips the documented JSON shape" do
      model = Outlier.load!(base())
      assert model.psi == 10
      assert model.n_features == 1
      assert length(model.trees) == 1
    end

    test "missing top-level key raises with the key named" do
      assert_raise ArgumentError, ~r/"trees"/, fn ->
        Outlier.load!(Map.delete(base(), "trees"))
      end
    end

    test "non-positive psi raises" do
      assert_raise ArgumentError, ~r/psi must be a positive integer/, fn ->
        Outlier.load!(%{base() | "psi" => 0})
      end
    end

    test "ragged columns within a tree raise" do
      tree = %{hd(base()["trees"]) | "size" => [10, 9]}

      assert_raise ArgumentError, ~r/column.*entries/, fn ->
        Outlier.load!(%{base() | "trees" => [tree]})
      end
    end

    test "feature index >= n_features raises" do
      tree = %{hd(base()["trees"]) | "feature" => [5, -1, -1]}

      assert_raise ArgumentError, ~r/out of range/, fn ->
        Outlier.load!(%{base() | "trees" => [tree]})
      end
    end

    test "missing tree column raises" do
      tree = Map.delete(hd(base()["trees"]), "threshold")

      assert_raise ArgumentError, ~r/missing column "threshold"/, fn ->
        Outlier.load!(%{base() | "trees" => [tree]})
      end
    end

    test "non-map input raises" do
      assert_raise ArgumentError, ~r/expected a decoded model map/, fn ->
        Outlier.load!([1, 2, 3])
      end
    end
  end

  describe "properties" do
    defp small_model_gen do
      gen all(
            psi <- StreamData.integer(2..512),
            n_features <- StreamData.integer(1..5)
          ) do
        # A two-leaf tree: root split on feature 0, leaves of random
        # training sizes. Enough to exercise scoring invariants.
        Outlier.load!(%{
          "psi" => psi,
          "n_features" => n_features,
          "trees" => [
            %{
              "feature" => [0, -1, -1],
              "threshold" => [0.0, 0.0, 0.0],
              "left" => [1, -1, -1],
              "right" => [2, -1, -1],
              "size" => [psi, max(div(psi, 2), 1), max(psi - div(psi, 2), 1)]
            }
          ]
        })
      end
    end

    defp vector_gen(n_features) do
      StreamData.list_of(StreamData.float(min: -100.0, max: 100.0), length: n_features)
    end

    property "scores always land in (0.0, 1.0]" do
      check all(
              model <- small_model_gen(),
              x <- vector_gen(model.n_features)
            ) do
        s = Outlier.score(model, x)
        assert s > 0.0
        assert s <= 1.0
      end
    end

    property "batch score equals mapping the single-vector score" do
      check all(
              model <- small_model_gen(),
              batch <-
                StreamData.list_of(vector_gen(model.n_features), min_length: 1, max_length: 8)
            ) do
        assert Outlier.score(model, batch) == Enum.map(batch, &Outlier.score(model, &1))
      end
    end

    property "permuting the tree order leaves the score unchanged" do
      check all(x <- vector_gen(2)) do
        trees = [
          %{
            "feature" => [0, -1, -1],
            "threshold" => [1.0, 0.0, 0.0],
            "left" => [1, -1, -1],
            "right" => [2, -1, -1],
            "size" => [10, 7, 3]
          },
          %{
            "feature" => [1, -1, -1],
            "threshold" => [-2.0, 0.0, 0.0],
            "left" => [1, -1, -1],
            "right" => [2, -1, -1],
            "size" => [10, 4, 6]
          },
          %{
            "feature" => [0, -1, -1],
            "threshold" => [5.0, 0.0, 0.0],
            "left" => [1, -1, -1],
            "right" => [2, -1, -1],
            "size" => [10, 9, 1]
          }
        ]

        base = Outlier.load!(%{"psi" => 10, "n_features" => 2, "trees" => trees})

        shuffled =
          Outlier.load!(%{"psi" => 10, "n_features" => 2, "trees" => Enum.reverse(trees)})

        assert_in_delta Outlier.score(base, x), Outlier.score(shuffled, x), 1.0e-12
      end
    end

    property "accepts a 1D tensor and a list interchangeably" do
      check all(x <- vector_gen(3)) do
        model =
          Outlier.load!(%{
            "psi" => 16,
            "n_features" => 3,
            "trees" => [
              %{
                "feature" => [2, -1, -1],
                "threshold" => [0.0, 0.0, 0.0],
                "left" => [1, -1, -1],
                "right" => [2, -1, -1],
                "size" => [16, 10, 6]
              }
            ]
          })

        assert Outlier.score(model, x) == Outlier.score(model, Nx.tensor(x))
      end
    end
  end
end
