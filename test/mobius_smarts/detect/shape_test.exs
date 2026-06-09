defmodule MobiusSmarts.Detect.ShapeTest do
  use ExUnit.Case, async: true

  alias Mobius.DDSketch
  alias MobiusSmarts.Detect.Shape

  doctest Shape

  describe "PSI conformance" do
    test "hand-computed example" do
      # p = [0.5, 0.5], q = [0.25, 0.75]:
      # (0.25-0.5)·ln(0.25/0.5) + (0.75-0.5)·ln(0.75/0.5)
      expected = -0.25 * :math.log(0.5) + 0.25 * :math.log(1.5)

      assert_in_delta Shape.psi([50, 50], [25, 75]), expected, 1.0e-9
    end

    test "symmetric in its standard form" do
      a = [10, 30, 60]
      b = [25, 40, 35]
      assert_in_delta Shape.psi(a, b), Shape.psi(b, a), 1.0e-9
    end

    test "small drift lands in the conventional watch band" do
      psi = Shape.psi([100, 200, 400, 200, 100], [80, 190, 400, 220, 110])
      assert psi > 0.0
      assert psi < 0.1
    end
  end

  describe "Jensen-Shannon conformance" do
    test "hand-computed example" do
      # p = [1, 0], q = [0.5, 0.5], m = [0.75, 0.25]:
      kl_pm = 1.0 * :math.log(1.0 / 0.75)
      kl_qm = 0.5 * :math.log(0.5 / 0.75) + 0.5 * :math.log(0.5 / 0.25)
      expected = 0.5 * kl_pm + 0.5 * kl_qm

      assert_in_delta Shape.js_divergence([10, 0], [5, 5]), expected, 1.0e-9
    end

    test "symmetric and bounded by ln 2" do
      a = [5, 10, 85]
      b = [60, 30, 10]

      jsd = Shape.js_divergence(a, b)
      assert_in_delta jsd, Shape.js_divergence(b, a), 1.0e-12
      assert jsd >= 0.0
      assert jsd <= :math.log(2.0) + 1.0e-12
    end
  end

  describe "Wasserstein conformance" do
    test "uniform unit shift moves distance one" do
      w = Shape.moved_by([1, 1, 1, 0], [0, 1, 1, 1], [0.0, 1.0, 2.0, 3.0])
      assert_in_delta w, 1.0, 1.0e-9
    end

    test "scales with how far mass moves, unlike PSI and JS" do
      values = [0.0, 1.0, 2.0, 3.0, 4.0]
      near = Shape.moved_by([1, 0, 0, 0, 0], [0, 1, 0, 0, 0], values)
      far = Shape.moved_by([1, 0, 0, 0, 0], [0, 0, 0, 0, 1], values)

      assert_in_delta near, 1.0, 1.0e-9
      assert_in_delta far, 4.0, 1.0e-9

      # The bin-position-blind scores cannot tell these apart.
      assert_in_delta Shape.js_divergence([1, 0, 0, 0, 0], [0, 1, 0, 0, 0]),
                      Shape.js_divergence([1, 0, 0, 0, 0], [0, 0, 0, 0, 1]),
                      1.0e-12
    end
  end

  describe "from_sketches/2" do
    test "aligns two sketches and recovers a known shift" do
      baseline = insert_all(DDSketch.new(relative_accuracy: 0.01), List.duplicate(10.0, 1000))
      current = insert_all(DDSketch.new(relative_accuracy: 0.01), List.duplicate(13.0, 1000))

      %{baseline: p, current: q, values: v} = Shape.from_sketches(baseline, current)

      assert Nx.size(p) == Nx.size(q)
      assert Nx.size(p) == Nx.size(v)

      # 1% relative accuracy bounds the bin-value error.
      w = Shape.moved_by(p, q, v)
      assert_in_delta w, 3.0, 0.3
    end

    test "handles zero and negative values via their dedicated bins" do
      baseline = insert_all(DDSketch.new(), [0.0, 0.0, -5.0, 5.0])
      current = insert_all(DDSketch.new(), [0.0, -5.0, -5.0, 5.0])

      %{values: v} = Shape.from_sketches(baseline, current)
      values = Nx.to_flat_list(v)

      assert Enum.any?(values, &(&1 == 0.0))
      assert Enum.any?(values, &(&1 < 0.0))
      assert values == Enum.sort(values)
    end

    test "identical sketches score zero drift on every distance" do
      sketch = insert_all(DDSketch.new(), Enum.map(1..500, &(&1 / 10)))

      %{baseline: p, current: q, values: v} = Shape.from_sketches(sketch, sketch)

      assert_in_delta Shape.psi(p, q), 0.0, 1.0e-12
      assert_in_delta Shape.js_divergence(p, q), 0.0, 1.0e-12
      assert_in_delta Shape.moved_by(p, q, v), 0.0, 1.0e-12
    end

    test "rejects sketches with different relative accuracy" do
      a = DDSketch.new(relative_accuracy: 0.01)
      b = DDSketch.new(relative_accuracy: 0.02)

      assert_raise ArgumentError, ~r/relative_accuracy/, fn ->
        Shape.from_sketches(a, b)
      end
    end
  end

  defp insert_all(sketch, values) do
    Enum.reduce(values, sketch, &DDSketch.insert(&2, &1))
  end
end
