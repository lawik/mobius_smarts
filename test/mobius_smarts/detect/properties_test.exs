defmodule MobiusSmarts.Detect.PropertiesTest do
  @moduledoc """
  Property-based tests over generated value ranges, tuning parameters,
  and series lengths. Where the conformance tests pin known cases to
  the textbook math, these assert the invariants that must hold for
  *every* input: definitional equivalences, bounds, symmetries, and
  batch/streaming agreement.
  """
  use ExUnit.Case, async: true
  use ExUnitProperties

  alias MobiusSmarts.Detect.{Changepoint, Drift, Jump, Novelty, Shape, Shift, Trend}

  # Detector inputs are window averages of real metrics: finite floats
  # over a generous but bounded range.
  defp value_gen, do: StreamData.float(min: -1.0e3, max: 1.0e3)

  defp series_gen(min_length, max_length) do
    StreamData.list_of(value_gen(), min_length: min_length, max_length: max_length)
  end

  describe "Drift (CUSUM)" do
    property "scan matches Page's recursion for arbitrary series and tuning" do
      check all(
              values <- series_gen(1, 60),
              target <- StreamData.float(min: -100.0, max: 100.0),
              sigma <- StreamData.float(min: 0.01, max: 50.0),
              k <- StreamData.float(min: 0.0, max: 3.0)
            ) do
        result = Drift.scan(values, target: target, sigma: sigma, k: k, h: 5.0)

        reference =
          values
          |> Enum.map(&((&1 - target) / sigma))
          |> Enum.scan({0.0, 0.0}, fn y, {up, down} ->
            {max(0.0, up + y - k), max(0.0, down - y - k)}
          end)

        # The reflection identity is algebraically exact; allow float
        # slop proportional to the magnitudes the cumulative sums reach.
        scale = (Enum.max_by(values, &abs/1) |> abs() |> Kernel.+(abs(target))) / sigma
        tol = 1.0e-9 * max(1.0, scale * length(values))

        got = Enum.zip(Nx.to_flat_list(result.upper), Nx.to_flat_list(result.lower))

        for {{got_up, got_down}, {ref_up, ref_down}} <- Enum.zip(got, reference) do
          assert_in_delta got_up, ref_up, tol
          assert_in_delta got_down, ref_down, tol
        end
      end
    end

    property "bucket levels are never negative and streaming agrees with batch" do
      check all(
              values <- series_gen(1, 50),
              sigma <- StreamData.float(min: 0.1, max: 20.0)
            ) do
        opts = [target: 0.0, sigma: sigma]
        result = Drift.scan(values, opts)

        assert Nx.to_number(Nx.reduce_min(result.upper)) >= 0.0
        assert Nx.to_number(Nx.reduce_min(result.lower)) >= 0.0

        {_statuses, final} =
          Enum.map_reduce(values, Drift.new(opts), fn x, state -> Drift.step(state, x) end)

        scale = (Enum.max_by(values, &abs/1) |> abs()) / sigma
        tol = 1.0e-9 * max(1.0, scale * length(values))

        assert_in_delta final.upper, Nx.to_number(result.upper[-1]), tol
        assert_in_delta final.lower, Nx.to_number(result.lower[-1]), tol
      end
    end
  end

  describe "Shift (EWMA)" do
    property "the impression is a convex combination: bounded by inputs and target" do
      check all(
              values <- series_gen(1, 50),
              target <- StreamData.float(min: -100.0, max: 100.0),
              lambda <- StreamData.float(min: 0.01, max: 1.0)
            ) do
        result = Shift.chart(values, target: target, sigma: 1.0, lambda: lambda)

        lo = min(Enum.min(values), target) - 1.0e-9
        hi = max(Enum.max(values), target) + 1.0e-9

        for z <- Nx.to_flat_list(result.smoothed) do
          assert z >= lo
          assert z <= hi
        end
      end
    end

    property "streaming agrees with batch on the final impression" do
      check all(
              values <- series_gen(1, 50),
              lambda <- StreamData.float(min: 0.01, max: 1.0)
            ) do
        opts = [target: 0.0, sigma: 1.0, lambda: lambda]
        result = Shift.chart(values, opts)

        {_statuses, final} =
          Enum.map_reduce(values, Shift.new(opts), fn x, state -> Shift.step(state, x) end)

        assert_in_delta final.smoothed, Nx.to_number(result.smoothed[-1]), 1.0e-9
      end
    end
  end

  describe "Jump (X-bar/S charts)" do
    property "violation masks are exactly consistent with the returned limits" do
      check all(
              rows <-
                StreamData.list_of(
                  StreamData.tuple(
                    {value_gen(), StreamData.float(min: 0.0, max: 100.0),
                     StreamData.integer(2..200)}
                  ),
                  min_length: 2,
                  max_length: 40
                )
            ) do
        avgs = Enum.map(rows, &elem(&1, 0))
        stds = Enum.map(rows, &elem(&1, 1))
        counts = Enum.map(rows, &elem(&1, 2))

        result = Jump.scan(avgs, stds, counts)

        jump_ucl = Nx.to_flat_list(result.jump_ucl)
        jump_lcl = Nx.to_flat_list(result.jump_lcl)
        jumps = Nx.to_flat_list(result.jumps)

        for {{avg, ucl, lcl}, flag} <- Enum.zip(Enum.zip([avgs, jump_ucl, jump_lcl]), jumps) do
          expected = if avg > ucl or avg < lcl, do: 1, else: 0
          assert flag == expected
        end

        # Pooled sigma is a dof-weighted RMS of the window stds, so it
        # must lie within their range.
        assert result.pooled_sigma >= Enum.min(stds) - 1.0e-9
        assert result.pooled_sigma <= Enum.max(stds) + 1.0e-9
      end
    end
  end

  describe "Shape (distribution distances)" do
    defp counts_pair_gen do
      StreamData.bind(StreamData.integer(2..30), fn len ->
        StreamData.tuple(
          {StreamData.list_of(StreamData.integer(0..1000), length: len),
           StreamData.list_of(StreamData.integer(0..1000), length: len)}
        )
      end)
    end

    # Guarantee each histogram holds at least some mass.
    defp ensure_mass({p, q}),
      do: {List.update_at(p, 0, &(&1 + 1)), List.update_at(q, 0, &(&1 + 1))}

    property "PSI is non-negative and zero on identical histograms" do
      check all(pair <- counts_pair_gen()) do
        {p, q} = ensure_mass(pair)

        assert Shape.psi(p, q) >= 0.0
        assert_in_delta Shape.psi(p, p), 0.0, 1.0e-12
      end
    end

    property "JS divergence is symmetric and bounded by ln 2" do
      check all(pair <- counts_pair_gen()) do
        {p, q} = ensure_mass(pair)

        jsd = Shape.js_divergence(p, q)

        assert jsd >= 0.0
        assert jsd <= :math.log(2.0) + 1.0e-12
        assert_in_delta jsd, Shape.js_divergence(q, p), 1.0e-12
      end
    end

    property "moved_by is symmetric, non-negative, and translation-invariant" do
      check all(
              pair <- counts_pair_gen(),
              start <- StreamData.float(min: -100.0, max: 100.0),
              shift_by <- StreamData.float(min: -50.0, max: 50.0)
            ) do
        {p, q} = ensure_mass(pair)

        # Strictly ascending bin values built from positive increments.
        increments = Enum.map(1..length(p), fn i -> 0.5 + rem(i * 7, 13) end)
        values = Enum.scan(increments, start, &(&2 + &1))

        w = Shape.moved_by(p, q, values)

        assert w >= 0.0
        assert_in_delta w, Shape.moved_by(q, p, values), 1.0e-9 * max(1.0, w)
        assert_in_delta Shape.moved_by(p, p, values), 0.0, 1.0e-12

        # The earth-mover's distance depends on bin spacing, not position.
        shifted = Enum.map(values, &(&1 + shift_by))
        assert_in_delta w, Shape.moved_by(p, q, shifted), 1.0e-9 * max(1.0, w)
      end
    end
  end

  describe "Novelty (Mahalanobis)" do
    property "the history mean scores zero and all scores are non-negative" do
      check all(
              m <- StreamData.integer(2..4),
              rows <-
                StreamData.list_of(
                  StreamData.list_of(StreamData.float(min: -100.0, max: 100.0), length: m),
                  min_length: 8,
                  max_length: 24
                )
            ) do
        model = Novelty.fit(rows, ridge: 1.0e-3)

        assert_in_delta Novelty.score(model, Nx.to_flat_list(model.mean)), 0.0, 1.0e-9

        batch = Novelty.score(model, rows)
        assert Nx.to_number(Nx.reduce_min(batch)) >= 0.0
      end
    end
  end

  describe "Changepoint" do
    property "change points are sorted, respect min_size margins, and yield to a huge penalty" do
      check all(
              values <- series_gen(4, 60),
              min_size <- StreamData.integer(2..6)
            ) do
        result = Changepoint.detect(values, min_size: min_size)
        n = length(values)

        assert result == Enum.sort(result)
        assert Enum.all?(result, &(&1 >= min_size and &1 <= n - min_size))
        assert Changepoint.detect(values, penalty: 1.0e18) == []
      end
    end
  end

  describe "Trend" do
    property "Theil-Sen recovers exact affine trends regardless of slope and offset" do
      check all(
              a <- StreamData.float(min: -1.0e3, max: 1.0e3),
              b <- StreamData.float(min: -100.0, max: 100.0),
              n <- StreamData.integer(3..40)
            ) do
        values = Enum.map(0..(n - 1), &(a + b * &1))

        %{slope: slope, intercept: intercept} = Trend.theil_sen(values)

        assert_in_delta slope, b, 1.0e-9 * max(1.0, abs(b))
        assert_in_delta intercept, a, 1.0e-9 * max(1.0, abs(a)) + 1.0e-6 * abs(b) * n
      end
    end

    property "Mann-Kendall S is antisymmetric under time reversal and bounded" do
      check all(values <- series_gen(3, 40)) do
        forward = Trend.mann_kendall(values)
        backward = Trend.mann_kendall(Enum.reverse(values))
        n = length(values)

        assert forward.s == -backward.s
        assert abs(forward.s) <= div(n * (n - 1), 2)
      end
    end
  end
end
