defmodule MobiusSmarts.ConfigTest do
  use ExUnit.Case, async: true

  alias MobiusSmarts.Config
  alias MobiusSmarts.Config.Metric

  doctest Config

  @duration_keys [
    :resolution,
    :trend_resolution,
    :interval,
    :sweep_interval,
    :analysis_window,
    :trend_window,
    :false_alarm_every,
    :warn_horizon,
    :critical_horizon,
    :refit_interval
  ]

  # The keys new!/1 refuses to default, merged under tests that probe
  # one specific validation each.
  @required [resolution: {1, :minute}, false_alarm_every: {1, :week}]

  defp new!(opts), do: Config.new!(Keyword.merge(@required, opts))

  describe "new!/1 accepts" do
    test "a minimal config: the required keys stated, every default valid" do
      config = Config.new!(resolution: {1, :minute}, false_alarm_every: {1, :week})

      assert config.resolution == {1, :minute}
      assert config.false_alarm_every == {1, :week}
      # :interval is scheduling-only and defaults to :resolution.
      assert config.interval == {1, :minute}
      assert config.watch == []
    end

    test "a fully specified valid config" do
      config =
        Config.new!(
          mobius_instance: :mobius,
          source: MobiusSmarts.Source,
          resolution: {1, :minute},
          trend_resolution: {1, :hour},
          interval: {30, :second},
          sweep_interval: {2, :hour},
          analysis_window: {4, :hour},
          trend_window: {48, :hour},
          false_alarm_every: {2, :week},
          warn_horizon: {14, :day},
          critical_horizon: {2, :day},
          min_baseline_windows: 30,
          refit_interval: {12, :hour},
          clear_after: 5,
          gap_factor: 2.5,
          cusum_k: 0.25,
          ewma_lambda: 0.1,
          novelty: true,
          watch: [
            "vm.memory.used_percent",
            [metric: "disk.used_percent", ceiling: 95.0],
            [metric: "battery.percent", floor: 5.0],
            [metric: "http.request.duration", histogram: true, tags: %{route: "/api"}]
          ]
        )

      assert [%Metric{name: "vm.memory.used_percent"} | _rest] = config.watch
    end

    test "raw positive-integer milliseconds for every duration" do
      config =
        Config.new!(
          resolution: 60_000,
          trend_resolution: 3_600_000,
          interval: 25,
          sweep_interval: 60_000,
          analysis_window: 7_200_000,
          trend_window: 86_400_000,
          false_alarm_every: 604_800_000,
          warn_horizon: 604_800_000,
          critical_horizon: 86_400_000,
          refit_interval: 86_400_000
        )

      assert Config.ms(config.interval) == 25
      assert Config.ms(config.sweep_interval) == 60_000
    end

    test "the boundary values of the bounded fields" do
      config =
        new!(
          cusum_k: 3.0,
          ewma_lambda: 1.0,
          clear_after: 1,
          min_baseline_windows: 2,
          gap_factor: 1.5
        )

      assert config.cusum_k == 3.0
      assert config.ewma_lambda == 1.0
    end
  end

  describe "seasonality" do
    test "accepts an exact multiple of resolution, rejects misalignment" do
      config = new!(seasonality: {2, :hour})
      assert config.seasonality == {2, :hour}

      assert_raise ArgumentError, ~r/exact multiple of :resolution/, fn ->
        new!(seasonality: {90, :second})
      end

      assert_raise ArgumentError, ~r/at least 2 windows/, fn ->
        new!(seasonality: {1, :minute})
      end
    end
  end

  describe "new!/1 requires" do
    # A missing :resolution is covered by the doctest.

    test "a stated :false_alarm_every" do
      error = assert_raise(ArgumentError, fn -> Config.new!(resolution: {1, :minute}) end)
      assert error.message =~ ":false_alarm_every"
    end

    test "a :trend_resolution once a watch entry has a ceiling or floor" do
      error =
        assert_raise(ArgumentError, fn ->
          Config.new!(
            watch: [[metric: "disk.pct", ceiling: 95.0]],
            resolution: {1, :minute},
            false_alarm_every: {1, :week}
          )
        end)

      assert error.message =~ ":trend_resolution"
    end
  end

  describe "new!/1 rejects" do
    test "unknown keys" do
      assert_raise ArgumentError, "unknown MobiusSmarts config keys: [:wat]", fn ->
        Config.new!(wat: 1)
      end
    end

    test "zero, negative, and non-integer duration counts, naming the key" do
      for key <- @duration_keys, bad <- [{0, :minute}, {-1, :hour}, {1.5, :day}, 0, -50] do
        error = assert_raise(ArgumentError, fn -> new!([{key, bad}]) end)
        assert error.message =~ inspect(key)
        assert error.message =~ inspect(bad)
      end
    end

    test "unknown duration units" do
      error = assert_raise(ArgumentError, fn -> new!(interval: {1, :fortnight}) end)
      assert error.message =~ ":interval"
      assert error.message =~ ":fortnight"
    end

    test "false_alarm_every: {0, :week}" do
      error = assert_raise(ArgumentError, fn -> new!(false_alarm_every: {0, :week}) end)
      assert error.message =~ ":false_alarm_every"
    end

    test "a false-alarm budget shorter than one resolution window" do
      error =
        assert_raise(ArgumentError, fn ->
          new!(false_alarm_every: {1, :second}, resolution: {1, :minute})
        end)

      assert error.message =~ ":false_alarm_every"
      assert error.message =~ ":resolution"
    end

    test "a trend_resolution wider than the trend_window" do
      error =
        assert_raise(ArgumentError, fn ->
          new!(trend_resolution: {2, :day}, trend_window: {24, :hour})
        end)

      assert error.message =~ ":trend_resolution"
      assert error.message =~ ":trend_window"
    end

    test "an analysis window too small to ever finish learning" do
      error =
        assert_raise(ArgumentError, fn ->
          new!(analysis_window: {30, :minute}, min_baseline_windows: 60)
        end)

      assert error.message =~ ":analysis_window"
      assert error.message =~ ":min_baseline_windows"
    end

    test "cusum_k: 0" do
      error = assert_raise(ArgumentError, fn -> new!(cusum_k: 0) end)
      assert error.message =~ ":cusum_k"
    end

    test "a negative cusum_k" do
      error = assert_raise(ArgumentError, fn -> new!(cusum_k: -0.5) end)
      assert error.message =~ ":cusum_k"
    end

    test "cusum_k beyond the exp-overflow bound" do
      error = assert_raise(ArgumentError, fn -> new!(cusum_k: 4) end)
      assert error.message =~ ":cusum_k"
      assert error.message =~ "(0, 3]"
    end

    test "ewma_lambda: 0" do
      error = assert_raise(ArgumentError, fn -> new!(ewma_lambda: 0) end)
      assert error.message =~ ":ewma_lambda"
      assert error.message =~ "(0, 1]"
    end

    test "an ewma_lambda above 1" do
      error = assert_raise(ArgumentError, fn -> new!(ewma_lambda: 1.5) end)
      assert error.message =~ ":ewma_lambda"
    end

    test "clear_after: 0" do
      error = assert_raise(ArgumentError, fn -> new!(clear_after: 0) end)
      assert error.message =~ ":clear_after"
    end

    test "a gap_factor at or below 1" do
      for bad <- [1, 1.0, 0.5, -3.0] do
        error = assert_raise(ArgumentError, fn -> new!(gap_factor: bad) end)
        assert error.message =~ ":gap_factor"
        assert error.message =~ inspect(bad)
      end
    end

    test "min_baseline_windows below 2" do
      error = assert_raise(ArgumentError, fn -> new!(min_baseline_windows: 1) end)
      assert error.message =~ ":min_baseline_windows"
    end

    test "a novelty outside :auto | boolean" do
      error = assert_raise(ArgumentError, fn -> new!(novelty: :sometimes) end)
      assert error.message =~ ":novelty"
    end

    test "a non-atom mobius_instance" do
      error = assert_raise(ArgumentError, fn -> new!(mobius_instance: "mobius") end)
      assert error.message =~ ":mobius_instance"
    end

    test "a non-module source" do
      error = assert_raise(ArgumentError, fn -> new!(source: "Source") end)
      assert error.message =~ ":source"
    end

    test "a non-list watch" do
      error = assert_raise(ArgumentError, fn -> new!(watch: "cpu.temp") end)
      assert error.message =~ ":watch"
    end

    test "a watch entry without a :metric name" do
      assert_raise ArgumentError, ~r/watch entry needs a :metric name/, fn ->
        new!(watch: [[ceiling: 95.0]])
      end
    end

    test "a non-numeric ceiling" do
      error =
        assert_raise(ArgumentError, fn ->
          new!(watch: [[metric: "disk.pct", ceiling: "95"]], trend_resolution: {1, :hour})
        end)

      assert error.message =~ ":ceiling"
      assert error.message =~ "disk.pct"
    end

    test "a non-numeric floor" do
      error =
        assert_raise(ArgumentError, fn ->
          new!(watch: [[metric: "battery.pct", floor: "5"]], trend_resolution: {1, :hour})
        end)

      assert error.message =~ ":floor"
      assert error.message =~ "battery.pct"
    end

    test "a floor at or above the ceiling" do
      error =
        assert_raise(ArgumentError, fn ->
          new!(
            watch: [[metric: "tank.level", floor: 90.0, ceiling: 10.0]],
            trend_resolution: {1, :hour}
          )
        end)

      assert error.message =~ ":floor"
      assert error.message =~ ":ceiling"
      assert error.message =~ "tank.level"
    end

    test "non-map tags" do
      error =
        assert_raise(ArgumentError, fn ->
          new!(watch: [[metric: "http.duration", tags: [route: "/api"]]])
        end)

      assert error.message =~ ":tags"
      assert error.message =~ "http.duration"
    end

    test "a non-boolean histogram" do
      error =
        assert_raise(ArgumentError, fn ->
          new!(watch: [[metric: "http.duration", histogram: 1]])
        end)

      assert error.message =~ ":histogram"
      assert error.message =~ "http.duration"
    end

    test "a Metric struct passed directly is validated too" do
      error =
        assert_raise(ArgumentError, fn ->
          new!(watch: [%Metric{name: "m", tags: nil}])
        end)

      assert error.message =~ ":tags"

      assert_raise ArgumentError, fn ->
        new!(watch: [%Metric{name: :not_a_string}])
      end
    end
  end
end
