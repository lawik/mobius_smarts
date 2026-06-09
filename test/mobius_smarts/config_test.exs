defmodule MobiusSmarts.ConfigTest do
  use ExUnit.Case, async: true

  alias MobiusSmarts.Config
  alias MobiusSmarts.Config.Metric

  doctest Config

  @duration_keys [
    :interval,
    :sweep_interval,
    :analysis_window,
    :trend_window,
    :false_alarm_budget,
    :warn_horizon,
    :critical_horizon,
    :refit_interval
  ]

  describe "new!/1 accepts" do
    test "an empty config: every default is valid" do
      config = Config.new!([])

      assert config.interval == {1, :minute}
      assert config.false_alarm_budget == {1, :week}
      assert config.watch == []
    end

    test "a fully specified valid config" do
      config =
        Config.new!(
          mobius_instance: :mobius,
          source: MobiusSmarts.Source,
          interval: {30, :second},
          sweep_interval: {2, :hour},
          analysis_window: {4, :hour},
          trend_window: {48, :hour},
          false_alarm_budget: {2, :week},
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
          interval: 25,
          sweep_interval: 60_000,
          analysis_window: 7_200_000,
          trend_window: 86_400_000,
          false_alarm_budget: 604_800_000,
          warn_horizon: 604_800_000,
          critical_horizon: 86_400_000,
          refit_interval: 86_400_000
        )

      assert Config.ms(config.interval) == 25
      assert Config.ms(config.sweep_interval) == 60_000
    end

    test "the boundary values of the bounded fields" do
      config =
        Config.new!(
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

  describe "new!/1 rejects" do
    test "unknown keys" do
      assert_raise ArgumentError, "unknown MobiusSmarts config keys: [:wat]", fn ->
        Config.new!(wat: 1)
      end
    end

    test "zero, negative, and non-integer duration counts, naming the key" do
      for key <- @duration_keys, bad <- [{0, :minute}, {-1, :hour}, {1.5, :day}, 0, -50] do
        error = assert_raise(ArgumentError, fn -> Config.new!([{key, bad}]) end)
        assert error.message =~ inspect(key)
        assert error.message =~ inspect(bad)
      end
    end

    test "unknown duration units" do
      error = assert_raise(ArgumentError, fn -> Config.new!(interval: {1, :fortnight}) end)
      assert error.message =~ ":interval"
      assert error.message =~ ":fortnight"
    end

    test "false_alarm_budget: {0, :week}" do
      error = assert_raise(ArgumentError, fn -> Config.new!(false_alarm_budget: {0, :week}) end)
      assert error.message =~ ":false_alarm_budget"
    end

    test "a false-alarm budget shorter than one tick" do
      error =
        assert_raise(ArgumentError, fn ->
          Config.new!(false_alarm_budget: {1, :second}, interval: {1, :minute})
        end)

      assert error.message =~ ":false_alarm_budget"
      assert error.message =~ ":interval"
    end

    test "cusum_k: 0" do
      error = assert_raise(ArgumentError, fn -> Config.new!(cusum_k: 0) end)
      assert error.message =~ ":cusum_k"
    end

    test "a negative cusum_k" do
      error = assert_raise(ArgumentError, fn -> Config.new!(cusum_k: -0.5) end)
      assert error.message =~ ":cusum_k"
    end

    test "cusum_k beyond the exp-overflow bound" do
      error = assert_raise(ArgumentError, fn -> Config.new!(cusum_k: 4) end)
      assert error.message =~ ":cusum_k"
      assert error.message =~ "(0, 3]"
    end

    test "ewma_lambda: 0" do
      error = assert_raise(ArgumentError, fn -> Config.new!(ewma_lambda: 0) end)
      assert error.message =~ ":ewma_lambda"
      assert error.message =~ "(0, 1]"
    end

    test "an ewma_lambda above 1" do
      error = assert_raise(ArgumentError, fn -> Config.new!(ewma_lambda: 1.5) end)
      assert error.message =~ ":ewma_lambda"
    end

    test "clear_after: 0" do
      error = assert_raise(ArgumentError, fn -> Config.new!(clear_after: 0) end)
      assert error.message =~ ":clear_after"
    end

    test "a gap_factor at or below 1" do
      for bad <- [1, 1.0, 0.5, -3.0] do
        error = assert_raise(ArgumentError, fn -> Config.new!(gap_factor: bad) end)
        assert error.message =~ ":gap_factor"
        assert error.message =~ inspect(bad)
      end
    end

    test "min_baseline_windows below 2" do
      error = assert_raise(ArgumentError, fn -> Config.new!(min_baseline_windows: 1) end)
      assert error.message =~ ":min_baseline_windows"
    end

    test "a novelty outside :auto | boolean" do
      error = assert_raise(ArgumentError, fn -> Config.new!(novelty: :sometimes) end)
      assert error.message =~ ":novelty"
    end

    test "a non-atom mobius_instance" do
      error = assert_raise(ArgumentError, fn -> Config.new!(mobius_instance: "mobius") end)
      assert error.message =~ ":mobius_instance"
    end

    test "a non-module source" do
      error = assert_raise(ArgumentError, fn -> Config.new!(source: "Source") end)
      assert error.message =~ ":source"
    end

    test "a non-list watch" do
      error = assert_raise(ArgumentError, fn -> Config.new!(watch: "cpu.temp") end)
      assert error.message =~ ":watch"
    end

    test "a watch entry without a :metric name" do
      assert_raise ArgumentError, ~r/watch entry needs a :metric name/, fn ->
        Config.new!(watch: [[ceiling: 95.0]])
      end
    end

    test "a non-numeric ceiling" do
      error =
        assert_raise(ArgumentError, fn ->
          Config.new!(watch: [[metric: "disk.pct", ceiling: "95"]])
        end)

      assert error.message =~ ":ceiling"
      assert error.message =~ "disk.pct"
    end

    test "a non-numeric floor" do
      error =
        assert_raise(ArgumentError, fn ->
          Config.new!(watch: [[metric: "battery.pct", floor: "5"]])
        end)

      assert error.message =~ ":floor"
      assert error.message =~ "battery.pct"
    end

    test "a floor at or above the ceiling" do
      error =
        assert_raise(ArgumentError, fn ->
          Config.new!(watch: [[metric: "tank.level", floor: 90.0, ceiling: 10.0]])
        end)

      assert error.message =~ ":floor"
      assert error.message =~ ":ceiling"
      assert error.message =~ "tank.level"
    end

    test "non-map tags" do
      error =
        assert_raise(ArgumentError, fn ->
          Config.new!(watch: [[metric: "http.duration", tags: [route: "/api"]]])
        end)

      assert error.message =~ ":tags"
      assert error.message =~ "http.duration"
    end

    test "a non-boolean histogram" do
      error =
        assert_raise(ArgumentError, fn ->
          Config.new!(watch: [[metric: "http.duration", histogram: 1]])
        end)

      assert error.message =~ ":histogram"
      assert error.message =~ "http.duration"
    end

    test "a Metric struct passed directly is validated too" do
      error =
        assert_raise(ArgumentError, fn ->
          Config.new!(watch: [%Metric{name: "m", tags: nil}])
        end)

      assert error.message =~ ":tags"

      assert_raise ArgumentError, fn ->
        Config.new!(watch: [%Metric{name: :not_a_string}])
      end
    end
  end
end
