defmodule MobiusSmarts.CalibrateTest do
  use ExUnit.Case, async: true

  alias MobiusSmarts.{Calibrate, Config}

  doctest Calibrate

  describe "probit/1" do
    test "matches known normal quantiles" do
      assert_in_delta Calibrate.probit(0.5), 0.0, 1.0e-9
      assert_in_delta Calibrate.probit(0.8413447), 1.0, 1.0e-4
      assert_in_delta Calibrate.probit(0.9986501), 3.0, 1.0e-4
    end

    test "is symmetric" do
      assert_in_delta Calibrate.probit(0.2), -Calibrate.probit(0.8), 1.0e-9
    end
  end

  describe "shewhart_limit/1" do
    test "recovers the textbook trio: 2-sigma, 3-sigma" do
      # P(|Z| > 2) = 0.0455 -> ARL 21.98; P(|Z| > 3) = 0.0027 -> ARL 370.4
      assert_in_delta Calibrate.shewhart_limit(21.98), 2.0, 0.01
      assert_in_delta Calibrate.shewhart_limit(370.4), 3.0, 0.01
    end

    test "grows with the budget" do
      assert Calibrate.shewhart_limit(10_000) > Calibrate.shewhart_limit(100)
    end
  end

  describe "cusum_h/2" do
    test "round-trips Siegmund: the detector default h=5, k=0.5 is ~940 per side" do
      assert_in_delta Calibrate.cusum_h(938.2, 0.5), 5.0, 0.05
    end

    test "monotone in ARL" do
      assert Calibrate.cusum_h(100_000, 0.5) > Calibrate.cusum_h(1_000, 0.5)
    end
  end

  describe "chi2_threshold/2" do
    test "df=2 at 95% is sqrt(5.99) within Wilson-Hilferty accuracy" do
      assert_in_delta Calibrate.chi2_threshold(20.0, 2), :math.sqrt(5.991), 0.05
    end

    test "more metrics need a higher threshold at the same rate" do
      assert Calibrate.chi2_threshold(1000.0, 10) > Calibrate.chi2_threshold(1000.0, 2)
    end
  end

  describe "for_config/1" do
    test "splits the budget across every alarm stream" do
      config =
        Config.new!(
          watch: ["a", "b", "c"],
          resolution: {1, :minute},
          false_alarm_every: {1, :week}
        )

      calib = Calibrate.for_config(config)

      # 10080 windows/week x (4 streams x 3 metrics + novelty) = 131_040
      assert_in_delta calib.arl, 131_040.0, 1.0
      # And the derived limits sit in a sane SPC range.
      assert calib.jump_limit > 4.0 and calib.jump_limit < 5.5
      assert calib.cusum_h > 8.0 and calib.cusum_h < 14.0
    end

    test "a tighter budget means lower thresholds" do
      loose =
        Calibrate.for_config(
          Config.new!(watch: ["a"], resolution: {1, :minute}, false_alarm_every: {30, :day})
        )

      tight =
        Calibrate.for_config(
          Config.new!(watch: ["a"], resolution: {1, :minute}, false_alarm_every: {1, :hour})
        )

      assert tight.jump_limit < loose.jump_limit
      assert tight.cusum_h < loose.cusum_h
    end
  end
end
