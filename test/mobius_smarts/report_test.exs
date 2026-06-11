defmodule MobiusSmarts.ReportTest do
  use ExUnit.Case, async: true

  alias MobiusSmarts.{Finding, Report}

  defp finding(overrides) do
    struct!(
      %Finding{
        metric: "vm.memory.total",
        tags: %{},
        detector: :drift,
        kind: :drifting_up,
        class: :condition,
        severity: :critical,
        onset: 1_781_157_840,
        raised_at: 1_781_157_900,
        last_seen_at: 1_781_157_900,
        status: :active,
        concern: 57.91,
        evidence: %{},
        message: "drifting up since ~2026-06-11 06:44Z"
      },
      overrides
    )
  end

  defp metric_entry(overrides) do
    Map.merge(
      %{
        metric: "vm.memory.total",
        tags: %{},
        detection: :active,
        detectors: [:jump, :shift, :drift, :shape, :changepoint],
        learning: nil
      },
      Map.new(overrides)
    )
  end

  test "renders level, metric posture, findings, and observations as scannable text" do
    status = %{
      level: :degraded,
      since: 1_781_124_863,
      concern: 96.44,
      novelty: :active,
      metrics: [
        metric_entry([]),
        metric_entry(
          metric: "vm.memory.binary",
          detection: :learning,
          detectors: [:changepoint],
          learning: %{reason: :insufficient, windows: 37, needed: 60, eta_s: 1380}
        ),
        metric_entry(
          metric: "vm.total_run_queue_lengths.io",
          detection: :blocked,
          detectors: [:changepoint],
          learning: %{reason: :no_dispersion, windows: 117, needed: 60, eta_s: nil}
        )
      ],
      findings: [
        finding(metric: "*", kind: :novel_behavior, concern: 96.44, message: "way off habits"),
        finding(severity: :warning, kind: :flatlined, concern: 1.01, message: "went flat")
      ],
      updated_at: 1_781_157_900
    }

    observations = [
      finding(kind: :spiked, class: :observation, severity: :info, message: "spiked, returned")
    ]

    report = Report.render(status, observations)

    assert report =~
             "MobiusSmarts — DEGRADED — concern 96.4 — since 2026-06-10 20:54Z — novelty active"

    # One posture line per metric: state cell plus armed detectors.
    assert report =~ ~r/vm\.memory\.total\s+active\s+jump shift drift shape changepoint/
    assert report =~ ~r/vm\.memory\.binary\s+learning 37\/60 \(~23m\)\s+changepoint/
    assert report =~ ~r/io\s+blocked: no within-window dispersion\s+changepoint/
    # Worst first, the novelty stream labeled as cross-metric.
    assert report =~ ~r/crit\s+96\.4×\s+novel_behavior\s+\(cross-metric\)/
    assert report =~ "way off habits"
    assert report =~ ~r/warn\s+1\.0×\s+flatlined/
    assert report =~ "recent observations:"
    assert report =~ "spiked, returned"
  end

  test "a healthy instance reads as such" do
    status = %{
      level: :ok,
      since: 1_781_124_863,
      concern: 0.0,
      novelty: :learning,
      metrics: [metric_entry([])],
      findings: [],
      updated_at: 1_781_157_900
    }

    report = Report.render(status, [])

    assert report =~ "MobiusSmarts — OK — concern 0.0"
    assert report =~ "novelty learning"
    assert report =~ "no active findings"
    refute report =~ "recent observations:"
  end
end
