defmodule MobiusSmarts.BoardTest do
  use ExUnit.Case, async: true

  alias MobiusSmarts.{Board, Config, Finding}

  setup ctx do
    name = :"board_test_#{ctx.test}"
    config = Config.new!(watch: ["m"], clear_after: 3)
    board = start_supervised!({Board, name: name, config: config})

    handler_id = "board-test-#{inspect(ctx.test)}"

    :telemetry.attach_many(
      handler_id,
      [
        [:mobius_smarts, :finding, :raised],
        [:mobius_smarts, :finding, :escalated],
        [:mobius_smarts, :finding, :cleared],
        [:mobius_smarts, :health, :level_changed]
      ],
      &__MODULE__.forward_telemetry/4,
      self()
    )

    on_exit(fn -> :telemetry.detach(handler_id) end)

    %{board: board, name: name}
  end

  def forward_telemetry(event, measurements, metadata, parent) do
    send(parent, {:telemetry, event, measurements, metadata})
  end

  defp candidate(overrides \\ %{}) do
    Map.merge(
      %{
        kind: :drifting_up,
        detector: :drift,
        class: :condition,
        severity: :warning,
        concern: 1.1,
        onset: 1_700_000_000,
        evidence: %{bucket: 6.6},
        message: "drifting up"
      },
      overrides
    )
  end

  @scope {"m", %{}}
  @kinds [:drifting_up, :drifting_down]

  test "raising a condition emits telemetry and shows in findings and status", %{name: name} do
    Board.report(name, @scope, @kinds, [candidate()])

    assert_receive {:telemetry, [:mobius_smarts, :finding, :raised], %{concern: concern}, meta}
    assert concern == 1.1
    assert %Finding{kind: :drifting_up, status: :active, metric: "m"} = meta.finding

    assert [%Finding{kind: :drifting_up}] = Board.findings(name)
    assert %{level: :watch, findings: [_]} = Board.status(name)
    assert_receive {:telemetry, [:mobius_smarts, :health, :level_changed], _m, %{level: :watch}}
  end

  test "re-confirming updates in place without re-raising", %{name: name} do
    Board.report(name, @scope, @kinds, [candidate()])
    assert_receive {:telemetry, [:mobius_smarts, :finding, :raised], _m, _meta}

    Board.report(name, @scope, @kinds, [candidate(%{concern: 1.3, message: "worse"})])
    refute_receive {:telemetry, [:mobius_smarts, :finding, :raised], _m, _meta}, 50

    assert [%Finding{concern: 1.3, message: "worse"}] = Board.findings(name)
  end

  test "severity escalates, never silently de-escalates", %{name: name} do
    Board.report(name, @scope, @kinds, [candidate()])
    Board.report(name, @scope, @kinds, [candidate(%{severity: :critical, concern: 1.8})])

    assert_receive {:telemetry, [:mobius_smarts, :finding, :escalated], _m, meta}
    assert meta.finding.severity == :critical

    # A weaker re-confirmation keeps the escalated severity.
    Board.report(name, @scope, @kinds, [candidate(%{severity: :warning, concern: 1.2})])
    assert [%Finding{severity: :critical}] = Board.findings(name)
  end

  test "clears only after clear_after consecutive misses, with telemetry", %{name: name} do
    Board.report(name, @scope, @kinds, [candidate()])

    Board.report(name, @scope, @kinds, [])
    Board.report(name, @scope, @kinds, [])
    assert [_still_active] = Board.findings(name)

    Board.report(name, @scope, @kinds, [])
    assert Board.findings(name) == []
    assert_receive {:telemetry, [:mobius_smarts, :finding, :cleared], _m, meta}
    assert meta.finding.status == :cleared
    assert %{level: :ok} = Board.status(name)
  end

  test "a re-confirmation resets the miss count", %{name: name} do
    Board.report(name, @scope, @kinds, [candidate()])
    Board.report(name, @scope, @kinds, [])
    Board.report(name, @scope, @kinds, [])
    Board.report(name, @scope, @kinds, [candidate()])
    Board.report(name, @scope, @kinds, [])
    Board.report(name, @scope, @kinds, [])

    assert [_still_active] = Board.findings(name)
  end

  test "uncovered kinds are not cleared by a reporter that cannot see them", %{name: name} do
    Board.report(name, @scope, @kinds, [candidate()])

    # A sweep reporting only its own kinds must not age tick findings.
    for _sweep <- 1..5 do
      Board.report(name, @scope, [:approaching_limit], [])
    end

    assert [%Finding{kind: :drifting_up}] = Board.findings(name)
  end

  test "cleared trouble coming back is a fresh raise", %{name: name} do
    Board.report(name, @scope, @kinds, [candidate()])
    for _miss <- 1..3, do: Board.report(name, @scope, @kinds, [])
    assert_receive {:telemetry, [:mobius_smarts, :finding, :cleared], _m, _meta}

    Board.report(name, @scope, @kinds, [candidate()])
    assert_receive {:telemetry, [:mobius_smarts, :finding, :raised], _m, _meta}
    assert [%Finding{status: :active}] = Board.findings(name)
  end

  test "observations record once, never clear, never affect the level", %{name: name} do
    observation =
      candidate(%{kind: :reporting_gap, class: :observation, severity: :info, onset: 123})

    Board.report(name, @scope, @kinds, [observation])
    assert_receive {:telemetry, [:mobius_smarts, :finding, :raised], _m, _meta}

    # Re-derived from overlapping history: no second raise.
    Board.report(name, @scope, @kinds, [observation])
    refute_receive {:telemetry, [:mobius_smarts, :finding, :raised], _m, _meta}, 50

    assert Board.findings(name) == []
    assert [%Finding{kind: :reporting_gap, status: :noted}] = Board.observations(name)
    assert %{level: :ok} = Board.status(name)
  end

  test "the level ladder", %{name: name} do
    # 1 warning -> :watch
    Board.report(name, @scope, @kinds, [candidate()])
    assert %{level: :watch} = Board.status(name)

    # a critical condition -> :degraded
    Board.report(name, {"m2", %{}}, [:jumped], [
      candidate(%{kind: :jumped, severity: :critical, concern: 2.0})
    ])

    assert %{level: :degraded} = Board.status(name)

    # a critical :approaching_limit -> :critical
    Board.report(name, {"disk", %{}}, [:approaching_limit], [
      candidate(%{kind: :approaching_limit, severity: :critical, concern: 3.0})
    ])

    assert %{level: :critical} = Board.status(name)
  end

  test "three warnings count as degraded", %{name: name} do
    for metric <- ["a", "b", "c"] do
      Board.report(name, {metric, %{}}, @kinds, [candidate()])
    end

    assert %{level: :degraded} = Board.status(name)
  end

  test "learning lists watched metrics without baselines", %{name: name} do
    assert %{learning: ["m"]} = Board.status(name)

    Board.put_baseline(name, {"m", %{}}, %{target: 1.0, sigma_avg: 0.1, fitted_at: 0})
    assert %{learning: []} = Board.status(name)
    assert %{target: 1.0} = Board.baseline(name, {"m", %{}})
  end
end
