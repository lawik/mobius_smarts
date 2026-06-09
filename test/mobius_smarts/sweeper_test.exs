defmodule MobiusSmarts.SweeperTest do
  use ExUnit.Case, async: true

  alias MobiusSmarts.{Board, Config, StubSource, Sweeper}

  @metric "http.request.duration"
  @tags_a %{route: "/a"}
  @tags_b %{route: "/b"}

  setup ctx do
    name = :"sweeper_test_#{ctx.test}"
    instance = name

    config =
      Config.new!(
        watch: [
          [metric: @metric, tags: @tags_a],
          [metric: @metric, tags: @tags_b]
        ],
        source: StubSource,
        mobius_instance: instance,
        # Long enough that the scheduled sweep never fires on its own;
        # the test drives sweeps explicitly.
        sweep_interval: {1, :hour},
        min_baseline_windows: 20
      )

    board = start_supervised!({Board, name: name, config: config})

    on_exit(fn -> StubSource.clear(instance) end)

    %{name: name, instance: instance, config: config, board: board}
  end

  defp healthy_windows(now, seed) do
    :rand.seed(:exsss, {seed, seed * 7, seed * 13})

    Enum.map(0..119, fn i ->
      %{
        timestamp: now - (120 - i) * 60,
        average: 50.0 + :rand.normal(),
        std_dev: 2.0 + 0.1 * abs(:rand.normal()),
        reports: 30
      }
    end)
  end

  defp drift_candidate(now) do
    %{
      kind: :drifting_up,
      detector: :drift,
      class: :condition,
      severity: :warning,
      concern: 1.2,
      onset: now - 600,
      evidence: %{bucket: 6.6},
      message: "drifting up"
    }
  end

  defp run_sweep(name, config) do
    sweeper = start_supervised!({Sweeper, name: name, config: config})
    send(sweeper, :sweep)
    # A sync call drains the mailbox up to and including :sweep.
    :sys.get_state(sweeper)
  end

  test "a finding on one tag set does not block refits for siblings sharing the name", ctx do
    now = System.system_time(:second)
    stale_fitted_at = now - 200_000

    StubSource.stage(ctx.instance, %{
      {@metric, @tags_a} => healthy_windows(now, 11),
      {@metric, @tags_b} => healthy_windows(now, 22)
    })

    stale = %{target: 50.0, sigma_avg: 0.5, fitted_at: stale_fitted_at}
    Board.put_baseline(ctx.name, {@metric, @tags_a}, stale)
    Board.put_baseline(ctx.name, {@metric, @tags_b}, stale)

    # Route /a is in trouble; route /b is quiet and due for a refit.
    Board.report(ctx.name, {@metric, @tags_a}, [:drifting_up], [drift_candidate(now)])
    assert [%{metric: @metric, tags: @tags_a}] = Board.findings(ctx.name)

    run_sweep(ctx.name, ctx.config)

    # The quiet sibling's baseline must be refreshed...
    assert %{fitted_at: refit_at} = Board.baseline(ctx.name, {@metric, @tags_b})
    assert refit_at > stale_fitted_at

    # ...while the troubled metric keeps its old baseline, so the
    # active drift is not learned as the new normal.
    assert %{fitted_at: ^stale_fitted_at} = Board.baseline(ctx.name, {@metric, @tags_a})
  end
end
