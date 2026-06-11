defmodule MobiusSmarts.StubSource do
  @moduledoc false
  # A `MobiusSmarts.Source`-shaped data module for runtime tests:
  # serves whatever windows the test has staged, keyed by the
  # `:mobius_instance` option so async tests don't collide. The staged
  # value is a fun or a map of `{name, tags} => windows`, where
  # windows are flat `%{timestamp:, average:, std_dev:, reports:}`
  # maps (the `Mobius.Data.summary_windows/3` shape).

  alias MobiusSmarts.Source

  def stage(instance, data) do
    :persistent_term.put({__MODULE__, instance}, data)
  end

  def clear(instance) do
    :persistent_term.erase({__MODULE__, instance})
  end

  def summary_series(metric_name, tags, opts) do
    instance = Keyword.fetch!(opts, :mobius_instance)

    windows =
      case :persistent_term.get({__MODULE__, instance}, %{}) do
        fun when is_function(fun, 3) -> fun.(metric_name, tags, opts)
        data when is_map(data) -> Map.get(data, {metric_name, tags}, [])
      end

    Source.from_summary_windows(windows)
  end

  # Sketches are staged as `{:sketch, {name, tags}} => fun/1` in the
  # same data map; the fun receives the sketch opts (so tests can
  # assert on the requested windows) and returns the sketch result.
  def sketch(metric_name, tags, opts) do
    instance = Keyword.fetch!(opts, :mobius_instance)

    case :persistent_term.get({__MODULE__, instance}, %{}) do
      data when is_map(data) ->
        case Map.get(data, {:sketch, {metric_name, tags}}) do
          fun when is_function(fun, 1) -> fun.(opts)
          nil -> {:error, :not_staged}
        end

      _fun ->
        {:error, :not_staged}
    end
  end
end
