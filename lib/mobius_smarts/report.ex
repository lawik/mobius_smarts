defmodule MobiusSmarts.Report do
  @moduledoc false
  # Renders `MobiusSmarts.status/1` plus recent observations as the
  # plain-text report behind `MobiusSmarts.report/1` — the
  # IEx-over-SSH view. Pure: takes the maps, returns a string.

  alias MobiusSmarts.Finding

  @spec render(map(), [Finding.t()]) :: String.t()
  def render(status, observations) do
    ([header(status)] ++
       metrics(status.metrics) ++ findings(status.findings) ++ recent(observations))
    |> List.flatten()
    |> Enum.join("\n")
  end

  defp header(status) do
    level = status.level |> Atom.to_string() |> String.upcase()

    "MobiusSmarts — #{level} — concern #{round1(status.concern)} — " <>
      "since #{fmt_ts(status.since)} — novelty #{status.novelty}"
  end

  defp metrics([]), do: []

  defp metrics(metrics) do
    name_width = metrics |> Enum.map(&String.length(&1.metric)) |> Enum.max()
    cells = Enum.map(metrics, &{&1, state_cell(&1)})
    cell_width = cells |> Enum.map(fn {_m, cell} -> String.length(cell) end) |> Enum.max()

    rows =
      for {entry, cell} <- cells do
        "  #{String.pad_trailing(entry.metric, name_width)}  " <>
          "#{String.pad_trailing(cell, cell_width)}  " <>
          Enum.map_join(entry.detectors, " ", &Atom.to_string/1)
      end

    ["", "metrics:" | rows]
  end

  defp state_cell(%{detection: :active}), do: "active"

  defp state_cell(%{detection: :unstable, learning: progress}) do
    "unstable: won't settle (#{progress.windows}/#{progress.needed})"
  end

  defp state_cell(%{learning: %{reason: :no_data}}), do: "no data yet"

  defp state_cell(%{learning: %{reason: :trending}}), do: "learning: still ramping"

  defp state_cell(%{learning: %{reason: :no_dispersion}}),
    do: "blocked: no within-window dispersion"

  defp state_cell(%{learning: %{reason: :zero_variance}}), do: "blocked: zero variance"

  defp state_cell(%{learning: %{reason: reason} = progress}) do
    verb = if reason == :unsettled, do: "resettling", else: "learning"
    "#{verb} #{progress.windows}/#{progress.needed} (~#{fmt_dur(progress.eta_s)})"
  end

  defp findings([]), do: ["", "  no active findings"]

  defp findings(findings) do
    kind_width = findings |> Enum.map(&String.length(Atom.to_string(&1.kind))) |> Enum.max()

    rows =
      Enum.flat_map(findings, fn finding ->
        [
          "  #{sev(finding.severity)}  #{String.pad_leading(round1(finding.concern), 6)}×  " <>
            "#{String.pad_trailing(Atom.to_string(finding.kind), kind_width)}  " <>
            metric(finding.metric),
          "  #{String.duplicate(" ", 12 + kind_width)} #{finding.message}"
        ]
      end)

    ["" | rows]
  end

  defp recent([]), do: []

  defp recent(observations) do
    rows =
      for obs <- observations do
        "  #{fmt_ts(obs.raised_at)}  #{metric(obs.metric)} — #{obs.message}"
      end

    ["", "recent observations:" | rows]
  end

  # The novelty stream reports against the whole instance, not a metric.
  defp metric("*"), do: "(cross-metric)"
  defp metric(name), do: name

  defp sev(:critical), do: "crit"
  defp sev(:warning), do: "warn"
  defp sev(:info), do: "info"

  defp round1(value), do: :erlang.float_to_binary(value * 1.0, decimals: 1)

  defp fmt_ts(unix) do
    unix |> DateTime.from_unix!() |> Calendar.strftime("%Y-%m-%d %H:%MZ")
  end

  defp fmt_dur(s) when s < 90, do: "#{s}s"
  defp fmt_dur(s) when s < 90 * 60, do: "#{round(s / 60)}m"
  defp fmt_dur(s) when s < 36 * 3600, do: "#{Float.round(s / 3600, 1)}h"
  defp fmt_dur(s), do: "#{Float.round(s / 86_400, 1)} days"
end
