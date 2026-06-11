defmodule MobiusSmarts.Report do
  @moduledoc false
  # Renders `MobiusSmarts.status/1` plus recent observations as the
  # plain-text report behind `MobiusSmarts.report/1` — the
  # IEx-over-SSH view. Pure: takes the maps, returns a string.

  alias MobiusSmarts.Finding

  @spec render(map(), [Finding.t()]) :: String.t()
  def render(status, observations) do
    ([header(status), learning(status.learning), findings(status.findings)] ++
       recent(observations))
    |> List.flatten()
    |> Enum.join("\n")
  end

  defp header(status) do
    level = status.level |> Atom.to_string() |> String.upcase()

    "MobiusSmarts — #{level} — concern #{round1(status.concern)} — " <>
      "since #{fmt_ts(status.since)}"
  end

  defp learning([]), do: []
  defp learning(metrics), do: ["learning: #{Enum.join(metrics, ", ")}"]

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
end
