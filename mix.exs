defmodule MobiusProcessing.MixProject do
  use Mix.Project

  def project do
    [
      app: :mobius_processing,
      version: "0.1.0",
      elixir: "~> 1.19",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      name: "Mobius processing",
      description:
        "Bridge between Mobius metric storage and the Nx ecosystem: " <>
          "pulls Mobius data out as Arrow columns and hands it to Nx as tensors.",
      docs: docs(),
      package: package(),
      aliases: aliases(),
      dialyzer: dialyzer()
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger]
    ]
  end

  def docs do
    [
      main: "readme",
      extra_section: "GUIDES",
      extras: [
        "README.md",
        "SPEC.md": [title: "Design spec"],
        "guides/recipes/overview.md": [title: "Recipes — overview"],
        "guides/recipes/core_nx.md": [title: "Core Nx"],
        "guides/recipes/scholar.md": [title: "Scholar"],
        "guides/recipes/nx_signal.md": [title: "NxSignal"],
        "guides/recipes/defn_kernels.md": [title: "Hand-written defn kernels"]
      ],
      groups_for_extras: [
        Recipes: ~r"guides/recipes/.*"
      ]
    ]
  end

  def package do
    [
      name: :mobius_processing,
      licenses: ["Apache-2.0"],
      links: %{"GitHub" => "https://github.com/TODO/mobius_processing"}
    ]
  end

  def aliases do
    [
      check: [
        "hex.audit",
        "compile --warnings-as-errors --force",
        "format --check-formatted",
        "credo",
        "deps.unlock --check-unused",
        "spellweaver.check",
        "dialyzer"
      ],
      precommit: [
        "hex.audit",
        "compile --warnings-as-errors --force",
        "format",
        "credo",
        "deps.unlock --unused",
        "spellweaver.check",
        "dialyzer",
        "test"
      ]
    ]
  end

  def cli do
    [
      preferred_envs: [precommit: :test]
    ]
  end

  def dialyzer do
    [
      plt_add_apps: [:mix],
      ignore_warnings: ".dialyzer_ignore.exs"
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:nx, "~> 0.7"},
      {:arrow, path: "../arrow"},
      {:mobius, "~> 0.6"},
      {:jason, "~> 1.4"},
      {:scholar, "~> 0.3", optional: true},
      {:nx_signal, "~> 0.2", optional: true},
      {:nstandard, "~> 0.3"},
      {:igniter, "~> 0.6", only: [:dev, :test]},
      {:ex_doc, "~> 0.40", only: [:dev, :test], runtime: false},
      {:dialyxir, "~> 1.4", only: [:dev, :test], runtime: false},
      {:credo, "~> 1.7", only: [:dev, :test], runtime: false},
      {:spellweaver, "~> 0.1", only: [:dev, :test], runtime: false}
    ]
  end
end
