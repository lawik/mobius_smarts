defmodule MobiusSmarts.MixProject do
  use Mix.Project

  def project do
    [
      app: :mobius_smarts,
      version: "0.1.0",
      elixir: "~> 1.19",
      elixirc_paths: elixirc_paths(Mix.env()),
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      name: "Mobius smarts",
      description:
        "On-device health detection over Mobius metric storage: " <>
          "control charts, CUSUM, trend and distribution-drift detectors in Nx.",
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
        "guides/recipes/overview.md": [title: "Recipes — overview"]
      ],
      groups_for_extras: [
        Recipes: ~r"guides/recipes/.*"
      ],
      groups_for_modules: [
        Runtime: [
          MobiusSmarts,
          MobiusSmarts.Config,
          MobiusSmarts.Config.Metric,
          MobiusSmarts.Finding,
          MobiusSmarts.Calibrate
        ],
        Detectors: [
          MobiusSmarts.Detect,
          MobiusSmarts.Detect.Jump,
          MobiusSmarts.Detect.Shift,
          MobiusSmarts.Detect.Drift,
          MobiusSmarts.Detect.Trend,
          MobiusSmarts.Detect.Changepoint,
          MobiusSmarts.Detect.Shape,
          MobiusSmarts.Detect.Novelty,
          MobiusSmarts.Detect.Outlier
        ],
        Recipes: [
          MobiusSmarts.Recipes.CoreNx,
          MobiusSmarts.Recipes.DefnKernels
        ]
      ]
    ]
  end

  def package do
    [
      name: :mobius_smarts,
      licenses: ["Apache-2.0"],
      links: %{"GitHub" => "https://github.com/TODO/mobius_smarts"}
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

  defp elixirc_paths(:test), do: ["lib", "test/support"]
  defp elixirc_paths(_env), do: ["lib"]

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:nx, "~> 0.7"},
      {:telemetry, "~> 1.0"},
      # Tracking main for the summary-window / histogram APIs; back to
      # hex once released.
      {:mobius, github: "mobius-home/mobius", branch: "main"},
      {:nstandard, "~> 0.3", only: [:dev, :test], runtime: false},
      {:stream_data, "~> 1.1", only: [:dev, :test]},
      {:igniter, "~> 0.6", only: [:dev, :test]},
      {:ex_doc, "~> 0.40", only: [:dev, :test], runtime: false},
      {:dialyxir, "~> 1.4", only: [:dev, :test], runtime: false},
      {:credo, "~> 1.7", only: [:dev, :test], runtime: false},
      {:spellweaver, "~> 0.1", only: [:dev, :test], runtime: false}
    ]
  end
end
