# MobiusProcessing

A bridge between [Mobius](https://github.com/mobius-home/mobius) metric storage
and the Nx ecosystem (Nx, Scholar, NxSignal). Pulls Mobius data out as Arrow
columns, hands it to Nx as tensors, and documents what processing the Nx
ecosystem can do on the typical shapes of data Nerves devices collect.

See [SPEC.md](SPEC.md) for the design.

## Installation

Add `mobius_processing` to your deps. We recommend pairing it with
[`nx_eigen`](https://github.com/elixir-nx/nx_eigen) as your Nx backend —
that's the CPU-SIMD-on-embedded story this library is built around, and
every example in the docs assumes it:

```elixir
def deps do
  [
    {:mobius_processing, "~> 0.1.0"},
    {:nx_eigen, "~> 0.1"}
  ]
end
```

Then configure `nx_eigen` as the default backend:

```elixir
# config/config.exs
config :nx, default_backend: NxEigen.Backend
```

`nx_eigen` is optional. Everything works on the built-in `Nx.BinaryBackend`
— slower but zero extra dependencies, which is the right pick for CI or
when you don't want a native toolchain in your build. You can also drop in
`EXLA` or `Torchx` if your host environment supports them.

### Optional companion deps

```elixir
{:scholar, "~> 0.3"},      # histograms, regression, clustering, PCA, ...
{:nx_signal, "~> 0.2"}     # FFT, filtering, spectrograms, peak detection
```

Both are loaded lazily — only the recipes that need them require them.

## Documentation

Documentation can be generated with
[ExDoc](https://github.com/elixir-lang/ex_doc) and published on
[HexDocs](https://hexdocs.pm). Once published, the docs can be found at
<https://hexdocs.pm/mobius_processing>.
