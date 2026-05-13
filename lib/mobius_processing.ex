defmodule MobiusProcessing do
  @moduledoc """
  Bridge between [Mobius](https://github.com/mobius-home/mobius) metric storage
  and the Nx ecosystem.

  Two thin layers of pure functions:

  - `MobiusProcessing.Source` — pulls Mobius history into `Arrow.RecordBatch`.
  - `MobiusProcessing.Tensor` — converts Arrow columns into Nx tensors.

  What you can *do* with those tensors — recipes drawing on Nx core,
  Scholar and NxSignal — is documented under the **Recipes** guides
  in the generated docs.

  This library does not pick an Nx backend; the caller does. The examples
  and docs assume `nx_eigen` as the demonstration backend because it's the
  right pick for the embedded Nerves context this library is aimed at.
  `Nx.BinaryBackend` works for everything here with no extra dep, slower
  but zero friction.

  The full design lives in `SPEC.md` at the project root.
  """
end
