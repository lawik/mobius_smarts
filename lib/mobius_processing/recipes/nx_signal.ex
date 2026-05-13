defmodule MobiusProcessing.Recipes.NxSignal do
  @moduledoc """
  Recipes that need NxSignal — the time-series and DSP toolkit. Right
  for any signal with cyclic structure: a compressor cycling, a fan
  modulating, a daily ambient pattern, a vibration trace. Optional dep:

      {:nx_signal, "~> 0.2"}

  ## Is the compressor cycling, and how often?

  A fridge's `cabinet_temp_c` over a few hours has a clear sawtooth.
  `dominant_period_seconds/2` pulls the dominant cycle out with FFT.

  ## Smoothing a noisy accelerometer trace

  `low_pass/3` builds a windowed-sinc FIR filter (via
  `NxSignal.Filters.firwin/3`) and convolves it through the signal.
  Sharper than a rolling mean — the cutoff frequency is explicit.

  ## Detecting button presses from a vibration signal

  `count_peaks_above/3` finds local maxima with a minimum-separation
  constraint and counts how many cross a height threshold.

  ## Windowing to clean up the spectrum

  `windowed_spectrum_magnitudes/1` multiplies a Hann window into the
  signal before the FFT. Sharper peaks; better at separating two
  close-by frequencies.

  ## Why detrend before FFT

  The DC component (mean) of a signal dominates the FFT magnitude
  spectrum at bin 0. The recipes below all detrend by subtracting the
  mean first — `detrend_constant/1` is the helper. Without it, every
  spectrum's biggest peak is bin 0 and the real periodicity is invisible.

  NxSignal's other dep — `NxSignal.Convolution.correlate/2`, used by
  `MobiusProcessing.Recipes.DefnKernels.lag_samples/3` — also lives here
  in spirit.
  """

  @doc """
  Subtracts the mean. Removes the DC component before spectral analysis.
  Use this in place of the scipy-style `detrend(type: :constant)` —
  NxSignal doesn't expose a `detrend` function of its own.

  ## Examples

      iex> t = Nx.tensor([2.0, 4.0, 6.0, 8.0])
      iex> MobiusProcessing.Recipes.NxSignal.detrend_constant(t)
      ...> |> Nx.to_flat_list()
      [-3.0, -1.0, 1.0, 3.0]
  """
  @spec detrend_constant(Nx.Tensor.t()) :: Nx.Tensor.t()
  def detrend_constant(signal) do
    Nx.subtract(signal, Nx.mean(signal))
  end

  @doc """
  Period of the dominant non-DC frequency component, in seconds.

  Detrends, FFTs, takes magnitude, finds the argmax of the first half of
  the spectrum (skipping the zero bin), converts the bin index to Hz,
  inverts to a period.

  ## Examples

      iex> # 5 Hz sine sampled at 100 Hz for 2 seconds — period is 0.2 s.
      iex> t = Nx.iota({200}, type: :f32) |> Nx.divide(100.0)
      iex> signal = Nx.sin(Nx.multiply(2.0 * :math.pi() * 5.0, t))
      iex> MobiusProcessing.Recipes.NxSignal.dominant_period_seconds(signal, 100.0)
      ...> |> Float.round(2)
      0.2
  """
  @spec dominant_period_seconds(Nx.Tensor.t(), float()) :: float()
  def dominant_period_seconds(signal, sample_rate_hz) do
    centered = detrend_constant(signal)
    n = Nx.size(centered)
    half = div(n, 2)

    spectrum = centered |> Nx.fft() |> Nx.abs()
    peak_bin = spectrum |> Nx.slice([1], [half - 1]) |> Nx.argmax() |> Nx.to_number()
    peak_hz = (peak_bin + 1) * sample_rate_hz / n
    1.0 / peak_hz
  end

  @doc """
  Convolves `signal` with a windowed-sinc FIR low-pass filter of
  `num_taps` taps cutting off at `cutoff_hz`. `num_taps` must be odd
  for a true low-pass response at Nyquist.

  Uses `NxSignal.Filters.firwin/3` (Hamming window by default) and
  `NxSignal.Convolution.convolve/2`.

  ## Examples

      iex> # Compose 1 Hz signal + 30 Hz noise sampled at 100 Hz.
      iex> # A 5 Hz low-pass should suppress the noise heavily.
      iex> t = Nx.iota({200}, type: :f32) |> Nx.divide(100.0)
      iex> low = Nx.sin(Nx.multiply(2.0 * :math.pi() * 1.0, t))
      iex> high = Nx.sin(Nx.multiply(2.0 * :math.pi() * 30.0, t))
      iex> noisy = Nx.add(low, high)
      iex> smooth = MobiusProcessing.Recipes.NxSignal.low_pass(noisy, 5.0, 51, sample_rate_hz: 100.0)
      iex> # Variance after filtering is lower than before.
      iex> Nx.to_number(Nx.variance(smooth)) < Nx.to_number(Nx.variance(noisy))
      true
  """
  @spec low_pass(Nx.Tensor.t(), float(), pos_integer(), keyword()) :: Nx.Tensor.t()
  def low_pass(signal, cutoff_hz, num_taps, opts \\ []) do
    sample_rate = Keyword.get(opts, :sample_rate_hz, 1.0)

    taps =
      NxSignal.Filters.firwin(num_taps, [cutoff_hz],
        sampling_rate: sample_rate,
        window: :hamming
      )

    NxSignal.Convolution.convolve(signal, taps, mode: :same)
  end

  @doc """
  Counts local maxima in `signal` that are at least `height` and
  separated by at least `min_distance` samples.

  Built on `NxSignal.PeakFinding.argrelmax/2`. The recipe shape is
  "discrete events extracted from a continuous signal" — button presses
  from a vibration trace, heartbeats from a sensor, footsteps from a
  strain gauge.

  ## Examples

      iex> # Two clear peaks above 2.0 — at indices 3 and 8.
      iex> signal = Nx.tensor([0.0, 0.1, 0.5, 3.0, 0.5, 0.1, 0.0, 0.5, 2.5, 0.3])
      iex> MobiusProcessing.Recipes.NxSignal.count_peaks_above(signal, 2.0, 2)
      2

      iex> # Same peaks but height threshold excludes the smaller one.
      iex> signal = Nx.tensor([0.0, 0.1, 0.5, 3.0, 0.5, 0.1, 0.0, 0.5, 2.5, 0.3])
      iex> MobiusProcessing.Recipes.NxSignal.count_peaks_above(signal, 2.8, 2)
      1
  """
  @spec count_peaks_above(Nx.Tensor.t(), float(), pos_integer()) :: non_neg_integer()
  def count_peaks_above(signal, height, min_distance) do
    %{indices: idx, valid_indices: nvalid} =
      NxSignal.PeakFinding.argrelmax(signal, order: min_distance)

    nvalid_int = Nx.to_number(nvalid)

    if nvalid_int == 0 do
      0
    else
      valid_1d =
        idx
        |> Nx.slice_along_axis(0, nvalid_int, axis: 0)
        |> Nx.flatten()

      values = Nx.take(signal, valid_1d)

      values
      |> Nx.greater_equal(height)
      |> Nx.sum()
      |> Nx.to_number()
    end
  end

  @doc """
  Hann-windowed FFT magnitude spectrum. Reduces spectral leakage
  compared to a raw FFT — sharper peaks, better at separating two
  close-by frequencies. Returns a `{n}` float tensor.

  ## Examples

      iex> # 5 Hz sine at 100 Hz sample rate, 200 samples.
      iex> t = Nx.iota({200}, type: :f32) |> Nx.divide(100.0)
      iex> signal = Nx.sin(Nx.multiply(2.0 * :math.pi() * 5.0, t))
      iex> mags = MobiusProcessing.Recipes.NxSignal.windowed_spectrum_magnitudes(signal)
      iex> half = div(Nx.size(mags), 2)
      iex> peak_bin = mags |> Nx.slice([1], [half - 1]) |> Nx.argmax() |> Nx.to_number()
      iex> peak_bin + 1
      10
  """
  @spec windowed_spectrum_magnitudes(Nx.Tensor.t()) :: Nx.Tensor.t()
  def windowed_spectrum_magnitudes(signal) do
    n = Nx.size(signal)
    window = NxSignal.Windows.hann(n)
    centered = detrend_constant(signal)

    centered
    |> Nx.multiply(window)
    |> Nx.fft()
    |> Nx.abs()
  end
end
