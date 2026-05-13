# Recipes with NxSignal

NxSignal is where the time-series and DSP recipes live. Right for any
signal with cyclic structure — a compressor cycling, a fan modulating,
a daily ambient pattern, a vibration trace. Optional dep:

```elixir
{:nx_signal, "~> 0.2"}
```

## Is the compressor cycling, and how often?

A fridge's `cabinet_temp_c` over a few hours has a clear sawtooth: rise
slowly while the compressor is off, drop sharply when it kicks in.
FFT pulls the dominant period out of that pattern in one call.

```elixir
# {n_samples} f32, °C, sampled at 1Hz for a few hours
cabinet_temp_c = pull_metric_tensor("cabinet_temperature")

# Detrend first — a non-zero mean dominates the spectrum and hides
# the real periodicity.
centered = NxSignal.detrend(cabinet_temp_c, type: :constant)

spectrum     = NxSignal.fft(centered) |> Nx.abs()
sample_rate  = 1.0
n            = Nx.size(centered)
freqs_hz     = Nx.iota({n}, type: :f32) |> Nx.multiply(sample_rate / n)

# Ignore the zero-frequency bin and look at the first half (real signal,
# symmetric spectrum).
peak_bin   = Nx.slice(spectrum, [1], [div(n, 2) - 1]) |> Nx.argmax() |> Nx.to_number()
peak_hz    = Nx.to_number(freqs_hz[peak_bin + 1])
period_min = 1.0 / peak_hz / 60.0
```

`period_min` is the cycle length in minutes. A fridge that should
cycle every 20 minutes but suddenly cycles every 5 has a problem —
the door seal, the refrigerant, the thermostat.

## Smoothing a noisy accelerometer trace

Vibration data from an MPU6050 is fundamentally noisy; you don't want
the noise, you want the underlying motion. A low-pass filter is a
sharper tool than a rolling mean — you specify the cutoff frequency.

```elixir
# {n_samples} f32, m/s², sampled at 100Hz
accel_z = pull_metric_tensor("accelerometer_z")
sample_rate_hz = 100.0
cutoff_hz      = 5.0

# Length-101 windowed-sinc FIR filter: pass < 5Hz, attenuate above.
taps   = NxSignal.Filters.fir_filter({101}, cutoff_hz / (sample_rate_hz / 2))
smooth_accel_z = NxSignal.Filters.convolve(accel_z, taps)
```

Compare with `Nx.window_mean(accel_z, 20)` — that also smooths, but the
moving average passes some high-frequency junk and rounds off real
peaks. A proper low-pass keeps the peaks sharp.

## Detecting button presses from a vibration signal

A button press is a brief spike in `vibration_g`. Peak-finding picks
those out as discrete events.

```elixir
vibration_g = pull_metric_tensor("vibration_magnitude_g")

peaks = NxSignal.Peaks.find_peaks(vibration_g,
  height: 2.0,        # at least 2g
  distance: 50        # at least 50 samples apart (debounce)
)

press_count = Nx.size(peaks)
```

The same recipe works for any "find the events in a continuous
signal" task — heartbeats in an optical sensor, footsteps in a
strain-gauge, claps in a microphone level.

## Watching the fan's frequency change over time

The compressor recipe above gives one number for a whole window. A
*spectrogram* shows how that number drifts — useful when you want to
catch the moment a device shifts behavior, not just summarise.

```elixir
# 1Hz cpu_usage_percent for the last day
cpu_usage_pct = pull_metric_tensor("cpu_usage_percent")

# 10-minute STFT windows, 75% overlap
{spec, _times, _freqs} =
  NxSignal.spectrogram(cpu_usage_pct,
    fs: 1.0,
    nperseg: 600,
    noverlap: 450
  )
# spec is {n_freq_bins, n_time_bins}; bright bands moving up/down show
# the dominant cycle period changing through the day.
```

When the device's main job-loop frequency shifts — a periodic task
slows down, a cron job moves, a background process kicks in — the
spectrogram will show a band moving.

## Why detrend before FFT

The DC component (mean) of a signal dominates the FFT magnitude
spectrum at bin 0. If you forget to detrend, that's the only peak
you'll see and every real periodic component will be invisible.

```elixir
# Raw — bin 0 swamps everything
NxSignal.fft(cabinet_temp_c) |> Nx.abs() |> Nx.argmax()
# => 0 (the DC term — useless)

# Detrended — real periodicity emerges
cabinet_temp_c
|> NxSignal.detrend(type: :constant)   # remove mean
|> NxSignal.fft()
|> Nx.abs()
|> Nx.argmax()
# => index of the actual dominant frequency
```

`type: :constant` removes the mean; `type: :linear` removes a linear
trend as well — use that on signals that are drifting upward or
downward over the window (a slowly warming room, a slowly-filling
buffer).

## Windowing to clean up the spectrum

A raw FFT of a finite chunk has "spectral leakage" — energy from each
real frequency smears into nearby bins because of the implicit
rectangular window at the edges. A Hann window suppresses that.

```elixir
n        = Nx.size(cabinet_temp_c)
window   = NxSignal.Windows.hann({n})
windowed = Nx.multiply(NxSignal.detrend(cabinet_temp_c), window)

spectrum = NxSignal.fft(windowed) |> Nx.abs()
# Sharper peaks; better at separating two close-by frequencies.
```

Reach for `hann` or `hamming` by default for general spectral
analysis. `blackman` if you really need to separate two close
frequencies and don't mind a slightly broader main lobe.
