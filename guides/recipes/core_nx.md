# Core Nx

What you can do on Mobius data using only Nx — no Scholar, no
NxSignal. Each recipe is framed around a question you might ask of
the kinds of metrics a Nerves device tends to emit; the data names
are illustrative.

All snippets assume you've already pulled the relevant column out into
a tensor — see `MobiusProcessing.Source` for the long-format batch and
`MobiusProcessing.Tensor.from_column/1` for `{tensor, validity}`.

## Is the CPU running hot?

Given a `cpu_temp_c` tensor at 1Hz from `nerves_motd`, the smallest
useful answer is "what's the worst it got, on any 5-minute window in
the last hour?" — short spikes hide in a plain mean.

```elixir
# {3600} f32, °C, one sample per second
cpu_temp_c = pull_metric_tensor("cpu_temperature")

window_max_5m = Nx.window_max(cpu_temp_c, 300)
overall_p95   = Nx.quantile(cpu_temp_c, 0.95)
ran_hot? =
  window_max_5m
  |> Nx.greater(75.0)
  |> Nx.any()
  |> Nx.to_number() == 1
```

`window_max` is the 5-minute envelope; `p95` is what "typical worst"
looked like. `ran_hot?` lifts the conclusion out into a boolean so a
caller can act on it.

## Is memory leaking?

Memory pressure rarely jumps — it creeps. A linear fit over a long
window catches a creeping leak in a way a rolling mean won't.

```elixir
# 24h at 1Hz: {86_400} f32, % used
mem_used_pct = pull_metric_tensor("memory_used_percent")
n            = Nx.size(mem_used_pct)
t_sec        = Nx.iota({n}, type: :f32)

# Least-squares fit of mem_used_pct ≈ a + b·t.
# Build the design matrix [1, t] and solve.
design        = Nx.stack([Nx.broadcast(1.0, {n}), t_sec], axis: 1)
{coeffs, _}   = Nx.LinAlg.lstsq(design, mem_used_pct)
[_intercept, slope_per_sec] = Nx.to_flat_list(coeffs)
slope_per_day = slope_per_sec * 86_400.0
```

`slope_per_day` is the answer: how many percentage points of RAM the
device gains per day. Positive and large means a leak; close to zero
means stable.

## How fast is the disk filling, and when will it fill up?

The disk-usage percentage moves slowly; a forward projection of the
slope tells you whether you have hours or weeks.

```elixir
# Daily samples of disk_used_percentage from filesystem_stats
disk_used_pct = pull_metric_tensor("disk_used_percentage")
n             = Nx.size(disk_used_pct)
t_days        = Nx.iota({n}, type: :f32)

design        = Nx.stack([Nx.broadcast(1.0, {n}), t_days], axis: 1)
{coeffs, _}   = Nx.LinAlg.lstsq(design, disk_used_pct)
[_a, pct_per_day] = Nx.to_flat_list(coeffs)

current_pct   = Nx.to_number(disk_used_pct[-1])
days_to_95 =
  if pct_per_day > 0, do: (95.0 - current_pct) / pct_per_day, else: :infinity
```

Same `lstsq` pattern as the memory leak — the meaning is just different
because the metric is different. Recipes don't change; the labels do.

## Turning a monotonic counter into a rate

`bytes_received_total` from `nerves_hub_link`'s health extension only
goes up (until the interface restarts). Math on the raw counter is
useless; first-differencing it gives bytes-per-sample, and dividing by
the inter-sample interval gives bytes/sec.

```elixir
# Monotonic counter, {n} s64, one sample per second
bytes_rx_total = pull_metric_tensor("eth0_bytes_received_total")

bytes_per_sec  = bytes_rx_total |> Nx.diff() |> Nx.as_type(:f32)

# Interfaces restart occasionally — that shows up as a negative diff.
# Mask those out before averaging so a counter reset doesn't drag the mean down.
rate_valid   = Nx.greater_equal(bytes_per_sec, 0)
clean_rate   = Nx.select(rate_valid, bytes_per_sec, 0.0)
mean_rx_rate =
  Nx.divide(Nx.sum(clean_rate), Nx.sum(Nx.as_type(rate_valid, :f32)))
```

`Nx.diff` is the trick on every monotonic counter — `vm.reductions`,
GC counts, packets in/out, energy use in kWh. After diffing it's a
gauge, and the rest of the gauge recipes apply.

## What does the temperature distribution look like?

A device that sits at 50°C all day is healthy; one that swings between
30°C and 80°C is being stressed. A histogram tells you which it is.

```elixir
cpu_temp_c = pull_metric_tensor("cpu_temperature")

# 1°C-wide buckets from 20°C to 90°C
bucket =
  cpu_temp_c
  |> Nx.subtract(20.0)
  |> Nx.floor()
  |> Nx.clip(0, 69)
  |> Nx.as_type(:u32)

temp_histogram = Nx.bincount(bucket, min_length: 70)
# temp_histogram[i] is the count of samples in [20+i, 21+i) °C
```

Same trick for *any* gauge — pick a low bound, pick a width, floor,
clip, `Nx.bincount`. The bucket vector is the histogram. To bin in 5°C
steps, divide by 5 instead of `floor`. To bin a rate, run the
counter→rate recipe first.

## Is the latest sample anomalous?

A live device emits one sample now; you have the last hour in memory.
Z-scoring the new sample against the window tells you how unusual it
is — a |z| > 3 is roughly "1-in-a-thousand" if the metric is
well-behaved.

```elixir
cpu_temp_window = pull_metric_tensor("cpu_temperature")  # last hour
latest_temp_c   = 71.4

mu      = Nx.mean(cpu_temp_window) |> Nx.to_number()
sigma   = Nx.standard_deviation(cpu_temp_window) |> Nx.to_number()
z_score = (latest_temp_c - mu) / sigma

anomalous? = abs(z_score) > 3
```

The same z-score is the basis for "alert me when the device behaves
out-of-character" on any metric you can reduce to a scalar.

## How loaded is the device, smoothed at three timescales?

System load is already a smoothed quantity, but you can produce your
own 1-/5-/15-minute rollups of *anything* the same way.

```elixir
# 1Hz cpu_usage_percent
cpu_usage_pct = pull_metric_tensor("cpu_usage_percent")

cpu_1m  = Nx.window_mean(cpu_usage_pct, 60)
cpu_5m  = Nx.window_mean(cpu_usage_pct, 300)
cpu_15m = Nx.window_mean(cpu_usage_pct, 900)
```

The three together tell you the *trend* of load over the hour:
`cpu_1m > cpu_5m > cpu_15m` says load is rising; the reverse says load
is falling.

## Null-aware reductions

If a metric has gaps, the validity tensor returned by `from_column/1`
is non-nil. Mask before reducing or the gaps skew the answer.

```elixir
{cabinet_temp_c, validity} = Tensor.from_column(temp_column)

# Mean of just the valid samples, ignoring nulls
valid_f          = Nx.as_type(validity, :f32)
sum_valid        = Nx.multiply(cabinet_temp_c, valid_f) |> Nx.sum()
n_valid          = Nx.sum(valid_f)
mean_cabinet_c   = Nx.divide(sum_valid, n_valid)
```

Validity is a `{n}` u8 tensor of `1` (present) / `0` (missing). The
mask threads through every reduction unchanged.
