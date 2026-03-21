# spark-gpu-throttle-check (enhanced)

A GPU throttle diagnostic tool for NVIDIA DGX Spark (GB10) and other NVIDIA systems. Detects clock throttling, identifies the cause, and tracks GPU health over time.

> **Fork of [hoesing/spark-gpu-throttle-check](https://github.com/hoesing/spark-gpu-throttle-check)**
> Original tool detects USB PD throttling. This fork adds NVML direct telemetry,
> throttle cause identification, PCIe link monitoring, stability analysis, and
> baseline drift detection.

## The Problem

DGX Spark systems can end up in a degraded power state where the GPU clock is capped far below normal — typically 500–850 MHz instead of ~2400 MHz. This causes significant performance degradation that's difficult to diagnose since the GPU otherwise appears healthy. Bad USB PD negotiation is a common cause, but thermal throttling, hardware slowdown, and power cap violations can produce similar symptoms.

The original tool answers: **"Is the GPU throttled?"**

This enhanced version answers: **"Is the GPU throttled, why, and is it getting worse?"**

## Why This Happens on DGX Spark

The DGX Spark uses USB Power Delivery (PD) negotiation with its power brick. When PD negotiation fails or enters a degraded state, the GPU remains in P0 but its clock speed is capped — typically 500–850 MHz instead of the expected ~2400 MHz. The GPU reports no errors and appears healthy to `nvidia-smi`, but performance drops by 60–80%.

Common triggers:
- Power brick disconnected/reconnected without a full discharge cycle
- Firmware updates that change PD negotiation behavior
- Faulty or marginal USB-C power cables
- Multiple power events (outages, surges) without a clean reset

The fix is usually a full power cycle: disconnect the power brick from both the wall and the Spark, wait 60 seconds, then reconnect. This forces a fresh PD negotiation.

This tool detects the condition and tells you whether the cause is power delivery, thermal throttling, or hardware-level slowdown — so you know whether to power cycle, check cooling, or contact NVIDIA.

## What's New (v2.0)

### NVML Direct Telemetry
Replaces `nvidia-smi` subprocess calls with direct NVML reads via ctypes. Faster sampling, richer data, no parsing overhead. Falls back to `nvidia-smi` if NVML library isn't found.

### Throttle Cause Identification
Decodes the NVML throttle reason bitmask into human-readable causes:

| Bitmask | Reason | What it means |
|---------|--------|---------------|
| `0x01` | `GPU_IDLE` | Normal idle state |
| `0x04` | `SW_POWER_CAP` | Software power limit hit |
| `0x08` | `HW_SLOWDOWN` | Hardware-enforced slowdown (power or thermal) |
| `0x20` | `SW_THERMAL_SLOWDOWN` | Driver thermal limit hit |
| `0x40` | `HW_THERMAL_SLOWDOWN` | Hardware thermal limit hit |
| `0x80` | `HW_POWER_BRAKE_SLOWDOWN` | Hardware power brake engaged |

The FAIL banner now tells you **why** clocks are low — power issue vs thermal issue vs hardware slowdown — instead of always suggesting USB PD.

### Clock Ramp-Up Timing
Measures how long the GPU takes to reach the threshold clock speed from idle. A healthy GPU ramps in under 1 second. Slow ramp-up can indicate power delivery issues.

### Clock Stability Score
Reports coefficient of variation (CV%) across all samples:
- **< 1% CV** — rock solid
- **1–5% CV** — minor variance
- **> 5% CV** — oscillating clocks, investigate

### Thermal Trajectory
Linear regression on temperature vs time during the test. Reports slope (°C/s), direction (rising/stable/cooling), and start→end temperatures. Rising temperature with dropping clocks = thermal throttle developing in real time.

### PCIe Link Monitoring
Captures PCIe link speed and width before and after the SGEMM load via `lspci`. Warns if the link degraded during the test — a signal of PCIe-level instability that can precede Xid 79 (GPU fell off bus) failures.

### Run-Length Display
Duplicate samples are collapsed into ranges instead of printing 20 identical rows:

```
       #  Clock    Max  PSt    Pwr   T°C  Fan%  Throttle
  ──────  ──────  ─────  ───  ─────  ────  ────  ────────
    1-20  1860   1911   P2  132.8    58     0  none (20x)
```

### Baseline / Compare
Save a healthy GPU snapshot and compare against it later:

```bash
# When GPU is healthy
python3 spark-gpu-throttle-check.py --save-baseline

# Later, when something seems wrong
python3 spark-gpu-throttle-check.py --compare
```

Shows delta for peak clock, average clock, power, temperature, and stability score — highlights regressions in color.

### JSON Report Export
Full machine-readable report with every sample, PCIe state, and all analysis:

```bash
python3 spark-gpu-throttle-check.py --report
```

Reports are saved to `~/.spark-throttle/reports/` with timestamps. Attach these to bug reports for complete diagnostic evidence.

### Multi-GPU Support
Test a specific GPU or all GPUs in the system:

```bash
# Test GPU 1
python3 spark-gpu-throttle-check.py --gpu 1

# Test every GPU
python3 spark-gpu-throttle-check.py --all-gpus
```

### Timeline Mode
High-frequency 100ms sampling to capture the clock ramp-up curve and catch transient throttle events:

```bash
python3 spark-gpu-throttle-check.py --timeline -n 50
```

## Usage

```bash
# Standard check (20 samples, 500ms intervals)
python3 spark-gpu-throttle-check.py

# Quick check with more samples
python3 spark-gpu-throttle-check.py -n 50

# Timeline capture
python3 spark-gpu-throttle-check.py --timeline -n 40

# Save baseline when healthy
python3 spark-gpu-throttle-check.py --save-baseline

# Compare against baseline
python3 spark-gpu-throttle-check.py --compare

# Full diagnostic with report
python3 spark-gpu-throttle-check.py --report --save-baseline

# Test all GPUs
python3 spark-gpu-throttle-check.py --all-gpus

# Quiet mode for scripting
python3 spark-gpu-throttle-check.py -q
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `-n`, `--samples` | 20 | Number of samples to collect |
| `-t`, `--threshold` | 1400 | Clock threshold (MHz) below which throttling is suspected |
| `-w`, `--warmup` | 2.0 | Warm-up time (seconds) before sampling begins |
| `-g`, `--gpu` | 0 | GPU index to test |
| `-q`, `--quiet` | — | Print only PASS/FAIL result line |
| `--timeline` | — | 100ms time-series mode |
| `--all-gpus` | — | Test every GPU in the system |
| `--save-baseline` | — | Save current results as baseline |
| `--compare` | — | Compare against saved baseline |
| `--report` | — | Export full JSON report |

### Exit Codes

- `0` — PASS, GPU clocks are healthy
- `1` — FAIL or WARNING, GPU appears throttled

## Sample Output

### Healthy System

```
============================================================
  Spark GPU Throttle Check — GPU 0 (enhanced)
============================================================

  GPU:     NVIDIA GeForce GTX 1080
  Driver:  535.274.02

  Idle:
    Clock:     582 / 1911 MHz
    P-state:   P8
    Power:     13.8 W
    Temp:      48 °C
    Throttle:  idle

  Collecting 20 samples (500ms), threshold 1400 MHz

         #  Clock    Max  PSt    Pwr   T°C  Fan%  Throttle
  ────────  ──────  ─────  ───  ─────  ────  ────  ────────────
      1-20  1860   1911   P2  132.8    58     0  none (20x)

────────────────────────────────────────────────────────────
  RESULTS
────────────────────────────────────────────────────────────
  Samples:         20
  Peak clock:      1860 MHz
  Average clock:   1860 MHz
  Avg power draw:  132.8 W
  Avg temperature: 58 °C
  Below threshold: 0% of samples < 1400 MHz
  Ramp-up time:    0.00s
  Clock stability: 0.00% CV (rock solid)
  Thermal trend:   rising (+0.36 °C/s) 56→60°C

  ┌────────────────────────────────────────────────────────┐
  │  PASS — GPU clocks look healthy under load.            │
  │  Peak: 1860 MHz, Avg: 1860 MHz                         │
  └────────────────────────────────────────────────────────┘
```

### Throttled System (USB PD Issue)

```
  ██████████████████████████████████████████████████████████
  █  FAIL — GPU IS THROTTLED                               █
  █  Clock never exceeded 1400 MHz under load.             █
  █  Cause: POWER — bad USB PD or PSU issue.               █
  █  Try: disconnect power, wait 60s, reconnect.           █
  ██████████████████████████████████████████████████████████
```

### Thermal Throttle

```
  ██████████████████████████████████████████████████████████
  █  FAIL — GPU IS THROTTLED                               █
  █  Clock never exceeded 1400 MHz under load.             █
  █  Cause: THERMAL — GPU overheating.                     █
  █  Check: fan speed, airflow, thermal paste.             █
  ██████████████████████████████████████████████████████████
```

### Baseline Comparison

```
────────────────────────────────────────────────────────────
  BASELINE COMPARISON
────────────────────────────────────────────────────────────
  Baseline from: 2026-03-21T11:10:03.975510+00:00

  Peak clock (MHz)        now: 1847  base: 1847  +0
  Avg clock (MHz)         now: 1845  base: 1841  +4
  Avg power (W)           now: 132.9  base: 131.2  +1.7
  Avg temp (°C)           now: 65  base: 66  -1
```

## Fix for PD Throttling

If the tool reports a FAIL with a power-related cause, try disconnecting the power brick from both the wall outlet and the Spark. Wait a minute, then reconnect. Run the check again to verify.

## Requirements

- Python 3.10+
- NVIDIA GPU driver with NVML, cuBLAS, and CUDA runtime libraries
- `nvidia-smi` in PATH (fallback only)
- `lspci` for PCIe monitoring (optional)

No pip packages required.

## Credits

- Original tool: [hoesing/spark-gpu-throttle-check](https://github.com/hoesing/spark-gpu-throttle-check)
- Enhanced by: [parallelArchitect](https://github.com/parallelArchitect) — human–AI collaborative engineering

## Related Tools

- [gpu-pcie-path-validator](https://github.com/parallelArchitect/gpu-pcie-path-validator) — sustained PCIe transport validation with replay counter monitoring
- [unified-memory-analyzer](https://github.com/parallelArchitect/cuda-unified-memory-analyzer) — CUDA Unified Memory fault and migration diagnostics
- [nvidia-gpu-val](https://github.com/parallelArchitect/nvidia-gpu-val) — gated GPU validation pipeline (PCIe → memory → compute → drift)

## License

MIT License
