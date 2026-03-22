# Changelog

All notable changes to this project will be documented in this file.

## [2.1.0] — 2026-03-22

### Added
- GPU utilization monitoring via `nvmlDeviceGetUtilizationRates`
- Util% column in sample table shows real-time GPU saturation
- Load adequacy gate: if avg GPU util <80% during test, verdict is INCONCLUSIVE instead of FAIL — prevents false throttle diagnosis on insufficient load
- Utilization included in JSON report (per-sample and analysis)

## [2.0.0] — 2026-03-21

### Added
- NVML direct telemetry via ctypes (replaces nvidia-smi subprocess)
- Throttle reason bitmask decoder (SW_POWER_CAP, HW_SLOWDOWN, HW_THERMAL_SLOWDOWN, etc.)
- PCIe link state snapshot before and after SGEMM load
- Temperature and fan speed monitoring per sample
- Clock ramp-up time measurement
- Clock stability score (coefficient of variation %)
- Thermal trajectory analysis (slope, direction, start→end)
- `--timeline` mode (100ms time-series capture)
- `--save-baseline` / `--compare` for drift detection
- `--report` for full JSON export with all samples
- `--all-gpus` to enumerate and test every GPU
- `--gpu N` to target specific GPU index
- Run-length display (duplicate samples collapsed into ranges)
- Progress bars for warmup and sample collection
- FAIL banner identifies cause: power vs thermal vs HW slowdown
- Per-GPU baseline storage (~/.spark-throttle/)
- nvidia-smi fallback for systems without NVML lib

### Fixed
- `--gpu 99` now errors cleanly instead of silent fallback to GPU 0
- Avg temperature shows N/A instead of 0 when no temp data available

## [1.0.0] — 2026-03-08

### Original (hoesing)
- cuBLAS SGEMM load via ctypes
- Clock threshold pass/fail against 1400 MHz
- nvidia-smi subprocess for telemetry
- Basic PASS/FAIL/WARNING verdicts
