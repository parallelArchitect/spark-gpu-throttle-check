#!/usr/bin/env python3
"""
spark-gpu-throttle-check.py — Enhanced GPU throttle diagnostic for DGX Spark
and other NVIDIA systems.

Original tool by hoesing: detect USB PD clock throttling via cuBLAS load.
Enhanced by parallelArchitect: NVML direct telemetry, throttle reason decoder,
PCIe link snapshot, baseline/compare, timeline, ramp analysis, stability score,
thermal trajectory, JSON report export.

Expected behavior:
  - Healthy PD:  graphics clock reaches ~2400 MHz under load
  - Bad PD:      graphics clock stays around ~850 MHz under load (P0 but capped)

Usage:
  python3 spark-gpu-throttle-check.py                    # standard check
  python3 spark-gpu-throttle-check.py --timeline         # time-series capture
  python3 spark-gpu-throttle-check.py --save-baseline    # save healthy snapshot
  python3 spark-gpu-throttle-check.py --compare          # diff against baseline
  python3 spark-gpu-throttle-check.py --report           # export JSON report
  python3 spark-gpu-throttle-check.py --all-gpus         # test every GPU
"""

import argparse
import ctypes
import ctypes.util
import json
import math
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

# ─── ANSI escape codes ───────────────────────────────────────────────────────

RED = "\033[31m"
BOLD_RED = "\033[1;31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"

# ─── Throttle reason bitmask decoder ─────────────────────────────────────────

THROTTLE_REASONS = {
    0x0000000000000001: "GPU_IDLE",
    0x0000000000000002: "APPLICATIONS_CLOCKS_SETTING",
    0x0000000000000004: "SW_POWER_CAP",
    0x0000000000000008: "HW_SLOWDOWN",
    0x0000000000000010: "SYNC_BOOST",
    0x0000000000000020: "SW_THERMAL_SLOWDOWN",
    0x0000000000000040: "HW_THERMAL_SLOWDOWN",
    0x0000000000000080: "HW_POWER_BRAKE_SLOWDOWN",
}

PROBLEM_REASONS = {
    0x0000000000000004,  # SW_POWER_CAP
    0x0000000000000008,  # HW_SLOWDOWN
    0x0000000000000020,  # SW_THERMAL_SLOWDOWN
    0x0000000000000040,  # HW_THERMAL_SLOWDOWN
    0x0000000000000080,  # HW_POWER_BRAKE_SLOWDOWN
}


def decode_throttle_bitmask(bitmask: int) -> list[str]:
    if bitmask == 0:
        return ["NONE"]
    reasons = []
    for bit, name in THROTTLE_REASONS.items():
        if bitmask & bit:
            reasons.append(name)
    return reasons if reasons else [f"UNKNOWN(0x{bitmask:016x})"]


def has_problem_throttle(bitmask: int) -> bool:
    return bool(bitmask & sum(PROBLEM_REASONS))


# ─── Progress bar ────────────────────────────────────────────────────────────

def progress_bar(current: int, total: int, width: int = 30, label: str = "") -> str:
    """Render an inline progress bar."""
    filled = int(width * current / total) if total > 0 else 0
    bar = "█" * filled + "░" * (width - filled)
    pct = 100 * current / total if total > 0 else 0
    return f"\r  {label}[{bar}] {pct:3.0f}%"


def progress_countdown(seconds: float, label: str = "Warming up"):
    """Show a countdown progress bar for warmup."""
    steps = max(int(seconds * 10), 1)
    step_time = seconds / steps
    for i in range(steps + 1):
        sys.stdout.write(progress_bar(i, steps, label=f"{label} "))
        sys.stdout.flush()
        if i < steps:
            time.sleep(step_time)
    sys.stdout.write("\r" + " " * 70 + "\r")
    sys.stdout.flush()


# ─── NVML direct interface via ctypes ────────────────────────────────────────

class NVMLDirect:
    """Lightweight NVML wrapper via ctypes. No pynvml dependency."""

    def __init__(self):
        self._lib = None
        self._handle = None
        self._available = False
        self._initialized = False

    def _load_lib(self) -> bool:
        """Load NVML shared library and call nvmlInit."""
        if self._initialized:
            return self._lib is not None
        self._initialized = True
        try:
            path = ctypes.util.find_library("nvidia-ml")
            if not path:
                for candidate in [
                    "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1",
                    "/usr/lib/aarch64-linux-gnu/libnvidia-ml.so.1",
                    "/usr/lib64/libnvidia-ml.so.1",
                    "/usr/lib/libnvidia-ml.so.1",
                ]:
                    if os.path.exists(candidate):
                        path = candidate
                        break
            if not path:
                return False
            self._lib = ctypes.CDLL(path)
            rc = self._lib.nvmlInit_v2()
            return rc == 0
        except (OSError, AttributeError):
            return False

    def init(self, gpu_index: int = 0) -> bool:
        if not self._load_lib():
            return False
        self._handle = ctypes.c_void_p()
        rc = self._lib.nvmlDeviceGetHandleByIndex_v2(
            ctypes.c_uint(gpu_index), ctypes.byref(self._handle)
        )
        if rc != 0:
            return False
        self._available = True
        return True

    def shutdown(self):
        if self._lib and self._initialized:
            try:
                self._lib.nvmlShutdown()
            except Exception:
                pass

    def get_device_count(self) -> int:
        if not self._load_lib():
            return 0
        val = ctypes.c_uint()
        rc = self._lib.nvmlDeviceGetCount_v2(ctypes.byref(val))
        return val.value if rc == 0 else 0

    @property
    def available(self) -> bool:
        return self._available

    def get_clock_mhz(self) -> int | None:
        if not self._available:
            return None
        val = ctypes.c_uint()
        rc = self._lib.nvmlDeviceGetClockInfo(self._handle, 0, ctypes.byref(val))
        return val.value if rc == 0 else None

    def get_max_clock_mhz(self) -> int | None:
        if not self._available:
            return None
        val = ctypes.c_uint()
        rc = self._lib.nvmlDeviceGetMaxClockInfo(self._handle, 0, ctypes.byref(val))
        return val.value if rc == 0 else None

    def get_power_w(self) -> float | None:
        if not self._available:
            return None
        val = ctypes.c_uint()
        rc = self._lib.nvmlDeviceGetPowerUsage(self._handle, ctypes.byref(val))
        return val.value / 1000.0 if rc == 0 else None

    def get_temperature(self) -> int | None:
        if not self._available:
            return None
        val = ctypes.c_uint()
        rc = self._lib.nvmlDeviceGetTemperature(self._handle, 0, ctypes.byref(val))
        return val.value if rc == 0 else None

    def get_pstate(self) -> str | None:
        if not self._available:
            return None
        val = ctypes.c_uint()
        rc = self._lib.nvmlDeviceGetPerformanceState(self._handle, ctypes.byref(val))
        return f"P{val.value}" if rc == 0 else None

    def get_throttle_reasons(self) -> int | None:
        if not self._available:
            return None
        val = ctypes.c_ulonglong()
        rc = self._lib.nvmlDeviceGetCurrentClocksThrottleReasons(
            self._handle, ctypes.byref(val)
        )
        return val.value if rc == 0 else None

    def get_fan_speed(self) -> int | None:
        if not self._available:
            return None
        val = ctypes.c_uint()
        rc = self._lib.nvmlDeviceGetFanSpeed(self._handle, ctypes.byref(val))
        return val.value if rc == 0 else None

    def get_gpu_name(self) -> str | None:
        if not self._available:
            return None
        buf = ctypes.create_string_buffer(256)
        rc = self._lib.nvmlDeviceGetName(self._handle, buf, 256)
        return buf.value.decode("utf-8", errors="replace") if rc == 0 else None

    def get_driver_version(self) -> str | None:
        if not self._available:
            return None
        buf = ctypes.create_string_buffer(256)
        rc = self._lib.nvmlSystemGetDriverVersion(buf, 256)
        return buf.value.decode("utf-8", errors="replace") if rc == 0 else None

    def get_pci_bus_id(self) -> str | None:
        if not self._available:
            return None

        class NvmlPciInfo(ctypes.Structure):
            _fields_ = [
                ("busIdLegacy", ctypes.c_char * 16),
                ("domain", ctypes.c_uint),
                ("bus", ctypes.c_uint),
                ("device", ctypes.c_uint),
                ("pciDeviceId", ctypes.c_uint),
                ("pciSubSystemId", ctypes.c_uint),
                ("busId", ctypes.c_char * 32),
            ]

        info = NvmlPciInfo()
        rc = self._lib.nvmlDeviceGetPciInfo_v3(self._handle, ctypes.byref(info))
        if rc == 0:
            bus_id = info.busId.decode("utf-8", errors="replace").strip()
            return bus_id if bus_id else info.busIdLegacy.decode("utf-8", errors="replace").strip()
        return None

    def sample(self) -> dict:
        throttle_raw = self.get_throttle_reasons()
        return {
            "timestamp": time.time(),
            "clk_mhz": self.get_clock_mhz(),
            "clk_max_mhz": self.get_max_clock_mhz(),
            "pstate": self.get_pstate(),
            "power_w": self.get_power_w(),
            "temp_c": self.get_temperature(),
            "fan_pct": self.get_fan_speed(),
            "throttle_raw": throttle_raw,
            "throttle_reasons": decode_throttle_bitmask(throttle_raw) if throttle_raw is not None else [],
            "throttle_problem": has_problem_throttle(throttle_raw) if throttle_raw is not None else False,
        }


# ─── PCIe link snapshot ─────────────────────────────────────────────────────

def get_pcie_link_state(bdf: str) -> dict | None:
    short_bdf = bdf.split(":")[-2] + ":" + bdf.split(":")[-1] if bdf.count(":") >= 2 else bdf
    try:
        result = subprocess.run(
            ["lspci", "-vvv", "-s", short_bdf],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return None
        state = {"bdf": bdf}
        for line in result.stdout.splitlines():
            line = line.strip()
            if line.startswith("LnkCap:"):
                for part in line.split(","):
                    part = part.strip()
                    if "Speed" in part and "GT/s" in part:
                        state["cap_speed"] = part.split("Speed")[-1].strip().rstrip(",")
                    if "Width" in part:
                        state["cap_width"] = part.split("Width")[-1].strip().rstrip(",")
            elif line.startswith("LnkSta:"):
                for part in line.split(","):
                    part = part.strip()
                    if "Speed" in part and "GT/s" in part:
                        speed_str = part.split("Speed")[-1].strip()
                        state["cur_speed"] = speed_str.split("(")[0].strip().rstrip(",")
                        state["downgraded"] = "downgraded" in part.lower()
                    if "Width" in part:
                        state["cur_width"] = part.split("Width")[-1].strip().rstrip(",")
        return state if "cur_speed" in state else None
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None


def format_pcie_state(state: dict | None, label: str = "") -> str:
    if state is None:
        return f"  PCIe {label}: unavailable"
    prefix = f"  PCIe {label}: " if label else "  PCIe: "
    cap = f"{state.get('cap_speed', '?')} {state.get('cap_width', '?')}"
    cur = f"{state.get('cur_speed', '?')} {state.get('cur_width', '?')}"
    downgraded = f" {YELLOW}(downgraded){RESET}" if state.get("downgraded") else ""
    return f"{prefix}Link {cur} (capable {cap}){downgraded}"


# ─── GPU load generator (from hoesing, unchanged) ───────────────────────────

def gpu_load(stop_event: threading.Event, load_ready: threading.Event, gpu_index: int = 0):
    """Sustained GPU load via cuBLAS sgemm through ctypes."""

    def find_lib(names):
        for name in names:
            path = ctypes.util.find_library(name)
            if path:
                return ctypes.CDLL(path)
            for prefix in ["/usr/lib/x86_64-linux-gnu", "/usr/lib/aarch64-linux-gnu",
                           "/usr/local/cuda/lib64", "/usr/lib64"]:
                for suffix in [".so", ".so.12", ".so.11"]:
                    try:
                        return ctypes.CDLL(f"{prefix}/lib{name}{suffix}")
                    except OSError:
                        continue
        return None

    cudart = find_lib(["cudart"])
    cublas = find_lib(["cublas"])
    if not cudart or not cublas:
        missing = []
        if not cudart:
            missing.append("libcudart")
        if not cublas:
            missing.append("libcublas")
        print(f"  ERROR: Cannot load {', '.join(missing)}")
        return

    c_size_t = ctypes.c_size_t
    c_int = ctypes.c_int
    c_float = ctypes.c_float
    c_void_p = ctypes.c_void_p

    cudart.cudaMalloc.argtypes = [ctypes.POINTER(c_void_p), c_size_t]
    cudart.cudaMalloc.restype = c_int
    cudart.cudaFree.argtypes = [c_void_p]
    cudart.cudaFree.restype = c_int
    cudart.cudaDeviceSynchronize.restype = c_int

    # Set device for multi-GPU
    if hasattr(cudart, 'cudaSetDevice'):
        cudart.cudaSetDevice.argtypes = [c_int]
        cudart.cudaSetDevice.restype = c_int
        cudart.cudaSetDevice(gpu_index)

    handle = c_void_p()
    cublas.cublasCreate_v2.argtypes = [ctypes.POINTER(c_void_p)]
    cublas.cublasCreate_v2.restype = c_int
    cublas.cublasDestroy_v2.argtypes = [c_void_p]
    cublas.cublasDestroy_v2.restype = c_int
    cublas.cublasSgemm_v2.argtypes = [
        c_void_p, c_int, c_int,
        c_int, c_int, c_int,
        ctypes.POINTER(c_float),
        c_void_p, c_int,
        c_void_p, c_int,
        ctypes.POINTER(c_float),
        c_void_p, c_int,
    ]
    cublas.cublasSgemm_v2.restype = c_int

    N = 4096
    nbytes = N * N * ctypes.sizeof(c_float)
    d_a, d_b, d_c = c_void_p(), c_void_p(), c_void_p()

    try:
        for ptr in [d_a, d_b, d_c]:
            rc = cudart.cudaMalloc(ctypes.byref(ptr), nbytes)
            if rc != 0:
                print(f"  ERROR: cudaMalloc failed (rc={rc})")
                return

        rc = cublas.cublasCreate_v2(ctypes.byref(handle))
        if rc != 0:
            print(f"  ERROR: cublasCreate failed (rc={rc})")
            return

        alpha = c_float(1.0)
        beta = c_float(0.0)
        CUBLAS_OP_N = 0

        load_ready.set()

        while not stop_event.is_set():
            cublas.cublasSgemm_v2(
                handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, N, N,
                ctypes.byref(alpha),
                d_a, N,
                d_b, N,
                ctypes.byref(beta),
                d_c, N,
            )
        cudart.cudaDeviceSynchronize()
    finally:
        cublas.cublasDestroy_v2(handle)
        for ptr in [d_a, d_b, d_c]:
            if ptr.value:
                cudart.cudaFree(ptr)


# ─── Formatting helpers ──────────────────────────────────────────────────────

def fmt(val, spec: str) -> str:
    if val is None:
        return "N/A"
    return f"{val:{spec}}"


def color_clock(clk, threshold: float) -> str:
    if clk is None:
        return "N/A"
    s = f"{clk:.0f}"
    return f"{RED}{s}{RESET}" if clk < threshold else f"{GREEN}{s}{RESET}"


def color_throttle(reasons: list[str]) -> str:
    if not reasons or reasons == ["NONE"]:
        return f"{DIM}none{RESET}"
    if reasons == ["GPU_IDLE"]:
        return f"{DIM}idle{RESET}"
    problem = [r for r in reasons if r not in ("GPU_IDLE", "APPLICATIONS_CLOCKS_SETTING", "SYNC_BOOST", "NONE")]
    if problem:
        return f"{RED}{','.join(problem)}{RESET}"
    return f"{YELLOW}{','.join(reasons)}{RESET}"


# ─── Run-length display ─────────────────────────────────────────────────────

def sample_signature(s: dict, threshold: float) -> tuple:
    """Create a comparison key for run-length grouping.
    Groups samples that have the same clock, pstate, and throttle state."""
    return (
        s.get("clk_mhz"),
        s.get("pstate"),
        tuple(sorted(s.get("throttle_reasons", []))),
    )


def print_run_length_table(samples: list[dict], threshold: float, timeline: bool):
    """Print samples with run-length encoding — collapse duplicate rows."""
    if not samples:
        return

    groups = []
    current_sig = None
    group_start = 0

    for i, s in enumerate(samples):
        sig = sample_signature(s, threshold)
        if sig != current_sig:
            if current_sig is not None:
                groups.append((group_start, i - 1, samples[group_start]))
            current_sig = sig
            group_start = i
    # Final group
    if current_sig is not None:
        groups.append((group_start, len(samples) - 1, samples[group_start]))

    if timeline:
        print(f"  {'t(s)':>8s}  {'Clock':>6s}  {'Max':>5s}  {'PSt':>3s}  {'Pwr':>5s}  {'T°C':>4s}  {'Fan%':>4s}  {'Throttle'}")
        print(f"  {'─'*8}  {'─'*6}  {'─'*5}  {'─'*3}  {'─'*5}  {'─'*4}  {'─'*4}  {'─'*20}")
    else:
        print(f"  {'#':>8s}  {'Clock':>6s}  {'Max':>5s}  {'PSt':>3s}  {'Pwr':>5s}  {'T°C':>4s}  {'Fan%':>4s}  {'Throttle'}")
        print(f"  {'─'*8}  {'─'*6}  {'─'*5}  {'─'*3}  {'─'*5}  {'─'*4}  {'─'*4}  {'─'*20}")

    for start, end, representative in groups:
        clk_str = color_clock(representative.get("clk_mhz"), threshold)
        throttle_str = color_throttle(representative.get("throttle_reasons", []))

        # Average power and temp across the group for display
        group_samples = samples[start:end + 1]
        avg_pwr = sum(s.get("power_w", 0) or 0 for s in group_samples) / len(group_samples)
        avg_tmp = sum(s.get("temp_c", 0) or 0 for s in group_samples) / len(group_samples)
        avg_fan = sum(s.get("fan_pct", 0) or 0 for s in group_samples) / len(group_samples)

        if timeline:
            t_start = representative.get("elapsed", 0)
            t_end = samples[end].get("elapsed", 0)
            if start == end:
                t_label = f"{t_start:7.1f}s"
            else:
                t_label = f"{t_start:.1f}-{t_end:.1f}s"
            print(
                f"  {t_label:>8s}  {clk_str:>6s}"
                f"  {fmt(representative.get('clk_max_mhz'), '.0f'):>5s}"
                f"  {(representative.get('pstate') or '?'):>3s}"
                f"  {avg_pwr:5.1f}"
                f"  {avg_tmp:4.0f}"
                f"  {avg_fan:4.0f}"
                f"  {throttle_str}"
            )
        else:
            if start == end:
                n_label = f"{start + 1}"
            else:
                n_label = f"{start + 1}-{end + 1}"
            count_note = f" {DIM}({end - start + 1}x){RESET}" if end > start else ""
            print(
                f"  {n_label:>8s}  {clk_str:>6s}"
                f"  {fmt(representative.get('clk_max_mhz'), '.0f'):>5s}"
                f"  {(representative.get('pstate') or '?'):>3s}"
                f"  {avg_pwr:5.1f}"
                f"  {avg_tmp:4.0f}"
                f"  {avg_fan:4.0f}"
                f"  {throttle_str}{count_note}"
            )


# ─── Analysis functions ──────────────────────────────────────────────────────

def compute_ramp_time(samples: list[dict], threshold: float) -> float | None:
    """Compute time from first sample to first sample at or above threshold.
    Returns seconds, or None if threshold was never reached."""
    if not samples:
        return None
    t0 = samples[0].get("timestamp", 0)
    for s in samples:
        clk = s.get("clk_mhz")
        if clk is not None and clk >= threshold:
            return s.get("timestamp", 0) - t0
    return None  # Never reached threshold


def compute_stability_score(clocks: list[float]) -> float:
    """Compute clock stability as coefficient of variation (lower = more stable).
    Returns 0.0 for perfectly stable, higher values for oscillating clocks."""
    if len(clocks) < 2:
        return 0.0
    mean = sum(clocks) / len(clocks)
    if mean == 0:
        return 0.0
    variance = sum((c - mean) ** 2 for c in clocks) / len(clocks)
    std = math.sqrt(variance)
    return (std / mean) * 100  # percentage


def compute_thermal_trajectory(samples: list[dict]) -> dict:
    """Compute temperature trend: slope (°C/s), direction, stability.
    Uses simple linear regression on temp vs elapsed time."""
    temps = [(s.get("elapsed", 0), s.get("temp_c")) for s in samples if s.get("temp_c") is not None]
    if len(temps) < 3:
        return {"slope": 0.0, "direction": "insufficient_data", "stable": True}

    n = len(temps)
    sum_x = sum(t[0] for t in temps)
    sum_y = sum(t[1] for t in temps)
    sum_xy = sum(t[0] * t[1] for t in temps)
    sum_x2 = sum(t[0] ** 2 for t in temps)

    denom = n * sum_x2 - sum_x ** 2
    if denom == 0:
        return {"slope": 0.0, "direction": "flat", "stable": True}

    slope = (n * sum_xy - sum_x * sum_y) / denom  # °C per second

    if slope > 0.1:
        direction = "rising"
    elif slope < -0.1:
        direction = "cooling"
    else:
        direction = "stable"

    return {
        "slope": round(slope, 3),
        "direction": direction,
        "stable": abs(slope) <= 0.1,
        "start_temp": temps[0][1],
        "end_temp": temps[-1][1],
    }


# ─── Baseline save/compare ───────────────────────────────────────────────────

BASELINE_DIR = Path.home() / ".spark-throttle"


def baseline_path(gpu_index: int = 0) -> Path:
    return BASELINE_DIR / f"baseline-gpu{gpu_index}.json"


def save_baseline(data: dict, gpu_index: int = 0):
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    data["saved_at"] = datetime.now(timezone.utc).isoformat()
    data["gpu_index"] = gpu_index
    path = baseline_path(gpu_index)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n  Baseline saved: {path}")


def load_baseline(gpu_index: int = 0) -> dict | None:
    path = baseline_path(gpu_index)
    if not path.exists():
        # Try legacy path
        legacy = Path.home() / ".spark-throttle-baseline.json"
        if legacy.exists():
            path = legacy
        else:
            return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def show_comparison(current: dict, baseline_data: dict):
    print()
    print("─" * 60)
    print("  BASELINE COMPARISON")
    print("─" * 60)
    print(f"  Baseline from: {baseline_data.get('saved_at', 'unknown')}")
    print()

    fields = [
        ("Peak clock (MHz)", "peak_clk", ".0f", "higher"),
        ("Avg clock (MHz)", "avg_clk", ".0f", "higher"),
        ("Avg power (W)", "avg_power", ".1f", "lower"),
        ("Avg temp (°C)", "avg_temp", ".0f", "lower"),
        ("Stability (%CV)", "stability_cv", ".2f", "lower"),
    ]

    for label, key, spec, better in fields:
        cur_val = current.get(key)
        base_val = baseline_data.get(key)
        if cur_val is not None and base_val is not None:
            delta = cur_val - base_val
            sign = "+" if delta >= 0 else ""
            if better == "higher":
                color = GREEN if delta >= 0 else RED
            else:
                color = GREEN if delta <= 0 else YELLOW
            print(f"  {label:<22s}  now: {cur_val:{spec}}  base: {base_val:{spec}}  {color}{sign}{delta:{spec}}{RESET}")

    # PCIe comparison
    cur_pcie = current.get("pcie_post")
    base_pcie = baseline_data.get("pcie_post")
    if cur_pcie and base_pcie:
        cur_speed = cur_pcie.get("cur_speed", "?")
        base_speed = base_pcie.get("cur_speed", "?")
        if cur_speed != base_speed:
            print(f"  {'PCIe link speed':<22s}  now: {cur_speed}  base: {base_speed}  {YELLOW}CHANGED{RESET}")

    cur_v = current.get("verdict", "?")
    base_v = baseline_data.get("verdict", "?")
    if cur_v != base_v:
        print(f"\n  {YELLOW}Verdict changed: {base_v} → {cur_v}{RESET}")
    print()


# ─── JSON report export ─────────────────────────────────────────────────────

def export_report(results: dict, samples: list[dict], gpu_index: int = 0):
    """Export a full JSON report with all samples and analysis."""
    report_dir = Path.home() / ".spark-throttle" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = report_dir / f"throttle-check_gpu{gpu_index}_{ts}.json"

    report = {
        "version": "2.0.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "gpu_index": gpu_index,
        "system": {
            "gpu_name": results.get("gpu_name"),
            "driver_version": results.get("driver_version"),
            "pci_bus_id": results.get("pci_bus_id"),
        },
        "config": {
            "threshold_mhz": results.get("threshold"),
            "num_samples": results.get("num_samples"),
        },
        "pcie": {
            "pre_load": results.get("pcie_pre"),
            "post_load": results.get("pcie_post"),
            "changed": results.get("pcie_changed"),
        },
        "analysis": {
            "verdict": results.get("verdict"),
            "peak_clk": results.get("peak_clk"),
            "avg_clk": results.get("avg_clk"),
            "avg_power": results.get("avg_power"),
            "avg_temp": results.get("avg_temp"),
            "pct_below": results.get("pct_below"),
            "stability_cv": results.get("stability_cv"),
            "ramp_time_s": results.get("ramp_time"),
            "thermal": results.get("thermal_trajectory"),
            "throttle_reasons_seen": results.get("throttle_reasons_seen"),
            "problem_throttle": results.get("problem_throttle"),
        },
        "samples": [
            {
                "n": s.get("sample_num"),
                "t": round(s.get("elapsed", 0), 3),
                "clk": s.get("clk_mhz"),
                "pwr": round(s.get("power_w", 0) or 0, 1),
                "temp": s.get("temp_c"),
                "pst": s.get("pstate"),
                "thr": s.get("throttle_raw"),
            }
            for s in samples
        ],
    }

    with open(filename, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved: {filename}")


# ─── nvidia-smi fallback ────────────────────────────────────────────────────

def _fallback_query_gpu() -> dict:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=clocks.current.graphics,clocks.max.graphics,"
                "pstate,power.draw,clocks_throttle_reasons.active",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return {}
        parts = [p.strip() for p in result.stdout.strip().split(",")]
        if len(parts) < 5:
            return {}

        def safe_float(s):
            try:
                return float(s)
            except ValueError:
                return None

        throttle_raw_str = parts[4] if len(parts) > 4 else "0x0"
        try:
            throttle_raw = int(throttle_raw_str, 16) if throttle_raw_str.startswith("0x") else 0
        except ValueError:
            throttle_raw = 0

        return {
            "timestamp": time.time(),
            "clk_mhz": safe_float(parts[0]),
            "clk_max_mhz": safe_float(parts[1]),
            "pstate": parts[2],
            "power_w": safe_float(parts[3]),
            "temp_c": None,
            "fan_pct": None,
            "throttle_raw": throttle_raw,
            "throttle_reasons": decode_throttle_bitmask(throttle_raw),
            "throttle_problem": has_problem_throttle(throttle_raw),
        }
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return {}


# ─── Main test runner ────────────────────────────────────────────────────────

def run_test(args, gpu_index: int = None) -> int:
    if gpu_index is None:
        gpu_index = args.gpu

    num_samples = args.samples
    threshold_mhz = args.threshold
    warmup = args.warmup
    quiet = args.quiet
    timeline = args.timeline
    interval = 0.1 if timeline else 0.5

    # ── Initialize NVML ──
    nvml = NVMLDirect()
    use_nvml = nvml.init(gpu_index=gpu_index)

    if not use_nvml:
        # Validate GPU index — don't silently fall back to GPU 0
        temp_nvml = NVMLDirect()
        if temp_nvml._load_lib():
            count = temp_nvml.get_device_count()
            temp_nvml.shutdown()
            if gpu_index >= count:
                print(f"ERROR: GPU {gpu_index} not found. System has {count} GPU(s) (index 0-{count-1}).")
                return 1
        if not quiet:
            print(f"  {YELLOW}NVML direct init failed — falling back to nvidia-smi{RESET}")

    if not quiet:
        print("=" * 60)
        title = f"  Spark GPU Throttle Check — GPU {gpu_index}"
        if use_nvml:
            title += " (enhanced)"
        print(title)
        print("=" * 60)
        print()

    # ── System info ──
    gpu_name = None
    driver_ver = None
    if use_nvml:
        gpu_name = nvml.get_gpu_name()
        driver_ver = nvml.get_driver_version()
        if not quiet:
            print(f"  GPU:     {gpu_name or 'unknown'}")
            print(f"  Driver:  {driver_ver or 'unknown'}")

    # ── PCIe pre-flight ──
    pci_bus_id = nvml.get_pci_bus_id() if use_nvml else None
    pcie_pre = get_pcie_link_state(pci_bus_id) if pci_bus_id else None

    if not quiet and pcie_pre:
        print(format_pcie_state(pcie_pre, "idle"))

    # ── Idle sample ──
    if use_nvml:
        idle = nvml.sample()
    else:
        idle = _fallback_query_gpu()

    if idle.get("clk_mhz") is None:
        print("ERROR: Cannot query GPU. Is the driver loaded?")
        nvml.shutdown()
        return 1

    if not quiet:
        print()
        print(f"  Idle:")
        print(f"    Clock:     {fmt(idle.get('clk_mhz'), '.0f')} / {fmt(idle.get('clk_max_mhz'), '.0f')} MHz")
        print(f"    P-state:   {idle.get('pstate', 'N/A')}")
        print(f"    Power:     {fmt(idle.get('power_w'), '.1f')} W")
        print(f"    Temp:      {fmt(idle.get('temp_c'), '.0f')} °C")
        print(f"    Throttle:  {color_throttle(idle.get('throttle_reasons', []))}")

    # ── Start load ──
    stop_event = threading.Event()
    load_ready = threading.Event()
    load_thread = threading.Thread(
        target=gpu_load, args=(stop_event, load_ready, gpu_index), daemon=True
    )
    load_thread.start()

    # ── Warmup with progress bar ──
    if not quiet and warmup > 0:
        print()
        progress_countdown(warmup, "Warming up")
    else:
        time.sleep(warmup)

    if not load_ready.wait(timeout=10):
        print("ERROR: GPU load failed to start.")
        stop_event.set()
        load_thread.join(timeout=5)
        nvml.shutdown()
        return 1

    # ── Collect samples with progress ──
    if not quiet:
        mode_label = "timeline (100ms)" if timeline else f"{num_samples} samples (500ms)"
        print(f"\n  Collecting {mode_label}, threshold {threshold_mhz:.0f} MHz\n")

    samples = []
    t_start = time.time()

    try:
        for i in range(1, num_samples + 1):
            if use_nvml:
                reading = nvml.sample()
            else:
                reading = _fallback_query_gpu()

            if reading and reading.get("clk_mhz") is not None:
                reading["sample_num"] = i
                reading["elapsed"] = time.time() - t_start
                samples.append(reading)

            # Progress bar on quiet mode or during collection
            if not quiet and not timeline:
                sys.stdout.write(progress_bar(i, num_samples, label="Sampling "))
                sys.stdout.flush()

            time.sleep(interval)

        if not quiet and not timeline:
            sys.stdout.write("\r" + " " * 70 + "\r")
            sys.stdout.flush()

    except KeyboardInterrupt:
        if not quiet:
            sys.stdout.write("\r" + " " * 70 + "\r")
            print("  (interrupted)")
    finally:
        stop_event.set()
        load_thread.join(timeout=5)

    # ── Print sample table (run-length encoded) ──
    if not quiet and samples:
        print()
        print_run_length_table(samples, threshold_mhz, timeline)

    # ── PCIe post-load ──
    pcie_post = get_pcie_link_state(pci_bus_id) if pci_bus_id else None

    if not quiet and pcie_post:
        print()
        print(format_pcie_state(pcie_post, "post-load"))

    pcie_changed = False
    if pcie_pre and pcie_post:
        if pcie_pre.get("cur_speed") != pcie_post.get("cur_speed"):
            pcie_changed = True
            if not quiet:
                print(f"  {YELLOW}PCIe link speed changed: {pcie_pre['cur_speed']} → {pcie_post['cur_speed']}{RESET}")
        if pcie_pre.get("cur_width") != pcie_post.get("cur_width"):
            pcie_changed = True
            if not quiet:
                print(f"  {YELLOW}PCIe link width changed: {pcie_pre['cur_width']} → {pcie_post['cur_width']}{RESET}")

    # ── Analysis ──
    if not samples:
        print("\nERROR: No samples collected.")
        nvml.shutdown()
        return 1

    clocks = [s["clk_mhz"] for s in samples if s["clk_mhz"] is not None]
    powers = [s["power_w"] for s in samples if s.get("power_w") is not None]
    temps = [s["temp_c"] for s in samples if s.get("temp_c") is not None]

    peak_clk = max(clocks)
    avg_clk = sum(clocks) / len(clocks)
    avg_pwr = sum(powers) / len(powers) if powers else 0
    avg_temp = sum(temps) / len(temps) if temps else None
    pct_below = sum(1 for c in clocks if c < threshold_mhz) / len(clocks) * 100

    # Ramp time
    ramp_time = compute_ramp_time(samples, threshold_mhz)

    # Stability score (CV%)
    stability_cv = compute_stability_score(clocks)

    # Thermal trajectory
    thermal = compute_thermal_trajectory(samples)

    # Throttle reasons
    all_throttle_reasons = set()
    problem_throttle_seen = False
    for s in samples:
        for r in s.get("throttle_reasons", []):
            if r not in ("NONE", "GPU_IDLE"):
                all_throttle_reasons.add(r)
        if s.get("throttle_problem"):
            problem_throttle_seen = True

    BOX_W = 56

    if not quiet:
        print()
        print("─" * 60)
        print("  RESULTS")
        print("─" * 60)
        print(f"  Samples:         {len(clocks)}")
        print(f"  Peak clock:      {peak_clk:.0f} MHz")
        print(f"  Average clock:   {avg_clk:.0f} MHz")
        print(f"  Avg power draw:  {avg_pwr:.1f} W")
        print(f"  Avg temperature: {fmt(avg_temp, '.0f')} °C")
        print(f"  Below threshold: {pct_below:.0f}% of samples < {threshold_mhz:.0f} MHz")

        # Ramp time
        if ramp_time is not None:
            color = GREEN if ramp_time < 1.0 else YELLOW if ramp_time < 3.0 else RED
            print(f"  Ramp-up time:    {color}{ramp_time:.2f}s{RESET}")
        else:
            print(f"  Ramp-up time:    {RED}never reached threshold{RESET}")

        # Stability
        color = GREEN if stability_cv < 1.0 else YELLOW if stability_cv < 5.0 else RED
        print(f"  Clock stability: {color}{stability_cv:.2f}% CV{RESET}", end="")
        if stability_cv < 1.0:
            print(f" {DIM}(rock solid){RESET}")
        elif stability_cv < 5.0:
            print(f" {DIM}(minor variance){RESET}")
        else:
            print(f" {DIM}(oscillating — investigate){RESET}")

        # Thermal trajectory
        t_color = GREEN if thermal["direction"] == "stable" else YELLOW if thermal["direction"] == "rising" else CYAN
        t_slope = f"{thermal['slope']:+.2f} °C/s"
        print(f"  Thermal trend:   {t_color}{thermal['direction']} ({t_slope}){RESET}", end="")
        if "start_temp" in thermal:
            print(f" {DIM}{thermal['start_temp']}→{thermal['end_temp']}°C{RESET}")
        else:
            print()

        if all_throttle_reasons:
            print(f"  Throttle seen:   {', '.join(sorted(all_throttle_reasons))}")
        if problem_throttle_seen:
            print(f"  {RED}Problem throttle detected during test{RESET}")
        if pcie_changed:
            print(f"  {YELLOW}PCIe link state changed during test{RESET}")
        print()

    # ── Verdict ──
    if peak_clk < threshold_mhz:
        verdict = "FAIL"
        exit_code = 1
        if quiet:
            print(f"FAIL gpu={gpu_index} peak={peak_clk:.0f}MHz avg={avg_clk:.0f}MHz threshold={threshold_mhz:.0f}MHz")
        else:
            cause_lines = ["FAIL — GPU IS THROTTLED"]
            cause_lines.append(f"Clock never exceeded {threshold_mhz:.0f} MHz under load.")
            if problem_throttle_seen:
                if any(r in all_throttle_reasons for r in ("SW_POWER_CAP", "HW_POWER_BRAKE_SLOWDOWN")):
                    cause_lines.append("Cause: POWER — bad USB PD or PSU issue.")
                    cause_lines.append("Try: disconnect power, wait 60s, reconnect.")
                elif any(r in all_throttle_reasons for r in ("HW_THERMAL_SLOWDOWN", "SW_THERMAL_SLOWDOWN")):
                    cause_lines.append("Cause: THERMAL — GPU overheating.")
                    cause_lines.append("Check: fan speed, airflow, thermal paste.")
                elif "HW_SLOWDOWN" in all_throttle_reasons:
                    cause_lines.append("Cause: HW_SLOWDOWN — power or thermal (HW-enforced).")
                    cause_lines.append("Try: power cycle, check PSU/cabling.")
                else:
                    cause_lines.append(f"Throttle: {', '.join(sorted(all_throttle_reasons))}")
            else:
                cause_lines.append("Likely cause: bad USB PD power negotiation.")
                cause_lines.append("Try: disconnect power brick, wait 60s, reconnect.")
            print(f"{BOLD_RED}  " + "█" * (BOX_W + 2))
            for line in cause_lines:
                print(f"  █  {line:<{BOX_W - 2}}█")
            print("  " + "█" * (BOX_W + 2) + RESET)
    elif pct_below > 50 or problem_throttle_seen:
        verdict = "WARNING"
        exit_code = 1
        if quiet:
            print(f"WARNING gpu={gpu_index} peak={peak_clk:.0f}MHz avg={avg_clk:.0f}MHz below={pct_below:.0f}%")
        else:
            warn_lines = [
                "WARNING — GPU clocks are intermittently low.",
                f"{pct_below:.0f}% of samples below {threshold_mhz:.0f} MHz.",
            ]
            if problem_throttle_seen:
                warn_lines.append(f"Throttle: {', '.join(sorted(all_throttle_reasons))}")
            if not thermal.get("stable"):
                warn_lines.append(f"Thermal: {thermal['direction']} ({thermal['slope']:+.2f} °C/s)")
            print("  ┌" + "─" * BOX_W + "┐")
            for line in warn_lines:
                print(f"  │  {line:<{BOX_W - 2}}│")
            print("  └" + "─" * BOX_W + "┘")
    else:
        verdict = "PASS"
        exit_code = 0
        if quiet:
            print(f"PASS gpu={gpu_index} peak={peak_clk:.0f}MHz avg={avg_clk:.0f}MHz")
        else:
            print("  ┌" + "─" * BOX_W + "┐")
            for line in [
                "PASS — GPU clocks look healthy under load.",
                f"Peak: {peak_clk:.0f} MHz, Avg: {avg_clk:.0f} MHz",
            ]:
                print(f"  │  {line:<{BOX_W - 2}}│")
            print("  └" + "─" * BOX_W + "┘")

    # ── Build results ──
    results = {
        "verdict": verdict,
        "peak_clk": peak_clk,
        "avg_clk": round(avg_clk, 1),
        "avg_power": round(avg_pwr, 1),
        "avg_temp": round(avg_temp, 1) if avg_temp is not None else None,
        "pct_below": round(pct_below, 1),
        "threshold": threshold_mhz,
        "num_samples": len(clocks),
        "ramp_time": round(ramp_time, 3) if ramp_time is not None else None,
        "stability_cv": round(stability_cv, 3),
        "thermal_trajectory": thermal,
        "throttle_reasons_seen": sorted(all_throttle_reasons),
        "problem_throttle": problem_throttle_seen,
        "pcie_pre": pcie_pre,
        "pcie_post": pcie_post,
        "pcie_changed": pcie_changed,
        "gpu_name": gpu_name,
        "driver_version": driver_ver,
        "pci_bus_id": pci_bus_id,
    }

    # ── Baseline / compare / report ──
    if args.save_baseline:
        save_baseline(results, gpu_index)

    if args.compare:
        bl = load_baseline(gpu_index)
        if bl:
            show_comparison(results, bl)
        else:
            print(f"\n  {YELLOW}No baseline found for GPU {gpu_index}. Run with --save-baseline first.{RESET}")

    if args.report:
        export_report(results, samples, gpu_index)

    nvml.shutdown()
    return exit_code


# ─── All-GPUs mode ──────────────────────────────────────────────────────────

def run_all_gpus(args) -> int:
    """Enumerate all GPUs and run the test on each."""
    nvml = NVMLDirect()
    if not nvml._load_lib():
        print("ERROR: Cannot load NVML to enumerate GPUs.")
        return 1

    count = nvml.get_device_count()
    nvml.shutdown()

    if count == 0:
        print("ERROR: No GPUs found.")
        return 1

    print(f"\n  {BOLD}Found {count} GPU(s) — testing each{RESET}\n")

    worst_exit = 0
    for i in range(count):
        if i > 0:
            print()
            print("─" * 60)
            print()
        exit_code = run_test(args, gpu_index=i)
        worst_exit = max(worst_exit, exit_code)

    if count > 1:
        print()
        print("=" * 60)
        if worst_exit == 0:
            print(f"  {GREEN}ALL {count} GPUs PASSED{RESET}")
        else:
            print(f"  {RED}ONE OR MORE GPUs FAILED — see above{RESET}")
        print("=" * 60)

    return worst_exit


# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="GPU throttle diagnostic — detect PD, thermal, and power throttling"
    )
    parser.add_argument("-n", "--samples", type=int, default=20,
                        help="Number of samples (default: 20)")
    parser.add_argument("-t", "--threshold", type=float, default=1400.0,
                        help="Clock threshold in MHz (default: 1400)")
    parser.add_argument("-w", "--warmup", type=float, default=2.0,
                        help="Warm-up time in seconds (default: 2.0)")
    parser.add_argument("-g", "--gpu", type=int, default=0,
                        help="GPU index (default: 0)")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="Print only PASS/FAIL result line")
    parser.add_argument("--timeline", action="store_true",
                        help="Time-series mode: 100ms intervals")
    parser.add_argument("--all-gpus", action="store_true",
                        help="Test every GPU in the system")
    parser.add_argument("--save-baseline", action="store_true",
                        help="Save results as baseline")
    parser.add_argument("--compare", action="store_true",
                        help="Compare against saved baseline")
    parser.add_argument("--report", action="store_true",
                        help="Export full JSON report with all samples")

    args = parser.parse_args()

    if args.samples < 1:
        parser.error("--samples must be at least 1")
    if args.threshold <= 0:
        parser.error("--threshold must be greater than 0")
    if args.warmup < 0:
        parser.error("--warmup cannot be negative")

    if args.all_gpus:
        sys.exit(run_all_gpus(args))
    else:
        sys.exit(run_test(args))


if __name__ == "__main__":
    main()
