"""Best-effort machine power/clock snapshot for benchmark forensics.

Phase-4 spec §4: latency numbers on this hardware swing up to ~3x with
power state (AC vs battery, boost vs sustained), so every benchmark
transaction records the state it ran under. Every field degrades to None
independently; the function never raises — measurement must never crash an
inference run (same principle as telemetry).
"""
from __future__ import annotations

import subprocess
from pathlib import Path


def snapshot(
    sysfs_root: str | Path = "/sys", nvidia_smi: str = "nvidia-smi"
) -> dict:
    """One point-in-time snapshot. Keys are stable; values may be None."""
    root = Path(sysfs_root)
    return {
        "ac_online": _ac_online(root),
        "battery_status": _battery_status(root),
        "cpu_governor": _read(
            root / "devices/system/cpu/cpu0/cpufreq/scaling_governor"
        ),
        "cpu_freq_mhz": _cpu_freq_mhz(root),
        "gpu": _gpu(nvidia_smi),
    }


def _read(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def _ac_online(root: Path) -> bool | None:
    for p in sorted(root.glob("class/power_supply/A*/online")):
        text = _read(p)
        if text in ("0", "1"):
            return text == "1"
    return None


def _battery_status(root: Path) -> str | None:
    for p in sorted(root.glob("class/power_supply/BAT*/status")):
        text = _read(p)
        if text:
            return text
    return None


def _cpu_freq_mhz(root: Path) -> float | None:
    freqs_khz: list[int] = []
    for p in root.glob("devices/system/cpu/cpu[0-9]*/cpufreq/scaling_cur_freq"):
        text = _read(p)
        try:
            freqs_khz.append(int(text))
        except (TypeError, ValueError):
            continue
    if not freqs_khz:
        return None
    return sum(freqs_khz) / len(freqs_khz) / 1000.0


def _gpu(nvidia_smi: str) -> dict | None:
    try:
        out = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=pstate,clocks.sm,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        pstate, sm, temp, power = (
            f.strip() for f in out.stdout.splitlines()[0].split(",")
        )
        return {
            "pstate": pstate,
            "sm_mhz": int(sm),
            "temp_c": int(temp),
            "power_w": float(power),
        }
    except Exception:
        return None
