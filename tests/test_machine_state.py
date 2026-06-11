"""machine_state.snapshot: per-field None degradation, never raises (no GPU)."""
import os
import stat

from lce.machine_state import snapshot

KEYS = {"ac_online", "battery_status", "cpu_governor", "cpu_freq_mhz", "gpu"}


def _fake_sysfs(tmp_path, *, ac="1", bat="Discharging", gov="powersave",
                freqs=(2_400_000, 3_600_000)):
    """Build a fake /sys tree. Pass None for a part to omit it."""
    root = tmp_path / "sys"
    if ac is not None:
        d = root / "class/power_supply/ADP0"
        d.mkdir(parents=True)
        (d / "online").write_text(f"{ac}\n")
    if bat is not None:
        d = root / "class/power_supply/BAT0"
        d.mkdir(parents=True, exist_ok=True)
        (d / "status").write_text(f"{bat}\n")
    if gov is not None:
        d = root / "devices/system/cpu/cpu0/cpufreq"
        d.mkdir(parents=True)
        (d / "scaling_governor").write_text(f"{gov}\n")
    for i, khz in enumerate(freqs or ()):
        d = root / f"devices/system/cpu/cpu{i}/cpufreq"
        d.mkdir(parents=True, exist_ok=True)
        (d / "scaling_cur_freq").write_text(f"{khz}\n")
    return root


def _fake_nvidia_smi(tmp_path, body):
    script = tmp_path / "nvidia-smi"
    script.write_text(f"#!/bin/sh\n{body}\n")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return str(script)


def test_full_snapshot(tmp_path):
    root = _fake_sysfs(tmp_path)
    smi = _fake_nvidia_smi(tmp_path, 'echo "P0, 1695, 53, 59.12"')
    snap = snapshot(sysfs_root=root, nvidia_smi=smi)
    assert set(snap) == KEYS
    assert snap["ac_online"] is True
    assert snap["battery_status"] == "Discharging"
    assert snap["cpu_governor"] == "powersave"
    assert snap["cpu_freq_mhz"] == 3000.0  # mean of 2.4 and 3.6 GHz in MHz
    assert snap["gpu"] == {"pstate": "P0", "sm_mhz": 1695, "temp_c": 53,
                           "power_w": 59.12}


def test_on_battery(tmp_path):
    root = _fake_sysfs(tmp_path, ac="0")
    snap = snapshot(sysfs_root=root, nvidia_smi="missing-binary-xyz")
    assert snap["ac_online"] is False
    assert snap["gpu"] is None


def test_empty_sysfs_all_none(tmp_path):
    snap = snapshot(sysfs_root=tmp_path / "nothing-here",
                    nvidia_smi="missing-binary-xyz")
    assert snap == {"ac_online": None, "battery_status": None,
                    "cpu_governor": None, "cpu_freq_mhz": None, "gpu": None}


def test_garbage_contents_degrade_per_field(tmp_path):
    root = _fake_sysfs(tmp_path, ac="banana", freqs=())
    d = root / "devices/system/cpu/cpu0/cpufreq"
    (d / "scaling_cur_freq").write_text("not-a-number\n")
    snap = snapshot(sysfs_root=root, nvidia_smi="missing-binary-xyz")
    assert snap["ac_online"] is None       # unparseable -> None
    assert snap["cpu_freq_mhz"] is None    # unparseable -> None
    assert snap["cpu_governor"] == "powersave"  # other fields unaffected


def test_garbage_nvidia_smi_output(tmp_path):
    root = _fake_sysfs(tmp_path)
    smi = _fake_nvidia_smi(tmp_path, 'echo "what even is this"')
    snap = snapshot(sysfs_root=root, nvidia_smi=smi)
    assert snap["gpu"] is None
    assert snap["ac_online"] is True


def test_never_raises_with_unreadable_file(tmp_path):
    root = _fake_sysfs(tmp_path)
    target = root / "class/power_supply/ADP0/online"
    os.chmod(target, 0)
    try:
        snap = snapshot(sysfs_root=root, nvidia_smi="missing-binary-xyz")
    finally:
        os.chmod(target, 0o644)
    assert snap["ac_online"] is None
