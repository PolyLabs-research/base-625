#!/usr/bin/env python3
"""
collect_system_info.py

Collects an at-a-glance system profile (hardware, OS, toolchain) and stores it
as a timestamped summary in this project root. Use it to snapshot the current
development environment before diving into work.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from textwrap import indent
from typing import Iterable


ROOT = Path(__file__).resolve().parent
TIMESTAMP = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
SUMMARY_PATH = ROOT / f"system_summary_{TIMESTAMP}.txt"


def _capture(cmd: Iterable[str], timeout: int = 20) -> str:
    """
    Run a command and return its stdout/stderr combined.
    On failure (missing binary or non-zero exit), include diagnostic text.
    """
    try:
        completed = subprocess.run(
            list(cmd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError:
        return f"{cmd[0]} (not installed)"
    except subprocess.TimeoutExpired:
        return f"{' '.join(cmd)} (timed out after {timeout}s)"

    stdout = (completed.stdout or "").strip()
    stderr = (completed.stderr or "").strip()

    if completed.returncode != 0:
        combined = "\n".join(part for part in (stdout, stderr) if part)
        return f"(exit {completed.returncode})\n{combined}".strip()

    if stdout:
        return stdout
    if stderr:
        return stderr
    return "(no output)"


def _read_first(path: Path, needle: str) -> str | None:
    """
    Search for the first line in `path` that starts with `needle` and return its value.
    """
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                if line.startswith(needle):
                    return line.strip().split(":", 1)[-1].strip()
    except OSError:
        return None
    return None


def _meminfo(field: str) -> str | None:
    value = _read_first(Path("/proc/meminfo"), field)
    if value is None:
        return None
    return value


def _cpu_model() -> str | None:
    value = _read_first(Path("/proc/cpuinfo"), "model name")
    return value


def _format_pip_summary() -> str:
    """
    Return a short summary of the active interpreter and its key packages.
    """
    interpreter = sys.executable or "python"
    version = sys.version.splitlines()[0]
    pip_cmd = [interpreter, "-m", "pip", "list", "--format=json"]

    packages: list[dict[str, str]] = []
    try:
        output = _capture(pip_cmd)
        # If pip is missing, _capture already returns a string message.
        packages = json.loads(output) if output.startswith("[") else []
    except json.JSONDecodeError:
        packages = []

    summary_lines = [
        f"Interpreter: {interpreter}",
        f"Version: {version}",
    ]

    if packages:
        packages_sorted = sorted(packages, key=lambda item: item["name"].lower())
        top_packages = packages_sorted[:20]
        pkg_lines = [f"{pkg['name']}=={pkg['version']}" for pkg in top_packages]
        if len(packages_sorted) > len(top_packages):
            pkg_lines.append(f"... ({len(packages_sorted) - len(top_packages)} more)")
        summary_lines.append("Packages:")
        summary_lines.extend(indent("\n".join(pkg_lines), "  ").splitlines())
    else:
        summary_lines.append("Packages: (pip unavailable or no packages listed)")

    return "\n".join(summary_lines)


def _format_section(title: str, body: str) -> str:
    return f"{title}\n{'-' * len(title)}\n{body.rstrip()}\n"


def main() -> None:
    sections: list[str] = []

    # Core system facts
    uname = platform.uname()
    core_lines = [
        f"Timestamp: {_dt.datetime.now().isoformat()}",
        f"Hostname: {socket.gethostname()}",
        f"User: {os.getenv('USER') or os.getenv('USERNAME') or '(unknown)'}",
        f"OS: {platform.platform()}",
        f"Kernel: {uname.system} {uname.release} (build {uname.version})",
        f"Architecture: {uname.machine}",
        f"Python (this run): {sys.version.splitlines()[0]}",
        f"Working directory: {ROOT}",
        f"Virtual env: {os.getenv('VIRTUAL_ENV') or '(none)'}",
        f"PATH: {os.getenv('PATH', '(unset)')}",
    ]
    sections.append(_format_section("Core", "\n".join(core_lines)))

    # Extended OS details
    os_release = _capture(["bash", "-lc", "cat /etc/os-release"]) if shutil.which("bash") else _capture(["cat", "/etc/os-release"])
    uptime = _capture(["uptime", "-p"])
    boot_time = _capture(["uptime", "-s"])
    sections.append(
        _format_section(
            "OS Details",
            f"/etc/os-release:\n{indent(os_release, '  ')}\n\n"
            f"Uptime: {uptime}\n"
            f"Boot time: {boot_time}",
        )
    )

    # CPU / Memory
    cpu_lines = [
        f"Logical CPUs: {os.cpu_count()}",
        f"Model: {_cpu_model() or '(unknown)'}",
        f"CPU Info (lscpu):\n{indent(_capture(['lscpu']), '  ')}",
    ]
    mem_total = _meminfo("MemTotal")
    mem_available = _meminfo("MemAvailable")
    mem_lines = [
        f"MemTotal: {mem_total or '(unknown)'}",
        f"MemAvailable: {mem_available or '(unknown)'}",
        f"Memory (free -h):\n{indent(_capture(['free', '-h']), '  ')}",
    ]
    sections.append(
        _format_section(
            "CPU & Memory",
            "\n".join(cpu_lines + [""] + mem_lines),
        )
    )

    # GPU / Accelerators
    gpu_outputs = []
    gpu_outputs.append(("nvidia-smi", _capture(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"])))
    gpu_outputs.append(("lspci (VGA)", _capture(["bash", "-lc", "lspci | grep -i 'vga\\|3d\\|display'"]) if shutil.which("bash") else _capture(["lspci"])))
    sections.append(
        _format_section(
            "GPU & Accelerators",
            "\n".join(f"{label}:\n{indent(output, '  ')}" for label, output in gpu_outputs),
        )
    )

    # Storage
    disk_usage = shutil.disk_usage(ROOT)
    df_root = _capture(["df", "-h"])
    sections.append(
        _format_section(
            "Storage",
            "\n".join(
                [
                    f"Project root: {ROOT}",
                    f"Disk usage at root: total={disk_usage.total / 2**30:.2f} GiB, used={disk_usage.used / 2**30:.2f} GiB, free={disk_usage.free / 2**30:.2f} GiB",
                    f"df -h:\n{indent(df_root, '  ')}",
                ]
            ),
        )
    )

    # Toolchain versions
    tools = {
        "git": ["git", "--version"],
        "gcc": ["gcc", "--version"],
        "clang": ["clang", "--version"],
        "cmake": ["cmake", "--version"],
        "make": ["make", "--version"],
        "python3": ["python3", "--version"],
        "python": ["python", "--version"],
        "pip": ["pip", "--version"],
        "node": ["node", "--version"],
        "npm": ["npm", "--version"],
        "yarn": ["yarn", "--version"],
        "java": ["java", "--version"],
    }
    tool_lines = []
    for name, cmd in tools.items():
        output = _capture(cmd)
        first_line = output.splitlines()[0] if output and "\n" in output else output
        tool_lines.append(f"{name}: {first_line}")
    sections.append(_format_section("Toolchain Versions", "\n".join(tool_lines)))

    # Python packages (active interpreter)
    sections.append(_format_section("Python Environment", _format_pip_summary()))

    # Network configuration (local)
    ip_addr = _capture(["ip", "addr"])
    dns = _capture(["systemd-resolve", "--status"]) if shutil.which("systemd-resolve") else _capture(["cat", "/etc/resolv.conf"])
    sections.append(
        _format_section(
            "Networking",
            "\n".join(
                [
                    "IP configuration:\n" + indent(ip_addr, "  "),
                    "\nDNS:\n" + indent(dns, "  "),
                ]
            ),
        )
    )

    # Active services snapshot (systemctl fallback)
    services = "systemctl (not available)"
    if shutil.which("systemctl"):
        services = _capture(["systemctl", "list-units", "--type=service", "--state=running"])
    sections.append(
        _format_section(
            "Running Services",
            indent(services, "  "),
        )
    )

    # Final assembly
    summary = "\n".join(sections)
    SUMMARY_PATH.write_text(summary, encoding="utf-8")
    print(f"Wrote {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
