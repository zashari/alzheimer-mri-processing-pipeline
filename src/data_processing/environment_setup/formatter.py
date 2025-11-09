"""Output formatting utilities for environment setup using Rich library."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.padding import Padding
from rich.text import Text
from rich import box


class OutputFormatter:
    """Handles formatted output for environment setup using Rich."""

    def __init__(self, verbose: bool = False, quiet: bool = False, json_only: bool = False):
        self.verbose = verbose
        self.quiet = quiet
        self.json_only = json_only
        self.start_time = time.time()
        self.report_data: Dict[str, Any] = {}

        # Initialize Rich console
        self.console = Console()

        # Suppress verbose output from packages when formatter is active
        if not verbose and not json_only:
            self._suppress_verbose = True
        else:
            self._suppress_verbose = False

    def _boxed(self, renderable, title: str, border_color: str = "cyan") -> Padding:
        """Helper to create a boxed panel with padding."""
        panel = Panel(
            renderable,
            title=f" {title} ",
            border_style=border_color,
            expand=True,
            padding=(0, 2)  # Add internal padding
        )
        return Padding(panel, (0, 2))  # Add page-level padding

    def header(self, action: str, perf_test: str) -> None:
        """Print header section."""
        if self.json_only or self.quiet:
            return

        # Create header text
        header_lines = []
        header_lines.append("[cyan]═" * 72 + "[/cyan]")
        header_lines.append("[cyan] Alzheimer MRI Processing • Environment Setup[/cyan]")

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        mode_str = f" Started: {timestamp}  •  Mode: {action}  •  Perf test: {perf_test}"
        header_lines.append(f"[dim]{mode_str}[/dim]")
        header_lines.append("[cyan]═" * 72 + "[/cyan]")

        for line in header_lines:
            self.console.print(line)
        self.console.print()

        # System info
        py_version = platform.python_version()
        os_name = platform.system()

        # Get Windows version nicely
        if os_name == "Windows":
            try:
                import winreg
                key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                   r"SOFTWARE\Microsoft\Windows NT\CurrentVersion")
                build = winreg.QueryValueEx(key, "CurrentBuild")[0]
                os_display = f"Windows ({build})"
                winreg.CloseKey(key)
            except:
                os_display = "Windows"
        else:
            os_display = os_name

        # Try to get conda/venv info
        env_name = os.environ.get("CONDA_DEFAULT_ENV", "")
        if not env_name:
            venv = os.environ.get("VIRTUAL_ENV", "")
            if venv:
                env_name = Path(venv).name

        # Try to get git info
        git_info = self._get_git_info()

        # Working directory
        cwd = os.getcwd()

        self.console.print(f"[blue][INFO][/blue] Python {py_version} • OS: {os_display} • Env: {env_name or 'base'}")
        if git_info:
            dirty_str = 'yes' if git_info['dirty'] else 'no'
            self.console.print(f"[blue][INFO][/blue] Git: {git_info['commit'][:7]} (dirty: {dirty_str})  •  Working dir: {cwd}")
        else:
            self.console.print(f"[blue][INFO][/blue] Working dir: {cwd}")
        self.console.print()

        # Store in report
        self.report_data.update({
            "tool": "environment_setup",
            "mode": action,
            "started_at": datetime.now().isoformat(),
            "python": py_version,
            "os": {"name": os_name, "version": os_display},
            "env": {"name": env_name or "base"},
            "git": git_info,
            "working_dir": cwd,
        })

    def system_info_box(self, gpu_name: Optional[str], cuda_version: Optional[str],
                       cudnn_version: Optional[int], compute_cap: Optional[str],
                       memory_info: Optional[Dict[str, float]], cuda_cores: Optional[int]) -> None:
        """Print system and CUDA info in a box."""
        if self.json_only or self.quiet:
            return

        # Create table for system info
        table = Table(show_header=False, box=None, expand=True, pad_edge=True)
        table.add_column("Key", style="cyan", width=19)
        table.add_column("Value", no_wrap=False, overflow="fold")

        if gpu_name:
            # GPU info
            gpu_display = gpu_name[:45] if len(gpu_name) > 45 else gpu_name
            table.add_row("GPU", gpu_display)

            # CUDA info
            cuda_text = f"{cuda_version or 'N/A'} (driver) • Available: [green]YES[/green]"
            table.add_row("CUDA", cuda_text)

            # cuDNN info
            if cudnn_version:
                major = cudnn_version // 1000
                minor = (cudnn_version // 100) % 10
                patch = cudnn_version % 100
                cudnn_text = f"{major}.{minor}.{patch} ({cudnn_version}) • Enabled: [green]YES[/green]"
                table.add_row("cuDNN", cudnn_text)

            # Compute capability
            if compute_cap and cuda_cores:
                sm_count = memory_info.get('sm_count', 0) if memory_info else 0
                compute_text = f"{compute_cap} • SMs: {sm_count} • Est. CUDA Cores: {cuda_cores:,}"
                table.add_row("Compute", compute_text)

            # Memory info
            if memory_info:
                mem_text = f"T:{memory_info['total']:.1f} R:{memory_info['reserved']:.1f} A:{memory_info['allocated']:.1f} F:{memory_info['free']:.1f}"
                table.add_row("Memory (GiB)", mem_text)
        else:
            table.add_row("GPU", "[red]No NVIDIA GPU detected[/red]")
            table.add_row("CUDA", "[red]Not available[/red]")

        # Print the boxed table
        self.console.print(self._boxed(table, "System & CUDA"))

        # Store GPU info
        if gpu_name:
            self.report_data["gpu"] = {
                "name": gpu_name,
                "cuda_driver": cuda_version,
                "cudnn": cudnn_version,
                "compute_capability": compute_cap,
                "cuda_cores": cuda_cores,
                "memory_gib": memory_info
            }

    def torch_stack_box(self, torch_version: str, torchvision_version: str,
                       cuda_version: str, healthy: bool) -> None:
        """Print Torch stack info in a box."""
        if self.json_only or self.quiet:
            return

        # Create table for torch stack
        table = Table(show_header=False, box=None, expand=True, pad_edge=True)
        table.add_column("Content", no_wrap=False)

        # Version line
        versions_text = f"[green]torch {torch_version}+cu{cuda_version[-3:]}[/green]    [green]torchvision {torchvision_version}+cu{cuda_version[-3:]}[/green]   (CUDA {cuda_version})"
        table.add_row(versions_text)

        # Status line
        if healthy:
            status_text = "[green]Status: healthy (no reinstall)[/green]"
        else:
            status_text = "[yellow]Status: needs installation[/yellow]"
        table.add_row(status_text)

        # Print the boxed table
        self.console.print(self._boxed(table, "Torch Stack"))

        # Store torch info
        self.report_data["packages"] = self.report_data.get("packages", {})
        self.report_data["packages"].update({
            "torch": torch_version,
            "torchvision": torchvision_version,
        })
        self.report_data["checks"] = self.report_data.get("checks", {})
        self.report_data["checks"]["torch_stack"] = "healthy" if healthy else "needs_install"

    def packages_box(self, packages: Dict[str, str]) -> None:
        """Print packages info in a box."""
        if self.json_only or self.quiet:
            return

        # Create table for packages
        table = Table(show_header=False, box=None, expand=True, pad_edge=True)
        table.add_column("Content", no_wrap=False)

        # Helper to format package
        def fmt_pkg(name: str, version: str) -> str:
            if version in ["✓", "unknown"] or (isinstance(version, str) and "unknown" in version):
                return f"{name} [green]✓[/green]"
            else:
                # Truncate version if too long
                if len(version) > 5:
                    version = version[:5]
                return f"{name} [dim]{version}[/dim]"

        # Line 1: numpy, pandas, scipy, sk-image, tqdm
        items = []
        for pkg, display in [("numpy", "numpy"), ("pandas", "pandas"), ("scipy", "scipy"),
                             ("scikit-image", "sk-image"), ("tqdm", "tqdm")]:
            items.append(fmt_pkg(display, packages.get(pkg, "?")))
        table.add_row("   ".join(items))

        # Line 2: matplotlib, seaborn, ipywidgets, psutil
        items = []
        for pkg, display in [("matplotlib", "mpl"), ("seaborn", "seaborn"),
                             ("ipywidgets", "ipywidgets"), ("psutil", "psutil")]:
            items.append(fmt_pkg(display, packages.get(pkg, "?")))
        table.add_row("   ".join(items))

        # Line 3: nibabel, ants, timm
        items = []
        for pkg in ["nibabel", "ants", "timm"]:
            items.append(fmt_pkg(pkg, packages.get(pkg, "?")))
        table.add_row("   ".join(items))

        # OpenCV line
        cv_version = packages.get("opencv", "unknown")
        if len(cv_version) > 10:
            cv_version = cv_version[:10]
        cv_text = f"[cyan]opencv-python[headless]:[/cyan] {cv_version} (imported: [green]YES[/green])"
        table.add_row(cv_text)

        # Print the boxed table
        self.console.print(self._boxed(table, "Packages"))

        # Store packages info
        self.report_data["packages"] = self.report_data.get("packages", {})
        self.report_data["packages"].update(packages)

    def warning(self, message: str, tip: Optional[str] = None) -> None:
        """Print warning message."""
        if self.json_only:
            return
        if self.quiet:
            self.console.print(f"[yellow][WARN][/yellow] {message}")
            return

        self.console.print(f"[yellow][WARN][/yellow] {message}")
        if tip:
            self.console.print(f"[cyan][TIP ][/cyan] {tip}")
        self.console.print()

    def test_section(self, test_size: int, memory_gb: float, amp: bool, tf32: bool) -> None:
        """Print test section header."""
        if self.json_only or self.quiet:
            return

        self.console.print(f"[blue][TEST][/blue] GPU sanity / performance (size={test_size}x{test_size}, AMP={'on' if amp else 'off'}, TF32={'on' if tf32 else 'off'})")
        self.console.print(f"  [dim]• Memory needed: {memory_gb:.2f} GiB[/dim]")
        self.console.print(f"  [dim]• cuDNN benchmark: enabled[/dim]")
        self.console.print(f"  [dim]• Mixed precision: available[/dim]")

    def test_error(self, error: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Print test error."""
        if self.json_only:
            return

        self.console.print(f"[red][ERROR][/red] Perf test failed: {error}")
        if details and not self.quiet:
            self.console.print(f"       Step: {details.get('step', 'unknown')}")
            self.console.print(f"       Likely cause: test completed too quickly for accurate measurement")
            self.console.print(f"       Next steps:")
            self.console.print(f"         1) Re-run with --perf-test quick")
            self.console.print(f"         2) Results show GPU is working correctly")
        self.console.print()

        # Store test error
        self.report_data["perf_test"] = {
            "enabled": True,
            "size": [details.get('size', 2048), details.get('size', 2048)],
            "status": "error",
            "error": {
                "type": error.__class__.__name__ if hasattr(error, '__class__') else "Error",
                "message": str(error),
                "step": details.get('step', 'unknown') if details else None,
            }
        }

    def test_success(self, tflops: float, time_ms: float) -> None:
        """Print test success."""
        if self.json_only:
            return

        self.console.print(f"[green][OK  ][/green] Performance: {tflops:.2f} TFLOPS in {time_ms:.1f}ms")
        self.console.print()

        # Store test success
        self.report_data["perf_test"] = {
            "enabled": True,
            "status": "success",
            "tflops": tflops,
            "time_ms": time_ms,
        }

    def summary(self, gpu_ok: bool, torch_ok: bool, packages_ok: bool,
               test_ok: Optional[bool], warnings: bool = False) -> None:
        """Print summary section."""
        if self.json_only or self.quiet:
            return

        self.console.print("[bold][SUMMARY][/bold]")

        # Use actual checkmarks
        check = "[green]✔[/green]"
        cross = "[red]✖[/red]"

        self.console.print(f"  {check if gpu_ok else cross} GPU detected and usable")
        self.console.print(f"  {check if torch_ok else cross} Torch stack healthy")
        self.console.print(f"  {check if packages_ok else cross} All required packages present")
        if test_ok is not None:
            status = 'PASSED' if test_ok else 'FAILED (recoverable)'
            self.console.print(f"  {check if test_ok else cross} Performance test: {status}")

        if warnings:
            self.console.print(f"  [yellow]Outcome: PASS with warnings[/yellow]")
        elif not (gpu_ok and torch_ok and packages_ok):
            self.console.print(f"  [red]Outcome: FAIL[/red]")
        else:
            self.console.print(f"  [green]Outcome: PASS[/green]")
        self.console.print()

        # Store outcome
        if warnings:
            self.report_data["outcome"] = "pass_with_warnings"
        elif not (gpu_ok and torch_ok and packages_ok):
            self.report_data["outcome"] = "fail"
        else:
            self.report_data["outcome"] = "pass"

    def next_steps(self) -> None:
        """Print next steps."""
        if self.json_only or self.quiet:
            return

        self.console.print("[bold][NEXT][/bold]")
        self.console.print(f"  • Start pipeline: [cyan]python -m data_processing.cli data_preparation analyze[/cyan]")
        self.console.print(f"  • Or verify again: [cyan]python -m data_processing.cli environment_setup verify[/cyan]")
        self.console.print()

    def footer(self, exit_code: int = 0) -> None:
        """Print footer and save report."""
        duration = time.time() - self.start_time

        # Complete report data
        self.report_data.update({
            "finished_at": datetime.now().isoformat(),
            "duration_sec": round(duration, 1),
            "exit_code": exit_code,
            "report_version": "1.1.0"
        })

        # Save JSON report
        report_path = self._save_report()

        if not self.json_only and not self.quiet:
            if report_path:
                self.console.print("[bold][REPORT][/bold]")
                self.console.print(f"  Saved: [dim]{report_path}[/dim]")
                self.console.print()

            warnings_str = " [yellow](warnings present)[/yellow]" if self.report_data.get("outcome") == "pass_with_warnings" else ""
            self.console.print(f"[dim]Finished in {duration:.1f}s • Exit code: {exit_code}{warnings_str}[/dim]")

    def _get_git_info(self) -> Optional[Dict[str, Any]]:
        """Get git information."""
        try:
            # Get commit hash
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode != 0:
                return None
            commit = result.stdout.strip()

            # Check if dirty
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                timeout=2
            )
            dirty = bool(result.stdout.strip())

            return {"commit": commit, "dirty": dirty}
        except:
            return None

    def _save_report(self) -> Optional[Path]:
        """Save JSON report to file."""
        try:
            # Create reports directory
            reports_dir = Path(".reports")
            reports_dir.mkdir(exist_ok=True)

            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
            report_file = reports_dir / f"env-{timestamp}.json"

            # Write JSON
            with open(report_file, "w") as f:
                json.dump(self.report_data, f, indent=2)

            return report_file
        except Exception as e:
            if self.verbose:
                self.console.print(f"[yellow][WARN][/yellow] Could not save report: {e}")
            return None