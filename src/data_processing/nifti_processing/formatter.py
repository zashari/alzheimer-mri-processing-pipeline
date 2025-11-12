"""Output formatter for NIfTI processing stage."""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from rich.console import Console, Group
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.live import Live
from rich.text import Text


class NiftiFormatter:
    """Handles formatted output for NIfTI processing stage."""

    def __init__(self, verbose=False, quiet=False, json_only=False):
        self.verbose = verbose
        self.quiet = quiet
        self.json_only = json_only
        self.console = Console()
        self.report_data = {}
        self.has_warnings = False
        self.warnings_list = []
        self.start_time = time.time()

    def header(self, action: str, substage: str, **kwargs) -> None:
        """Print header section."""
        if self.json_only or self.quiet:
            return

        self.console.print("[cyan]═" * 70 + "[/cyan]")
        self.console.print(f"[cyan] NIfTI Processing • {substage.replace('_', ' ').title()} • {action.title()}[/cyan]")

        # Build context line
        timestamp = datetime.now().strftime("%H:%M:%S")
        info_parts = [f"Started: {timestamp}"]

        # Add action-specific context
        for key, value in kwargs.items():
            info_parts.append(f"{key.replace('_', ' ').title()}: {value}")

        info_line = "  •  ".join(info_parts)
        self.console.print(f" [dim]{info_line}[/dim]")
        self.console.print("[cyan]═" * 70 + "[/cyan]")
        self.console.print()

        # Initialize report
        self.report_data = {
            "tool": "nifti_processing",
            "substage": substage,
            "action": action,
            "started_at": datetime.now().isoformat(),
            "parameters": kwargs
        }

    def hd_bet_status(self, available: bool, gpu_info: Optional[Dict] = None) -> None:
        """Print HD-BET availability and GPU status."""
        if self.json_only:
            return

        self.console.print("[blue]🔍 HD-BET STATUS[/blue]")

        if available:
            self.console.print("  [green]✅[/green] HD-BET command-line tool found")
        else:
            self.console.print("  [red]❌[/red] HD-BET not found in PATH")
            self.console.print("  [yellow]💡[/yellow] Install with: pip install hd-bet")

        if gpu_info:
            self.console.print(f"  [green]🎮[/green] GPU: {gpu_info['name']}")
            self.console.print(f"  [green]💾[/green] VRAM: {gpu_info['total_gb']:.1f} GB")
            
            # Show GPU utilization if available (preferred), otherwise fall back to memory usage
            if gpu_info.get('utilization_gpu') is not None:
                self.console.print(
                    f"  [green]⚡[/green] GPU Utilization: {gpu_info['utilization_gpu']:.1f}%"
                )
                if gpu_info.get('memory_used_mb') is not None:
                    self.console.print(
                        f"  [green]📊[/green] Memory: {gpu_info['memory_used_mb']:.1f} MB / "
                        f"{gpu_info['total_mb']:.1f} MB ({gpu_info.get('memory_usage_percent', 0):.1f}%)"
                    )
            else:
                # Fallback to memory-based display if utilization not available
                self.console.print(
                    f"  [green]📊[/green] Memory: {gpu_info.get('used_mb', 0):.1f} MB / "
                    f"{gpu_info['total_mb']:.1f} MB ({gpu_info.get('usage_percent', 0):.1f}%)"
                )
        else:
            self.console.print("  [yellow]⚠️[/yellow] GPU not available, using CPU")

        self.console.print()
    
    def print_gpu_status_update(self, gpu_info: Optional[Dict] = None) -> None:
        """
        Print updated GPU status during processing (for periodic updates).
        
        Args:
            gpu_info: Updated GPU information dictionary
        """
        if self.json_only or self.quiet:
            return
        
        if not gpu_info:
            return
        
        # Print GPU utilization update on a new line
        if gpu_info.get('utilization_gpu') is not None:
            self.console.print(
                f"  [dim]⚡ GPU Utilization: {gpu_info['utilization_gpu']:.1f}%[/dim]",
                end=""
            )
            if gpu_info.get('memory_used_mb') is not None:
                self.console.print(
                    f"  [dim]📊 Memory: {gpu_info['memory_used_mb']:.1f} MB / "
                    f"{gpu_info['total_mb']:.1f} MB ({gpu_info.get('memory_usage_percent', 0):.1f}%)[/dim]"
                )
            else:
                self.console.print()  # New line if no memory info

    def configuration(self, device: str, profile: str, use_tta: bool,
                     batch_size: int, cleanup_enabled: bool) -> None:
        """Print configuration summary."""
        if self.json_only or self.quiet:
            return

        self.console.print("[blue]⚙️  CONFIGURATION[/blue]")
        self.console.print(f"  • Profile: {profile}")
        self.console.print(f"  • Device: {device.upper()}")
        self.console.print(f"  • Test-Time Augmentation: {'Enabled' if use_tta else 'Disabled'}")

        if cleanup_enabled:
            self.console.print(f"  • GPU Cleanup: Every {batch_size} subjects")
        else:
            self.console.print("  • GPU Cleanup: Disabled")

        self.console.print()

    def existing_progress(self, completed_files: int, completed_subjects: int,
                         breakdown: Dict[str, int]) -> None:
        """Print existing progress check results."""
        if self.json_only:
            return

        self.console.print("[blue]📁 EXISTING PROGRESS[/blue]")

        if completed_files > 0:
            self.console.print(
                f"  [green]✅[/green] Found {completed_files} completed files "
                f"from {completed_subjects} subjects"
            )
            for split, count in breakdown.items():
                if count > 0:
                    self.console.print(f"     • {split}: {count} files")
        else:
            self.console.print("  [dim]No existing files found[/dim]")

        self.console.print()

    def task_summary(self, total_tasks: int, to_process: int, to_skip: int) -> None:
        """Print task summary before processing."""
        if self.json_only:
            return

        self.console.print("[blue]📊 TASK SUMMARY[/blue]")
        self.console.print(f"  • Total tasks: {total_tasks}")
        self.console.print(f"  • To process: [green]{to_process}[/green]")
        self.console.print(f"  • To skip (existing): [dim]{to_skip}[/dim]")
        self.console.print()

        # Store in report
        self.report_data["task_summary"] = {
            "total_tasks": total_tasks,
            "to_process": to_process,
            "to_skip": to_skip
        }

    def create_progress_bar(self) -> Progress:
        """Create and return a progress bar."""
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=self.console
        )
    
    def create_live_display(
        self, 
        progress: Progress, 
        gpu_monitor=None,
        refresh_per_second: float = 2.0
    ) -> Live:
        """
        Create a Live display that combines Progress bar and GPU status.
        
        Args:
            progress: Rich Progress bar instance
            gpu_monitor: Optional GPUMonitor instance for real-time GPU updates
            refresh_per_second: Refresh rate for Live display (default: 2.0)
        
        Returns:
            Rich Live context manager
        """
        def render_gpu_status() -> Optional[Panel]:
            """Render GPU status panel if monitor is available."""
            if not gpu_monitor:
                return None
            
            gpu_info = gpu_monitor.get_current_info()
            if not gpu_info:
                return None
            
            # Build GPU status text
            lines = []
            if gpu_info.get('utilization_gpu') is not None:
                util_gpu = gpu_info['utilization_gpu']
                util_color = "green" if util_gpu > 50 else "yellow" if util_gpu > 10 else "dim"
                lines.append(
                    Text(f"⚡ GPU Utilization: ", style="blue") + 
                    Text(f"{util_gpu:.1f}%", style=util_color)
                )
            
            if gpu_info.get('memory_used_mb') is not None:
                mem_used = gpu_info['memory_used_mb']
                mem_total = gpu_info['total_mb']
                mem_percent = gpu_info.get('memory_usage_percent', 0)
                lines.append(
                    Text(f"📊 Memory: ", style="blue") +
                    Text(f"{mem_used:.0f} MB / {mem_total:.0f} MB ", style="cyan") +
                    Text(f"({mem_percent:.1f}%)", style="dim")
                )
            
            if not lines:
                return None
            
            return Panel(
                Group(*lines),
                title="[blue]GPU Status[/blue]",
                border_style="blue",
                padding=(0, 1)
            )
        
        def render() -> Group:
            """Render function for Live display."""
            renderables = [progress]
            
            gpu_panel = render_gpu_status()
            if gpu_panel:
                renderables.append(gpu_panel)
            
            return Group(*renderables)
        
        return Live(
            render(),
            refresh_per_second=refresh_per_second,
            console=self.console,
            screen=False
        )

    def gpu_cleanup(self, before_mb: float, after_mb: float, freed_mb: float) -> None:
        """Print GPU cleanup results."""
        if self.json_only or self.quiet:
            return

        self.console.print("\n[blue]🧹 GPU MEMORY CLEANUP[/blue]")
        self.console.print(f"  Before: {before_mb:.1f} MB")
        self.console.print(f"  After: {after_mb:.1f} MB")
        if freed_mb > 0:
            self.console.print(f"  [green]✅[/green] Freed {freed_mb:.1f} MB")
        self.console.print()

    def batch_results(self, batch_num: int, success: int, skipped: int,
                      failed: int, errors: List[str]) -> None:
        """Print batch processing results."""
        if self.json_only:
            return

        self.console.print(f"\n[blue]Batch {batch_num} Results:[/blue]")
        self.console.print(f"  [green]✅[/green] Success: {success}")

        if skipped > 0:
            self.console.print(f"  [dim]⏭️  Skipped: {skipped}[/dim]")

        if failed > 0:
            self.console.print(f"  [red]❌[/red] Failed: {failed}")
            if self.verbose and errors:
                for error in errors[:5]:  # Show first 5 errors
                    self.console.print(f"     • {error}")

    def test_results(self, results: List[Dict]) -> None:
        """Print test mode results."""
        if self.json_only:
            return

        self.console.print("[blue]🧪 TEST RESULTS[/blue]")

        table = Table(title=None, box=None)
        table.add_column("File", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Time (s)", justify="right")
        table.add_column("Output Files", style="dim")

        for result in results:
            status = "[green]✅ Success[/green]" if result["success"] else "[red]❌ Failed[/red]"
            time_str = f"{result.get('time', 0):.1f}" if result["success"] else "-"

            output_files = []
            if result.get("brain_file"):
                output_files.append("brain")
            if result.get("mask_file"):
                output_files.append("mask")
            output_str = ", ".join(output_files) if output_files else "-"

            table.add_row(
                result["input_file"],
                status,
                time_str,
                output_str
            )

        self.console.print(table)
        self.console.print()

    def final_summary(self, total: int, success: int, skipped: int,
                     failed: int, avg_time: Optional[float] = None,
                     errors: Optional[List[str]] = None) -> None:
        """Print final processing summary."""
        if self.json_only:
            return

        self.console.print("\n" + "=" * 70)
        self.console.print("[bold blue]📊 FINAL SUMMARY[/bold blue]")
        self.console.print("=" * 70)

        self.console.print(f"[green]✅[/green] Successful: {success}/{total}")

        if skipped > 0:
            self.console.print(f"[dim]⏭️  Skipped (existing): {skipped}[/dim]")

        if failed > 0:
            self.console.print(f"[red]❌[/red] Failed: {failed}/{total}")
            self.has_warnings = True

        if avg_time:
            self.console.print(f"⏱️  Average processing time: {avg_time:.1f} seconds")

        # Show sample errors if any
        if errors and len(errors) > 0 and not self.quiet:
            self.console.print("\n[yellow]⚠️  Sample errors:[/yellow]")
            for error in errors[:10]:
                self.console.print(f"  • {error}")
            if len(errors) > 10:
                self.console.print(f"  ... and {len(errors) - 10} more")

        # Store in report
        self.report_data["results"] = {
            "total": total,
            "success": success,
            "skipped": skipped,
            "failed": failed,
            "avg_time_seconds": avg_time,
            "errors": errors if self.verbose else errors[:10] if errors else []
        }

    def print(self, message: str) -> None:
        """Print a message."""
        if not self.json_only:
            self.console.print(message)

    def info(self, message: str) -> None:
        """Print info message."""
        if not self.json_only:
            self.console.print(f"[blue][INFO][/blue] {message}")

    def success(self, message: str) -> None:
        """Print success message."""
        if not self.json_only:
            self.console.print(f"[green]✅[/green] {message}")

    def warning(self, message: str) -> None:
        """Print warning and track it."""
        if not self.json_only:
            self.console.print(f"[yellow]⚠️[/yellow] {message}")
        self.has_warnings = True
        self.warnings_list.append(message)

    def error(self, message: str, details: Optional[Dict] = None) -> None:
        """Print error with optional details."""
        if not self.json_only:
            self.console.print(f"[red]❌[/red] {message}")

            if details and not self.quiet:
                if "cause" in details:
                    self.console.print(f"  Cause: {details['cause']}")
                if "next_steps" in details:
                    self.console.print(f"  Next steps: {details['next_steps']}")

    def next_steps(self, steps: List[str]) -> None:
        """Print next steps."""
        if self.json_only or self.quiet:
            return

        self.console.print("\n[blue]🔄 NEXT STEPS:[/blue]")
        for i, step in enumerate(steps, 1):
            self.console.print(f"  {i}. {step}")
        self.console.print()

    def footer(self, exit_code: int = 0) -> None:
        """Print footer and save report."""
        duration = time.time() - self.start_time

        # Complete report
        self.report_data.update({
            "finished_at": datetime.now().isoformat(),
            "duration_sec": round(duration, 1),
            "exit_code": exit_code,
            "has_warnings": self.has_warnings,
            "warnings": self.warnings_list,
            "report_version": "1.1.0"
        })

        # Save JSON report
        report_path = self._save_report()

        if not self.json_only and not self.quiet:
            if report_path:
                self.console.print(f"[blue][REPORT][/blue] Saved detailed JSON: {report_path}")

            # Determine outcome
            if exit_code != 0:
                outcome = "[red]FAIL[/red]"
            elif self.has_warnings:
                outcome = "[yellow]PASS with warnings[/yellow]"
            else:
                outcome = "[green]PASS[/green]"

            self.console.print(f"[dim]Finished in {duration:.1f}s • Outcome: {outcome}[/dim]")

    def _save_report(self) -> Optional[Path]:
        """Save JSON report to file."""
        try:
            reports_dir = Path(".reports")
            reports_dir.mkdir(exist_ok=True)

            timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
            substage = self.report_data.get("substage", "nifti")
            report_file = reports_dir / f"nifti_processing_{substage}-{timestamp}.json"

            with open(report_file, "w") as f:
                json.dump(self.report_data, f, indent=2)

            return report_file
        except Exception as e:
            if self.verbose:
                self.warning(f"Could not save report: {e}")
            return None