"""HD-BET skull stripping processor."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..gpu_utils import kill_zombie_processes


class HDBETProcessor:
    """Handles HD-BET skull stripping operations."""

    def __init__(
        self,
        device: str = "cuda",
        use_tta: bool = False,
        timeout_sec: int = 600,
        temp_dir: Optional[Path] = None
    ):
        self.device = device
        self.use_tta = use_tta
        self.timeout_sec = timeout_sec
        self.temp_dir = temp_dir or Path(tempfile.gettempdir()) / "hd_bet_temp"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.dir_lock = threading.Lock()  # Thread-safe directory creation

    def check_availability(self) -> bool:
        """Check if HD-BET is installed and accessible."""
        try:
            # Create temp files for output
            temp_out = self.temp_dir / "hd_bet_test.txt"
            temp_err = self.temp_dir / "hd_bet_test_err.txt"

            with open(temp_out, "w") as fout, open(temp_err, "w") as ferr:
                result = subprocess.run(
                    ["hd-bet", "--help"],
                    stdout=fout,
                    stderr=ferr,
                    timeout=60,
                )

            # Clean up temp files
            temp_out.unlink(missing_ok=True)
            temp_err.unlink(missing_ok=True)

            return result.returncode == 0

        except FileNotFoundError:
            return False
        except subprocess.TimeoutExpired:
            # HD-BET command timed out but might still work
            return True
        except Exception:
            return False

    def process_file(
        self,
        input_path: Path,
        output_brain: Path,
        output_mask: Path,
        task_id: Optional[str] = None
    ) -> Tuple[str, Optional[str]]:
        """
        Process a single NIfTI file with HD-BET.

        Args:
            input_path: Path to input NIfTI file
            output_brain: Path for output brain file
            output_mask: Path for output mask file
            task_id: Optional task identifier for temp files

        Returns:
            Tuple of (status, error_message)
            Status can be: "success", "skip", "timeout", or error message
        """
        # Check if output already exists
        if output_brain.exists():
            return "skip", None

        # Ensure output directory exists (thread-safe)
        with self.dir_lock:
            output_brain.parent.mkdir(parents=True, exist_ok=True)

        # Create temporary files for subprocess output
        task_id = task_id or f"{time.time()}"
        temp_stdout = self.temp_dir / f"hd_bet_{task_id}.out"
        temp_stderr = self.temp_dir / f"hd_bet_{task_id}.err"

        try:
            # HD-BET requires output to end with .nii.gz
            # Use a temporary name that HD-BET expects
            temp_output = output_brain.parent / f"temp_{task_id}.nii.gz"

            # Build HD-BET command
            cmd = [
                "hd-bet",
                "-i", str(input_path),
                "-o", str(temp_output),
                "-device", self.device
            ]

            # Add optional flags
            if not self.use_tta:
                cmd.append("--disable_tta")

            # Save mask file
            cmd.append("--save_bet_mask")

            # Run HD-BET with output redirection
            with open(temp_stdout, "w") as fout, open(temp_stderr, "w") as ferr:
                # Create process with proper cleanup handling
                if os.name != "nt":  # Unix-like systems
                    process = subprocess.Popen(
                        cmd,
                        stdout=fout,
                        stderr=ferr,
                        close_fds=True,
                        preexec_fn=os.setsid
                    )
                else:  # Windows
                    process = subprocess.Popen(
                        cmd,
                        stdout=fout,
                        stderr=ferr,
                        close_fds=True
                    )

                try:
                    # Wait for completion with timeout
                    returncode = process.wait(timeout=self.timeout_sec)

                    if returncode != 0:
                        # Read error output for debugging
                        with open(temp_stderr, "r") as ferr:
                            error_msg = ferr.read()[:500]  # First 500 chars
                        return "error", f"HD-BET failed (code {returncode}): {error_msg}"

                except subprocess.TimeoutExpired:
                    # Kill the process on timeout
                    if os.name != "nt":
                        try:
                            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                            time.sleep(2)
                            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                        except:
                            process.kill()
                    else:
                        process.terminate()
                        time.sleep(2)
                        process.kill()

                    return "timeout", f"Timeout after {self.timeout_sec}s"

            # HD-BET creates files with specific naming
            hd_bet_brain = temp_output
            hd_bet_mask = Path(str(temp_output).replace(".nii.gz", "_bet.nii.gz"))

            # Move files to final locations
            if hd_bet_brain.exists():
                shutil.move(str(hd_bet_brain), str(output_brain))
            else:
                return "error", "Brain extraction file not created"

            if hd_bet_mask.exists():
                shutil.move(str(hd_bet_mask), str(output_mask))
            else:
                # Mask might not be created in some modes
                pass

            # Verify output
            if output_brain.exists():
                return "success", None
            else:
                return "error", "Output file not created"

        except Exception as e:
            return "error", str(e)

        finally:
            # Clean up temporary files
            temp_stdout.unlink(missing_ok=True)
            temp_stderr.unlink(missing_ok=True)
            # Clean up any remaining temp files
            for temp_file in output_brain.parent.glob(f"temp_{task_id}*"):
                temp_file.unlink(missing_ok=True)

    def process_batch(
        self,
        tasks: List[Tuple[Path, Path, Path, str]],
        progress_callback=None
    ) -> Dict[str, List]:
        """
        Process a batch of files.

        Args:
            tasks: List of (input_path, output_brain, output_mask, task_id) tuples
            progress_callback: Optional callback function for progress updates

        Returns:
            Dictionary with success, skipped, and failed lists
        """
        results = {
            "success": [],
            "skipped": [],
            "failed": [],
            "errors": []
        }

        for i, (input_path, output_brain, output_mask, task_id) in enumerate(tasks):
            if progress_callback:
                progress_callback(i, len(tasks), f"Processing {input_path.name}")

            status, error_msg = self.process_file(
                input_path, output_brain, output_mask, task_id
            )

            if status == "success":
                results["success"].append((input_path, output_brain, output_mask))
            elif status == "skip":
                results["skipped"].append((input_path, output_brain, output_mask))
            else:
                results["failed"].append((input_path, output_brain, output_mask))
                if error_msg:
                    results["errors"].append(f"{input_path.name}: {error_msg}")

        return results

    def cleanup(self) -> None:
        """Clean up temporary files and kill zombie processes."""
        # Kill any hanging HD-BET processes
        killed = kill_zombie_processes("hd-bet")
        if killed > 0:
            print(f"Killed {killed} hanging HD-BET processes")

        # Clean up temp directory
        if self.temp_dir.exists():
            for temp_file in self.temp_dir.glob("hd_bet_*"):
                try:
                    temp_file.unlink()
                except:
                    pass