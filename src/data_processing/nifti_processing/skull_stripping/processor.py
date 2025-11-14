"""HD-BET skull stripping processor."""

from __future__ import annotations

import shutil
import subprocess
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
        # temp_dir kept for compatibility but not used with subprocess.run()
        self.dir_lock = threading.Lock()  # Thread-safe directory creation

    def check_availability(self) -> bool:
        """Check if HD-BET is installed and accessible."""
        print("[DEBUG] Starting HD-BET availability check...")
        try:
            # Use PIPE to avoid file locking issues
            print("[DEBUG] Running: hd-bet --help")
            result = subprocess.run(
                ["hd-bet", "--help"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=60,
            )
            print(f"[DEBUG] HD-BET check completed with return code: {result.returncode}")

            return result.returncode == 0

        except FileNotFoundError:
            print("[DEBUG] HD-BET not found in PATH")
            return False
        except subprocess.TimeoutExpired:
            # HD-BET command timed out but might still work
            print("[DEBUG] HD-BET check timed out after 60s")
            return True
        except Exception as e:
            print(f"[DEBUG] HD-BET check failed with error: {e}")
            return False

    def process_file(
        self,
        input_path: Path,
        output_brain: Path,
        output_mask: Path,
        task_id: Optional[str] = None
    ) -> Tuple[str, Optional[str]]:
        """
        Process a single NIfTI file with HD-BET using subprocess.run().

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

        task_id = task_id or f"{time.time()}"

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

            # Use subprocess.run() with capture_output for simplicity and reliability
            # This matches the working pattern from the notebook test
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_sec,
                    # close_fds is automatically handled by subprocess.run()
                    # On Windows with Python 3.7+, it defaults correctly
                )

                if result.returncode != 0:
                    # Extract error message from stderr
                    error_msg = result.stderr[:500] if result.stderr else "Unknown error"
                    return "error", f"HD-BET failed (code {result.returncode}): {error_msg}"

            except subprocess.TimeoutExpired as e:
                # The process timed out
                return "timeout", f"Timeout after {self.timeout_sec}s"

            except Exception as e:
                # Other subprocess errors
                return "error", f"Subprocess error: {str(e)}"

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
            # Clean up any remaining temp output files (HD-BET output files)
            for temp_file in output_brain.parent.glob(f"temp_{task_id}*"):
                try:
                    temp_file.unlink()
                except:
                    pass

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
        """Kill any zombie HD-BET processes."""
        # Kill any hanging HD-BET processes
        killed = kill_zombie_processes("hd-bet")
        if killed > 0:
            print(f"Killed {killed} hanging HD-BET processes")