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
        temp_dir: Optional[Path] = None,
        verbose: bool = False,
        is_test_mode: bool = False
    ):
        self.device = device
        self.use_tta = use_tta
        # Increase timeout for test mode (first run may need to load models)
        self.timeout_sec = 1200 if is_test_mode else timeout_sec
        self.verbose = verbose
        self.is_test_mode = is_test_mode

        # Setup temp directory for subprocess output files
        if temp_dir:
            self.temp_dir = temp_dir
        else:
            self.temp_dir = Path(tempfile.gettempdir()) / "hd_bet_output"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.dir_lock = threading.Lock()  # Thread-safe directory creation

        # Check if HD-BET models are downloaded
        hd_bet_home = Path.home() / ".hd-bet"
        if not hd_bet_home.exists() and self.verbose:
            print("Note: HD-BET models may need to be downloaded on first run (this can take time)")

    def check_availability(self) -> bool:
        """Check if HD-BET is installed and accessible."""
        try:
            # Use PIPE to avoid file locking issues
            result = subprocess.run(
                ["hd-bet", "--help"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=60,
            )

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
        Process a single NIfTI file with HD-BET using file-based output redirection.

        This implementation uses subprocess.Popen with file handles to avoid
        pipe buffer overflow issues that can cause hanging when HD-BET produces
        large amounts of output (model loading, progress bars, etc.).

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

        # Create temporary files for subprocess output
        temp_stdout = self.temp_dir / f"hd_bet_{task_id}.out"
        temp_stderr = self.temp_dir / f"hd_bet_{task_id}.err"

        try:
            # HD-BET requires output to end with .nii.gz
            # Use a temporary name that HD-BET expects
            temp_output = output_brain.parent / f"temp_{task_id}.nii.gz"

            # Build HD-BET command with absolute paths
            cmd = [
                "hd-bet",
                "-i", str(input_path.absolute()),
                "-o", str(temp_output.absolute()),
                "-device", self.device
            ]

            # Add optional flags
            if not self.use_tta:
                cmd.append("--disable_tta")

            # Save mask file
            cmd.append("--save_bet_mask")

            # Log command if verbose
            if self.verbose:
                print(f"Running HD-BET command: {' '.join(cmd)}")

            # Run HD-BET with file-based output handling to avoid pipe buffer issues
            with open(temp_stdout, "w") as fout, open(temp_stderr, "w") as ferr:
                process = subprocess.Popen(
                    cmd,
                    stdout=fout,
                    stderr=ferr,
                    cwd=str(Path.cwd()),  # Explicitly set working directory
                    # Prevent process from hanging on to resources
                    close_fds=(os.name != 'nt'),  # close_fds=True not supported on Windows
                    # For Unix: create new process group for better control
                    preexec_fn=os.setsid if os.name != "nt" else None,
                )

                try:
                    # Wait for completion with timeout
                    returncode = process.wait(timeout=self.timeout_sec)

                    if returncode != 0:
                        # Read full output for debugging
                        with open(temp_stdout, "r") as f:
                            stdout_content = f.read()
                        with open(temp_stderr, "r") as f:
                            stderr_content = f.read()

                        # Log verbose output if requested
                        if self.verbose:
                            print(f"HD-BET stdout: {stdout_content[:1000]}")
                            print(f"HD-BET stderr: {stderr_content[:1000]}")

                        error_msg = stderr_content[:500] if stderr_content else stdout_content[:500]
                        return "error", f"HD-BET failed (code {returncode}): {error_msg}"

                except subprocess.TimeoutExpired:
                    # Kill the process on timeout
                    if os.name != "nt":
                        # Unix: Kill entire process group
                        try:
                            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                            time.sleep(2)  # Give it time to terminate
                            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                        except ProcessLookupError:
                            pass  # Process already dead
                    else:
                        # Windows: Use terminate/kill
                        process.terminate()
                        time.sleep(2)
                        try:
                            process.kill()
                        except:
                            pass

                    # Read any output that was produced before timeout
                    try:
                        with open(temp_stdout, "r") as f:
                            stdout_content = f.read()
                        with open(temp_stderr, "r") as f:
                            stderr_content = f.read()

                        if self.verbose:
                            print(f"HD-BET timed out. Stdout: {stdout_content[:500]}")
                            print(f"HD-BET timed out. Stderr: {stderr_content[:500]}")

                        # Check if models need to be downloaded
                        if "downloading" in stderr_content.lower() or "downloading" in stdout_content.lower():
                            return "timeout", f"Timeout after {self.timeout_sec}s (possibly downloading models - try again)"
                    except:
                        pass

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
            for temp_file in [temp_stdout, temp_stderr]:
                try:
                    temp_file.unlink(missing_ok=True)
                except:
                    pass

            # Clean up any remaining HD-BET temp output files
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
        """Kill any zombie HD-BET processes and clean up temp directory."""
        # Kill any hanging HD-BET processes
        killed = kill_zombie_processes("hd-bet")
        if killed > 0:
            print(f"Killed {killed} hanging HD-BET processes")

        # Clean up temp directory
        try:
            if self.temp_dir.exists():
                for temp_file in self.temp_dir.glob("hd_bet_*.out"):
                    try:
                        temp_file.unlink(missing_ok=True)
                    except:
                        pass
                for temp_file in self.temp_dir.glob("hd_bet_*.err"):
                    try:
                        temp_file.unlink(missing_ok=True)
                    except:
                        pass
        except Exception:
            pass  # Silently fail cleanup