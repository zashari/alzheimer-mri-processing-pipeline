"""HD-BET skull stripping processor."""

from __future__ import annotations

import os
import platform
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..gpu_utils import kill_zombie_processes


class HDBETProcessor:
    """Handles HD-BET skull stripping operations with adaptive execution backend."""

    def __init__(
        self,
        device: str = "cuda",
        use_tta: bool = False,
        timeout_sec: int = 600,
        temp_dir: Optional[Path] = None,
        verbose: bool = False,
        is_test_mode: bool = False,
        execution_method: str = "auto"  # New: "auto", "subprocess", "module", "api"
    ):
        self.device = device
        self.use_tta = use_tta
        # Increase timeout for test mode (first run may need to load models)
        self.timeout_sec = 1200 if is_test_mode else timeout_sec
        self.verbose = verbose
        self.is_test_mode = is_test_mode
        self.execution_method = execution_method
        self._execution_backend = None  # Will be determined in _setup_execution_backend()
        self._hd_bet_fork_version = False  # Will be detected when checking availability
        self._hd_bet_command = "hd-bet"  # Will be updated based on what's available

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
        """Check if HD-BET is installed and accessible, and detect version."""
        try:
            # On Windows, try different command variants
            commands_to_try = []
            if platform.system() == "Windows":
                commands_to_try = ["hd-bet.cmd", "hd-bet", "hd-bet.py"]
            else:
                commands_to_try = ["hd-bet"]

            for cmd_variant in commands_to_try:
                try:
                    result = subprocess.run(
                        [cmd_variant, "--help"],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        timeout=60,
                    )

                    if result.returncode == 0:
                        self._hd_bet_command = cmd_variant  # Store the working command
                        # Detect fork version from help text
                        help_text = result.stdout + result.stderr
                        if "-tta" in help_text and "-mode" in help_text:
                            self._hd_bet_fork_version = True
                            if self.verbose:
                                print(f"✓ Using patched HD-BET fork ({cmd_variant}) with Windows fixes")
                        else:
                            self._hd_bet_fork_version = False
                            if self.verbose:
                                print(f"✓ Using original HD-BET version ({cmd_variant})")
                        return True
                except:
                    continue

            return False

        except Exception:
            return False

    def _setup_execution_backend(self) -> str:
        """
        Determine the best execution method for HD-BET based on platform and availability.

        Returns:
            str: The backend to use: "subprocess_native", "subprocess_module", or "api_direct"
        """
        # If user specified a method, try to use it
        if self.execution_method != "auto":
            if self.execution_method == "api":
                return "api_direct"
            elif self.execution_method == "module":
                return "subprocess_module"
            else:
                return "subprocess_native"

        # Auto-detection logic
        system = platform.system()

        # Strategy 1: Check if native hd-bet command works (best for Unix)
        if system in ["Linux", "Darwin"]:  # Unix-like systems
            if self._test_native_command():
                if self.verbose:
                    print("✓ Using native hd-bet command (Unix)")
                return "subprocess_native"

        # Strategy 2: For Windows or if native fails, try Python module execution
        if self._test_module_execution():
            if self.verbose:
                print(f"✓ Using Python module execution (-m HD_BET) on {system}")
            return "subprocess_module"

        # Strategy 3: Check if we can import HD_BET directly (fallback)
        if self._test_api_import():
            if self.verbose:
                print("✓ Using direct API import (fallback)")
            return "api_direct"

        # Strategy 4: Last resort - try native anyway
        if self.verbose:
            print("⚠ No optimal method found, attempting native command")
        return "subprocess_native"

    def _test_native_command(self) -> bool:
        """Test if native hd-bet command is available and detect version."""
        try:
            result = None
            # On Windows, try different command variants
            if platform.system() == "Windows":
                # Try hd-bet.cmd first (patched fork on Windows)
                for cmd_variant in ["hd-bet.cmd", "hd-bet", "hd-bet.py"]:
                    try:
                        result = subprocess.run(
                            [cmd_variant, "--help"],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True,
                            timeout=5
                        )
                        if result.returncode == 0:
                            self._hd_bet_command = cmd_variant  # Store the working command
                            break
                    except:
                        continue
            else:
                # Unix systems
                result = subprocess.run(
                    ["hd-bet", "--help"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=5
                )
                self._hd_bet_command = "hd-bet"

            # Check if we found a working command
            if result and result.returncode == 0:
                # Check if it's the patched fork by looking for -tta in help
                help_text = result.stdout + result.stderr
                if "-tta" in help_text and "-mode" in help_text:
                    self._hd_bet_fork_version = True
                    if self.verbose:
                        print(f"Detected patched HD-BET fork ({self._hd_bet_command})")
                else:
                    self._hd_bet_fork_version = False
                    if self.verbose:
                        print(f"Detected original HD-BET version ({self._hd_bet_command})")
                return True
            return False
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
        except Exception:
            return False

    def _test_module_execution(self) -> bool:
        """Test if HD_BET can be run as a Python module."""
        try:
            result = subprocess.run(
                [sys.executable, "-m", "HD_BET", "--help"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=5
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, Exception):
            return False

    def _test_api_import(self) -> bool:
        """Test if HD_BET can be imported directly."""
        try:
            import HD_BET.run
            return True
        except ImportError:
            return False

    def process_file(
        self,
        input_path: Path,
        output_brain: Path,
        output_mask: Path,
        task_id: Optional[str] = None
    ) -> Tuple[str, Optional[str]]:
        """
        Process a single NIfTI file using the optimal execution method.

        This implementation automatically selects the best execution backend
        based on platform and availability (subprocess_native, subprocess_module,
        or api_direct).

        Args:
            input_path: Path to input NIfTI file
            output_brain: Path for output brain file
            output_mask: Path for output mask file
            task_id: Optional task identifier for temp files

        Returns:
            Tuple of (status, error_message)
            Status can be: "success", "skip", "timeout", or error message
        """
        # Setup execution backend on first run
        if self._execution_backend is None:
            self._execution_backend = self._setup_execution_backend()

        # Check if output already exists
        if output_brain.exists():
            return "skip", None

        # Route to appropriate execution method
        if self._execution_backend == "api_direct":
            return self._process_with_api(input_path, output_brain, output_mask, task_id)
        else:
            return self._process_with_subprocess(input_path, output_brain, output_mask, task_id)

    def _process_with_subprocess(
        self,
        input_path: Path,
        output_brain: Path,
        output_mask: Path,
        task_id: Optional[str] = None
    ) -> Tuple[str, Optional[str]]:
        """
        Process using subprocess (either native command or Python module).
        """
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

            # Build command based on backend
            if self._execution_backend == "subprocess_module":
                cmd = [sys.executable, "-m", "HD_BET"]
            else:
                cmd = [self._hd_bet_command]  # Use the detected command variant

            # Add arguments
            cmd.extend([
                "-i", str(input_path.absolute()),
                "-o", str(temp_output.absolute()),
                "-device", self.device
            ])

            # Add version-specific arguments
            if self._hd_bet_fork_version:
                # Patched fork format (sh-shahrokhi version)
                # Add mode for better performance
                cmd.extend(["-mode", "accurate" if self.use_tta else "fast"])
                # TTA flag: -tta 1 (enable) or -tta 0 (disable)
                cmd.extend(["-tta", "1" if self.use_tta else "0"])
                # Save mask: -s 1 (save) or -s 0 (don't save)
                cmd.extend(["-s", "1"])
                # Enable postprocessing
                cmd.extend(["-pp", "1"])
            else:
                # Original HD-BET format
                if not self.use_tta:
                    cmd.append("--disable_tta")
                cmd.append("--save_bet_mask")

            # Log command if verbose
            if self.verbose:
                backend_name = "module" if self._execution_backend == "subprocess_module" else "native"
                print(f"Running HD-BET ({backend_name}): {' '.join(cmd)}")

            # Setup environment for Windows to prevent multiprocessing deadlocks
            env = os.environ.copy()
            if platform.system() == "Windows":
                # Force single-threading to prevent Windows multiprocessing issues
                env["OMP_NUM_THREADS"] = "1"
                env["MKL_NUM_THREADS"] = "1"
                env["NUMEXPR_NUM_THREADS"] = "1"
                env["OPENBLAS_NUM_THREADS"] = "1"
                if self.verbose:
                    print("Windows detected: Using single-threaded execution to prevent deadlocks")

            # Run HD-BET with file-based output handling to avoid pipe buffer issues
            with open(temp_stdout, "w") as fout, open(temp_stderr, "w") as ferr:
                process = subprocess.Popen(
                    cmd,
                    stdout=fout,
                    stderr=ferr,
                    env=env,  # Pass the environment with Windows fixes
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

    def _process_with_api(
        self,
        input_path: Path,
        output_brain: Path,
        output_mask: Path,
        task_id: Optional[str] = None
    ) -> Tuple[str, Optional[str]]:
        """
        Process using direct Python API import (no subprocess).
        Best for small files or when subprocess fails.
        """
        try:
            from HD_BET.run import run_hd_bet

            # Ensure output directory exists (thread-safe)
            with self.dir_lock:
                output_brain.parent.mkdir(parents=True, exist_ok=True)

            task_id = task_id or f"{time.time()}"

            # HD_BET API expects specific naming
            temp_output = str(output_brain.parent / f"temp_{task_id}.nii.gz")

            if self.verbose:
                print(f"Using HD-BET API directly for {input_path.name}")

            # Run HD-BET with timeout using threading
            result = {"success": False, "error": None}

            def run_hd_bet_thread():
                try:
                    # The patched fork uses different argument names
                    run_hd_bet(
                        input=str(input_path),  # Changed from input_file
                        output=temp_output,     # Changed from output_file
                        mode="accurate" if self.use_tta else "fast",
                        device=self.device,
                        tta=1 if self.use_tta else 0,  # Fork expects 0/1 not bool
                        save_mask=1,  # Fork expects 1 not True
                        overwrite_existing=1  # Fork expects 1 not True
                    )
                    result["success"] = True
                except Exception as e:
                    result["error"] = str(e)

            # Run in thread with timeout
            thread = threading.Thread(target=run_hd_bet_thread)
            thread.daemon = True
            thread.start()
            thread.join(timeout=self.timeout_sec)

            if thread.is_alive():
                # Timeout occurred
                return "timeout", f"API call timeout after {self.timeout_sec}s"

            if not result["success"]:
                return "error", f"API error: {result['error']}"

            # Move output files
            hd_bet_brain = Path(temp_output)
            hd_bet_mask = Path(str(temp_output).replace(".nii.gz", "_bet.nii.gz"))

            if hd_bet_brain.exists():
                shutil.move(str(hd_bet_brain), str(output_brain))
            else:
                return "error", "Brain extraction file not created"

            if hd_bet_mask.exists():
                shutil.move(str(hd_bet_mask), str(output_mask))

            # Verify output
            if output_brain.exists():
                return "success", None
            else:
                return "error", "Output file not created"

        except ImportError as e:
            return "error", f"Failed to import HD_BET: {e}"
        except Exception as e:
            return "error", f"API execution failed: {str(e)}"
        finally:
            # Clean up any remaining temp files
            try:
                for temp_file in output_brain.parent.glob(f"temp_{task_id}*"):
                    temp_file.unlink(missing_ok=True)
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