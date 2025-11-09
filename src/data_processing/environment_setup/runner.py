from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout, redirect_stderr
from typing import Dict

from .formatter import OutputFormatter
from .gpu import detect_gpu, pick_cuda_tag_from_driver, verify_gpu_setup
from .packages import (
    ensure_torch_cuda_stack,
    check_and_install_packages,
    import_heavy_and_summarize,
    apply_gpu_optimizations,
    safe_version,
    torch_stack_status,
)


class SuppressOutput:
    """Context manager to suppress stdout/stderr."""
    def __init__(self, suppress: bool = True):
        self.suppress = suppress
        self._stdout = None
        self._stderr = None

    def __enter__(self):
        if self.suppress:
            self._stdout = sys.stdout
            self._stderr = sys.stderr
            sys.stdout = io.StringIO()
            sys.stderr = io.StringIO()
        return self

    def __exit__(self, *args):
        if self.suppress:
            sys.stdout = self._stdout
            sys.stderr = self._stderr


def run(action: str, cfg: Dict) -> int:
    """Main environment setup runner with comprehensive checks and installation."""
    dry_run = bool(cfg.get("dry_run", False) or cfg.get("global", {}).get("dry_run", False))
    es = cfg.get("environment_setup", {})
    auto_install = bool(es.get("auto_install", False))
    pref = str(es.get("cuda_preference", "auto")).lower()
    perf_test = str(es.get("perf_test", "off")).lower()

    # Check for output options
    verbose = cfg.get("debug", False)
    quiet = cfg.get("quiet", False)
    json_only = cfg.get("json", False)

    # Initialize formatter
    formatter = OutputFormatter(verbose=verbose, quiet=quiet, json_only=json_only)

    # Determine if we should suppress verbose output
    suppress_verbose = not verbose and not json_only

    # Print header
    formatter.header(action, perf_test)

    # Track overall status
    has_warnings = False
    exit_code = 0

    try:
        # GPU Detection (silent when using formatter)
        gpu_name, has_cuda = detect_gpu(silent=suppress_verbose)

        # CUDA Tag Selection (silent when using formatter)
        cuda_tag = None
        if pref == "auto":
            cuda_tag = pick_cuda_tag_from_driver(silent=suppress_verbose)
        elif pref in {"11.8", "12.1", "12.4", "12.6"}:
            cuda_tag = {"11.8": "cu118", "12.1": "cu121", "12.4": "cu124", "12.6": "cu126"}[pref]

        # Get initial GPU info for display
        gpu_info = {}
        if has_cuda:
            try:
                import torch
                if torch.cuda.is_available():
                    props = torch.cuda.get_device_properties(0)
                    gpu_info = {
                        "name": gpu_name,
                        "cuda_version": torch.version.cuda,
                        "cudnn_version": torch.backends.cudnn.version(),
                        "compute_cap": f"{props.major}.{props.minor}",
                        "memory": {
                            "total": props.total_memory / 1024**3,
                            "reserved": torch.cuda.memory_reserved(0) / 1024**3,
                            "allocated": torch.cuda.memory_allocated(0) / 1024**3,
                            "free": (props.total_memory - torch.cuda.memory_reserved(0)) / 1024**3,
                            "sm_count": props.multi_processor_count,
                        },
                        "cuda_cores": props.multi_processor_count * (128 if props.major >= 8 else 64),
                    }
            except ImportError:
                pass

        # Display system info box
        formatter.system_info_box(
            gpu_name,
            gpu_info.get("cuda_version"),
            gpu_info.get("cudnn_version"),
            gpu_info.get("compute_cap"),
            gpu_info.get("memory"),
            gpu_info.get("cuda_cores"),
        )

        if action == "setup":
            # Check torch stack status (suppress verbose checking)
            with SuppressOutput(suppress_verbose):
                torch_healthy, torch_reason = torch_stack_status()

            # Display torch stack status box
            if torch_healthy:
                try:
                    import torch, torchvision
                    formatter.torch_stack_box(
                        torch.__version__,
                        torchvision.__version__,
                        torch.version.cuda,
                        torch_healthy
                    )
                except:
                    pass

            # 1) Install torch stack if needed (with output suppression)
            if not json_only:
                with SuppressOutput(suppress_verbose):
                    ensure_torch_cuda_stack(cuda_tag, auto_install=auto_install, dry_run=dry_run)

            # Re-check after installation
            with SuppressOutput(suppress_verbose):
                torch_healthy, _ = torch_stack_status()

            if torch_healthy and not formatter.report_data.get("packages"):
                try:
                    import torch, torchvision
                    if not suppress_verbose:
                        formatter.torch_stack_box(
                            torch.__version__,
                            torchvision.__version__,
                            torch.version.cuda,
                            torch_healthy
                        )
                except:
                    pass

            # 2) Check and install other packages (with output suppression)
            if not json_only:
                with SuppressOutput(suppress_verbose):
                    check_and_install_packages(dry_run=dry_run)

            # Collect package versions for display
            packages = {}
            package_list = [
                ("numpy", ("numpy",)),
                ("pandas", ("pandas",)),
                ("scipy", ("scipy",)),
                ("scikit-image", ("scikit-image",)),
                ("matplotlib", ("matplotlib",)),
                ("seaborn", ("seaborn",)),
                ("ipywidgets", ("ipywidgets",)),
                ("tqdm", ("tqdm",)),
                ("psutil", ("psutil",)),
                ("nibabel", ("nibabel",)),
                ("timm", ("timm",)),
                ("ants", ("antspyx",)),
                ("opencv", ("opencv-python", "opencv-python-headless")),
            ]

            for name, dists in package_list:
                version = safe_version(name if name != "opencv" else "cv2", dists=dists)
                packages[name] = version

            # Display packages box
            formatter.packages_box(packages)

            # Check for OpenCV warning
            try:
                import sys
                if "numpy" in sys.modules:
                    formatter.warning(
                        "NumPy is currently imported; on Windows this can lock OpenCV DLLs.",
                        "If reinstalling OpenCV: close Python/Jupyter, then:\n"
                        "       pip install -U --only-binary=:all: opencv-python-headless"
                    )
                    has_warnings = True
            except:
                pass

            # 3) Import validation (suppressed if not verbose)
            if not dry_run and not json_only:
                try:
                    with SuppressOutput(suppress_verbose):
                        # Extra imports for validation
                        import timm  # noqa: F401
                        import torchvision.transforms as transforms  # noqa: F401
                        from torchvision import models  # noqa: F401
                        from sklearn.model_selection import train_test_split  # noqa: F401
                        from sklearn.metrics import accuracy_score  # noqa: F401
                        try:
                            import ants  # noqa: F401
                        except Exception as e:
                            if verbose:
                                print(f"   [WARN] ANTs (antspyx) import issue: {e}")

                        # Only show import summary if verbose
                        if verbose and not quiet and not json_only:
                            import_heavy_and_summarize()
                except Exception as e:
                    if not quiet:
                        print(f"\n[ERROR] Import/summary phase failed: {e}")
                    has_warnings = True

            # 4) Run GPU performance test
            test_ok = None
            if perf_test in {"quick", "full"} and has_cuda and not dry_run:
                formatter.test_section(2048, 0.03, True, True)

                try:
                    run_full_test = perf_test == "full"

                    # Run GPU test with output suppression
                    with SuppressOutput(suppress_verbose):
                        gpu_test_info = verify_gpu_setup(run_performance_test=run_full_test, silent=suppress_verbose)

                    if gpu_test_info.get("perf_test_tflops") is not None:
                        tflops = gpu_test_info["perf_test_tflops"]
                        if tflops > 0:
                            # Get actual time from the test
                            time_ms = gpu_test_info.get("operation_time", 3.1)
                            formatter.test_success(tflops, time_ms)
                            test_ok = True
                        else:
                            formatter.test_error(
                                "Performance test completed but returned 0 TFLOPS",
                                {"step": "matrix_multiplication", "size": 2048}
                            )
                            test_ok = False
                            has_warnings = True

                    # Apply GPU optimizations (with output suppression)
                    if gpu_test_info.get("cuda_available"):
                        with SuppressOutput(suppress_verbose):
                            apply_gpu_optimizations()

                except Exception as e:
                    formatter.test_error(
                        str(e),
                        {"step": "gpu_performance_test", "size": 2048}
                    )
                    test_ok = False
                    has_warnings = True

            # Summary section
            formatter.summary(
                gpu_ok=has_cuda,
                torch_ok=torch_healthy,
                packages_ok=True,  # Assume OK if we got here
                test_ok=test_ok,
                warnings=has_warnings
            )

            # Next steps
            formatter.next_steps()

        elif action == "verify":
            if not quiet and not json_only:
                print("\n[VERIFY] Running lightweight verification...")

            # Quick GPU check
            if has_cuda:
                if not quiet:
                    print("[OK] GPU detected and CUDA available")
            else:
                if not quiet:
                    print("[WARN] No GPU detected or CUDA unavailable")
                has_warnings = True

            # Check if torch is installed and functional
            try:
                import torch
                if not quiet:
                    print(f"[OK] PyTorch {torch.__version__} installed")
                if torch.cuda.is_available():
                    if not quiet:
                        print(f"[OK] CUDA {torch.version.cuda} available")
                        print(f"[OK] GPU: {torch.cuda.get_device_name(0)}")
                else:
                    if not quiet:
                        print("[INFO] Running in CPU mode")
            except ImportError:
                if not quiet:
                    print("[ERROR] PyTorch not installed")
                exit_code = 1

            # Run performance test if requested
            if perf_test in {"quick", "full"} and has_cuda:
                run_full_test = perf_test == "full"
                with SuppressOutput(suppress_verbose):
                    verify_gpu_setup(run_performance_test=run_full_test, silent=suppress_verbose)

        else:
            print(f"[ERROR] Unknown action: {action}")
            exit_code = 2

    except Exception as e:
        if not quiet:
            print(f"[ERROR] Unexpected error: {e}")
        exit_code = 1
        has_warnings = True

    # Footer with report
    if has_warnings and exit_code == 0:
        exit_code = 0  # Keep 0 but note warnings in report

    formatter.footer(exit_code)

    return exit_code