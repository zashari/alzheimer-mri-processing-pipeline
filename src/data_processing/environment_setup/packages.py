from __future__ import annotations

import importlib
import importlib.metadata as ilmd
import os
import subprocess
import sys
import warnings
from typing import Dict, Optional, Tuple


def _run(cmd: list[str], timeout: int | None = None) -> subprocess.CompletedProcess:
    """Execute a command with proper error handling."""
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except Exception as e:
        print(f"   [X] Failed: {' '.join(cmd)}\n   -> {e}")
        return subprocess.CompletedProcess(cmd, 1, "", str(e))


def _pip(args: list[str]) -> int:
    """Execute pip commands."""
    cmd = [sys.executable, "-m", "pip"] + args
    cp = _run(cmd)
    if cp.stdout:
        print(cp.stdout.strip())
    if cp.returncode != 0 and cp.stderr:
        print(cp.stderr.strip())
    return cp.returncode


def install_package(pip_name: str, index_url: Optional[str] = None, dry_run: bool = False) -> bool:
    """Install a package via pip."""
    args = ["install", "-U", pip_name]
    if index_url:
        args += ["--index-url", index_url]
    print(f"   [PIP] pip {' '.join(args)}")
    if dry_run:
        return True
    return _pip(args) == 0


def safe_version(
    module_name: str,
    *,
    attr: str = "__version__",
    dists: tuple[str, ...] | None = None,
    fallback: str = "unknown"
) -> str:
    """
    Try module.__version__; if missing, try importlib.metadata for provided
    distributions (first that exists). Never raises; returns 'unknown' fallback.
    """
    try:
        mod = importlib.import_module(module_name)
        if hasattr(mod, attr):
            v = getattr(mod, attr)
            if isinstance(v, str):
                return v
    except Exception:
        pass

    if dists:
        for d in dists:
            try:
                return ilmd.version(d)
            except ilmd.PackageNotFoundError:
                continue
            except Exception:
                continue
    return fallback


def torchvision_ops_ok(verbose: bool = False) -> bool:
    """Check if TorchVision compiled ops are available."""
    try:
        import torchvision
        from torchvision.extension import _has_ops
        ok = bool(_has_ops())
        if ok:
            from torchvision.ops import nms  # sentinel import
        if verbose:
            print(f"   [CHECK] TorchVision {getattr(torchvision, '__version__', '?')}: ops -> {ok}")
        return ok
    except Exception:
        return False


def torch_stack_status() -> Tuple[bool, str]:
    """
    Returns (healthy, reason).
    healthy == True when: torch, torchvision, torchaudio import AND TV ops present.
    """
    try:
        import torch, torchvision, torchaudio  # noqa
    except Exception as e:
        return False, f"missing or broken import: {e}"
    if not torchvision_ops_ok(verbose=True):
        return False, "torchvision compiled ops missing"
    return True, "ok"


def _install_trio_from_index(tag: Optional[str]) -> bool:
    """Install PyTorch trio from specific index."""
    index = "https://download.pytorch.org/whl/cpu" if tag is None else f"https://download.pytorch.org/whl/{tag}"
    label = "CPU wheels" if tag is None else f"{tag} wheels"
    print(f"   [PIP] Installing PyTorch trio from {label}...")
    # Upgrade only; avoid --force-reinstall unless strictly necessary
    rc = _pip(["install", "--no-cache-dir",
               "torch", "torchvision", "torchaudio", "--index-url", index])
    return rc == 0


def ensure_torch_cuda_stack(cuda_tag: Optional[str] = None, *, auto_install: bool = False, dry_run: bool = False) -> None:
    """
    Behavior:
      * If torch/vision/audio already healthy (incl. ops) -> SKIP install.
      * Otherwise, install matched trio from cu126 -> cu124 -> CPU, verifying ops.
    """
    healthy, reason = torch_stack_status()
    if healthy:
        import torch, torchvision
        print("   [OK] Torch stack already healthy; skipping install")
        print(f"      torch: {torch.__version__} (CUDA {torch.version.cuda}) | torchvision: {torchvision.__version__}")
        return

    print(f"   [INFO] Torch stack not healthy ({reason}); preparing install...")

    if not auto_install:
        print("   [WARN] Torch stack missing; auto-install disabled")
        return

    if dry_run:
        print("   [DRY-RUN] Would install PyTorch trio")
        return

    # Detect GPU for smart selection
    from .gpu import detect_gpu, pick_cuda_tag_from_driver
    _, has_cuda = detect_gpu()
    primary = cuda_tag or (pick_cuda_tag_from_driver() if has_cuda else None)

    # Prioritize cu126 as per your environment
    candidates = []
    if primary == "cu126":
        candidates = ["cu126", "cu124"]
    elif primary == "cu124":
        candidates = ["cu124", "cu126"]
    elif primary:
        candidates = [primary, "cu126", "cu124"]
    else:
        candidates = ["cu126", "cu124"]

    candidates.append(None)  # CPU fallback

    for tag in candidates:
        if _install_trio_from_index(tag) and torchvision_ops_ok(verbose=True):
            import torch
            print(f"   [OK] Torch installed: {torch.__version__} (CUDA {torch.version.cuda})")
            if torch.cuda.is_available():
                print(f"   [OK] CUDA device: {torch.cuda.get_device_name(0)}")
            else:
                print("   [INFO] CPU-only build in use.")
            return
        print("   [WARN] Attempt didn't pass ops check; trying next candidate...")

    raise RuntimeError("Exhausted candidates; TorchVision ops remain unavailable.")


def ensure_torch_stack(cuda_tag: Optional[str], *, auto_install: bool, dry_run: bool) -> None:
    """Legacy interface for compatibility."""
    ensure_torch_cuda_stack(cuda_tag, auto_install=auto_install, dry_run=dry_run)


def ensure_sklearn_submodules() -> None:
    """
    Guarantee scikit-learn is importable with submodules. Handles:
    * Half-imported 'sklearn' left in sys.modules after a failed import
    * Local shadowing packages/files named 'sklearn'
    Strategy: purge sys.modules -> try import -> if still broken, reinstall wheels-only and retry.
    """
    import sys, importlib

    def _purge():
        for k in list(sys.modules):
            if k == "sklearn" or k.startswith("sklearn."):
                del sys.modules[k]
        importlib.invalidate_caches()

    # Warn if a local shadowing path exists
    try:
        here = os.getcwd()
        if os.path.isdir(os.path.join(here, "sklearn")) or os.path.exists(os.path.join(here, "sklearn.py")):
            print("   [WARN] A local 'sklearn' package/file exists in your project directory; it may shadow the real package.")
    except Exception:
        pass

    try:
        importlib.import_module("sklearn.model_selection")
        importlib.import_module("sklearn.metrics")
        return
    except Exception:
        _purge()
        try:
            importlib.import_module("sklearn.model_selection")
            importlib.import_module("sklearn.metrics")
            return
        except Exception:
            print("   [FIX] Reinstalling scikit-learn wheels...")
            _ = _pip(["install", "-U", "--only-binary=:all:", "scikit-learn"])
            _purge()
            importlib.import_module("sklearn.model_selection")
            importlib.import_module("sklearn.metrics")
            print("   [OK] scikit-learn submodules OK")


def ensure_cv2() -> None:
    """
    Ensure 'cv2' is a working OpenCV module (not shadowed, not half-imported).
    IMPORTANT (Windows): avoid modifying NumPy from an active kernel.
    Strategy:
      * If NumPy is already imported, do NOT pip-reinstall here -> instruct restart.
      * Otherwise, reinstall OpenCV with --no-deps (so NumPy isn't touched).
      * Prefer opencv-python-headless; fall back to opencv-python if you need GUI.
    """
    import sys, importlib, os

    def _purge():
        for k in list(sys.modules):
            if k == "cv2" or k.startswith("cv2."):
                del sys.modules[k]
        importlib.invalidate_caches()

    # Shadow check
    here = os.getcwd()
    shadows = []
    if os.path.isdir(os.path.join(here, "cv2")):
        shadows.append(os.path.join(here, "cv2"))
    if os.path.exists(os.path.join(here, "cv2.py")):
        shadows.append(os.path.join(here, "cv2.py"))
    if shadows:
        print(f"   [WARN] Local shadowing detected for 'cv2': {shadows} - rename/remove them.")

    def _validate() -> bool:
        try:
            import cv2  # noqa
            ok_api = hasattr(cv2, "imread") and callable(getattr(cv2, "imread", None))
            # Accept if __version__ exists OR we can read dist metadata
            from importlib.metadata import version, PackageNotFoundError
            has_ver = hasattr(cv2, "__version__")
            if not has_ver:
                try:
                    _ = version("opencv-python")
                    has_ver = True
                except PackageNotFoundError:
                    try:
                        _ = version("opencv-python-headless")
                        has_ver = True
                    except PackageNotFoundError:
                        has_ver = False
            return ok_api and has_ver
        except Exception:
            return False

    if _validate():
        return

    # Try a purge + re-import first
    _purge()
    if _validate():
        return

    # If NumPy is loaded, avoid reinstall from this kernel (Windows file locks)
    if "numpy" in sys.modules:
        print("   [WARN] NumPy is currently imported in this kernel. "
              "On Windows, reinstalling OpenCV can fail with file locks.")
        print("   [TIP] Restart the kernel (or stop Jupyter) and then install from a terminal:")
        print("      pip install -U --only-binary=:all: opencv-python-headless")
        return

    # Reinstall OpenCV WITHOUT touching dependencies (avoid NumPy swap)
    print("   [FIX] Reinstalling OpenCV (headless) with --no-deps...")
    if _pip(["install", "-U", "--only-binary=:all:", "--no-deps", "opencv-python-headless"]) != 0:
        print("   [WARN] Headless install failed; trying full OpenCV with --no-deps...")
        _ = _pip(["install", "-U", "--only-binary=:all:", "--no-deps", "opencv-python"])

    _purge()
    if not _validate():
        raise RuntimeError("OpenCV (cv2) remains broken; check for local 'cv2' shadowing and retry after kernel restart.")


def check_and_install_packages(dry_run: bool = False) -> None:
    """Check and install all required scientific packages."""
    pkgs: Dict[str, str] = {
        "numpy": "numpy",
        "pandas": "pandas",
        "cv2": "opencv-python-headless",
        "PIL": "Pillow",
        "skimage": "scikit-image",
        "scipy": "scipy",
        "matplotlib": "matplotlib",
        "seaborn": "seaborn",
        "IPython": "ipython",
        "tqdm": "tqdm",
        "ipywidgets": "ipywidgets",
        "sklearn": "scikit-learn",
        "timm": "timm",
        "nibabel": "nibabel",
        "ants": "antspyx",
        "psutil": "psutil",
    }

    print("\n[CHECK] Checking non-Torch packages...")
    missing = []
    for imp, pipname in pkgs.items():
        try:
            importlib.import_module(imp)
            print(f"   [OK] {imp} - present")
        except Exception:
            print(f"   [X] {imp} - missing")
            missing.append(pipname)

    if missing:
        print("\n[PIP] Installing missing packages...")
        if not dry_run:
            _pip(["install", "--upgrade", "pip"])
            for name in missing:
                if name == "scikit-learn":
                    ok = _pip(["install", "-U", "--only-binary=:all:", name]) == 0
                elif name in ["opencv-python-headless", "opencv-python"]:
                    ok = _pip(["install", "-U", "--only-binary=:all:", "--no-deps", name]) == 0
                else:
                    ok = install_package(name, dry_run=dry_run)
                if not ok:
                    print(f"   [WARN] Installation may have failed for: {name}")
        else:
            print("   [DRY-RUN] Would install:", ", ".join(missing))
    else:
        print("[OK] All non-Torch packages already installed.")

    # Always validate sklearn & cv2 (handles half-imports/shadowing)
    if not dry_run:
        ensure_sklearn_submodules()
        ensure_cv2()


def ensure_science_stack(dry_run: bool) -> None:
    """Legacy interface - now calls comprehensive package checker."""
    check_and_install_packages(dry_run)


def import_heavy_and_summarize() -> None:
    """Import heavy dependencies and print summary."""
    warnings.filterwarnings("ignore")
    import numpy as np, pandas as pd, cv2, nibabel as nib, matplotlib.pyplot as plt, torch

    np_v   = safe_version("numpy")
    pd_v   = safe_version("pandas")
    cv_v   = safe_version("cv2", dists=("opencv-python", "opencv-python-headless"))
    nib_v  = safe_version("nibabel")
    plt_v  = getattr(plt.matplotlib, "__version__", "unknown")
    th_v   = safe_version("torch")
    cu_v   = getattr(torch.version, "cuda", "unknown")
    ipyw_v = safe_version("ipywidgets", dists=("ipywidgets",))

    print("\n[PKG] Package Import Summary:")
    print(f"   * NumPy:       {np_v}")
    print(f"   * Pandas:      {pd_v}")
    print(f"   * OpenCV:      {cv_v}")
    print(f"   * NiBabel:     {nib_v}")
    print(f"   * PyTorch:     {th_v}")
    print(f"   * Matplotlib:  {plt_v}")
    print(f"   * ipywidgets:  {ipyw_v}")
    if torch.cuda.is_available():
        print(f"   * GPU:         {torch.cuda.get_device_name(0)} (CUDA {cu_v})")
    else:
        print("   * GPU:         Not available (CPU mode)")
    print("[OK] All critical packages imported successfully.")
    print("[INFO] GPU acceleration enabled." if torch.cuda.is_available() else "[INFO] Running in CPU mode.")


def apply_gpu_optimizations() -> None:
    """Apply optimal settings for machine learning on GPU."""
    try:
        import torch
        import gc

        if not torch.cuda.is_available():
            return

        # Clear any existing allocations
        torch.cuda.empty_cache()
        gc.collect()

        gpu_name = torch.cuda.get_device_name(0)

        # Enable optimizations based on GPU generation
        torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes

        # Enable TF32 on Ampere and newer (compute capability >= 8.0)
        props = torch.cuda.get_device_properties(0)
        if props.major >= 8:
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True
            tf32_enabled = True
        else:
            tf32_enabled = False

        print(f"\n[OPT] Applied GPU Optimizations:")
        print(f"   [OK] cuDNN benchmark enabled")
        if tf32_enabled:
            print(f"   [OK] TensorFloat-32 enabled (Tensor Cores)")
        else:
            print(f"   [INFO] TF32 not available on this GPU")

        # Set memory management
        torch.cuda.empty_cache()  # Clear any existing cache
        print(f"   [OK] GPU memory cache cleared")

        # Additional settings for better performance
        if props.major >= 7:  # Volta and newer
            print(f"   [OK] Mixed precision training available")

    except ImportError:
        pass
    except Exception as e:
        print(f"[WARN] Could not apply GPU optimizations: {e}")