"""Image enhancement processor using Grey Wolf Optimization (GWO)."""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import cv2
    import numpy as np
    from scipy import ndimage
    from skimage.exposure import equalize_adapthist
    CV2_AVAILABLE = True
    NUMPY_AVAILABLE = True
    SCIPY_AVAILABLE = True
    SKIMAGE_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    NUMPY_AVAILABLE = False
    SCIPY_AVAILABLE = False
    SKIMAGE_AVAILABLE = False


def get_brain_mask(image: np.ndarray, threshold: float = 1e-6) -> np.ndarray:
    """
    Return boolean mask: True = brain, False = background (≈0).

    Args:
        image: Float array in [0,1]
        threshold: Pixels ≤ threshold are considered background

    Returns:
        Boolean mask array
    """
    return image > threshold


class GreyWolfOptimizer:
    """Grey Wolf Optimization algorithm for parameter optimization."""

    def __init__(
        self,
        objective_func,
        bounds: List[Tuple[float, float]],
        num_wolves: int = 20,
        max_iterations: int = 50,
        convergence_threshold: float = 1e-6,
        seed: Optional[int] = None
    ):
        """
        Initialize Grey Wolf Optimizer.

        Args:
            objective_func: Function to optimize (takes params array, returns fitness)
            bounds: List of (min, max) tuples for each parameter dimension
            num_wolves: Number of wolves in the pack
            max_iterations: Maximum optimization iterations
            convergence_threshold: Threshold for early stopping
            seed: Random seed for reproducibility
        """
        self.objective_func = objective_func
        self.bounds = np.array(bounds, dtype=np.float32)
        self.num_wolves = num_wolves
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.dim = len(bounds)

        # Set random seed
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # Initialize wolves uniformly in bounds
        self.wolves = np.zeros((self.num_wolves, self.dim), dtype=np.float32)
        for d in range(self.dim):
            self.wolves[:, d] = np.random.uniform(
                self.bounds[d, 0],
                self.bounds[d, 1],
                self.num_wolves
            )
        self.fitness = np.full(self.num_wolves, -np.inf, dtype=np.float32)

        # Leader positions/scores
        self.alpha_pos = np.zeros(self.dim, dtype=np.float32)
        self.beta_pos = np.zeros(self.dim, dtype=np.float32)
        self.delta_pos = np.zeros(self.dim, dtype=np.float32)
        self.alpha_score = self.beta_score = self.delta_score = -np.inf

        self.convergence_curve = []

    def _clip(self) -> None:
        """Clip wolves to bounds."""
        for d in range(self.dim):
            self.wolves[:, d] = np.clip(
                self.wolves[:, d],
                self.bounds[d, 0],
                self.bounds[d, 1]
            )

    def optimize(self, verbose: bool = False) -> Tuple[np.ndarray, float]:
        """
        Run optimization.

        Args:
            verbose: Print progress information

        Returns:
            Tuple of (best_params, best_fitness)
        """
        for it in range(self.max_iterations):
            # Evaluate fitness
            for i in range(self.num_wolves):
                self.fitness[i] = self.objective_func(self.wolves[i])

            # Update alpha, beta, delta
            for i in range(self.num_wolves):
                fit = self.fitness[i]
                if fit > self.alpha_score:
                    self.delta_score, self.delta_pos = self.beta_score, self.beta_pos.copy()
                    self.beta_score, self.beta_pos = self.alpha_score, self.alpha_pos.copy()
                    self.alpha_score, self.alpha_pos = fit, self.wolves[i].copy()
                elif fit > self.beta_score:
                    self.delta_score, self.delta_pos = self.beta_score, self.beta_pos.copy()
                    self.beta_score, self.beta_pos = fit, self.wolves[i].copy()
                elif fit > self.delta_score:
                    self.delta_score, self.delta_pos = fit, self.wolves[i].copy()

            # Linearly decreasing a from 2→0
            a = 2.0 - (2.0 * it) / self.max_iterations

            # Update positions
            for i in range(self.num_wolves):
                for d in range(self.dim):
                    r1, r2 = np.random.rand(), np.random.rand()
                    A1 = 2 * a * r1 - a
                    C1 = 2 * r2
                    D_alpha = abs(C1 * self.alpha_pos[d] - self.wolves[i, d])
                    X1 = self.alpha_pos[d] - A1 * D_alpha

                    r1, r2 = np.random.rand(), np.random.rand()
                    A2 = 2 * a * r1 - a
                    C2 = 2 * r2
                    D_beta = abs(C2 * self.beta_pos[d] - self.wolves[i, d])
                    X2 = self.beta_pos[d] - A2 * D_beta

                    r1, r2 = np.random.rand(), np.random.rand()
                    A3 = 2 * a * r1 - a
                    C3 = 2 * r2
                    D_delta = abs(C3 * self.delta_pos[d] - self.wolves[i, d])
                    X3 = self.delta_pos[d] - A3 * D_delta

                    self.wolves[i, d] = (X1 + X2 + X3) / 3.0

            self._clip()
            self.convergence_curve.append(self.alpha_score)

            # Early stop on plateau
            if it > 10:
                recent = self.convergence_curve[-10:]
                if max(recent) - min(recent) < self.convergence_threshold:
                    if verbose:
                        print(f"Converged at iter {it}")
                    break

            if verbose and it % 10 == 0:
                print(f"Iter {it:02d}  best fitness = {self.alpha_score:.6f}")

        return self.alpha_pos, self.alpha_score


class ImageEnhancer:
    """Collection of basic enhancement operators."""

    @staticmethod
    def clahe_enhancement(img: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
        if img.max() > 1.0:
            img = img.astype(np.float32) / 255.0
        out = equalize_adapthist(img, kernel_size=tile_grid_size, clip_limit=clip_limit)
        return (out * 255).astype(np.uint8)

    @staticmethod
    def gabor_enhancement(
        img: np.ndarray,
        wavelength: float = 10,
        theta: float = 0,
        sigma_x: float = 5,
        sigma_y: float = 5,
        gamma: float = 0.5
    ) -> np.ndarray:
        """Apply Gabor filter enhancement."""
        if img.max() > 1.0:
            img = img.astype(np.float32) / 255.0
        ksize = int(6 * max(sigma_x, sigma_y)) | 1  # ensure odd
        kernel = cv2.getGaborKernel(
            (ksize, ksize),
            sigma_x,
            np.deg2rad(theta),
            2 * np.pi / wavelength,
            gamma,
            0,
            ktype=cv2.CV_32F
        )
        resp = cv2.filter2D(img, cv2.CV_32F, kernel)
        out = np.clip(img + 0.3 * resp, 0, 1)
        return (out * 255).astype(np.uint8)

    @staticmethod
    def unsharp_masking(img: np.ndarray, sigma: float = 1.0, strength: float = 1.5) -> np.ndarray:
        """Apply unsharp masking."""
        if img.max() > 1.0:
            img = img.astype(np.float32) / 255.0
        blurred = ndimage.gaussian_filter(img, sigma=sigma)
        mask = img - blurred
        out = np.clip(img + strength * mask, 0, 1)
        return (out * 255).astype(np.uint8)

    @staticmethod
    def adaptive_enhancement(
        img: np.ndarray,
        clahe_clip: float = 2.0,
        gabor_strength: float = 0.3,
        unsharp_sigma: float = 1.0,
        unsharp_strength: float = 1.5
    ) -> np.ndarray:
        """Apply adaptive enhancement combining CLAHE, Gabor, and Unsharp."""
        if img.max() > 1.0:
            img = img.astype(np.float32) / 255.0
        # CLAHE
        out = equalize_adapthist(img, clip_limit=clahe_clip)
        # Gabor
        if gabor_strength > 0:
            kernel = cv2.getGaborKernel((15, 15), 3, 0, 10, 0.5, 0, ktype=cv2.CV_32F)
            resp = cv2.filter2D(out, cv2.CV_32F, kernel)
            out = out + gabor_strength * resp
        # Unsharp
        blur = ndimage.gaussian_filter(out, sigma=unsharp_sigma)
        out = np.clip(out + unsharp_strength * (out - blur), 0, 1)
        return (out * 255).astype(np.uint8)


class ImageQualityMetrics:
    """Mask-aware image quality metrics."""

    @staticmethod
    def _prep(img: np.ndarray) -> np.ndarray:
        """Prepare image for metric calculation."""
        return img.astype(np.float32) / 255.0 if img.max() > 1.0 else img

    @staticmethod
    def calculate_entropy(img: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
        """Calculate image entropy."""
        img = ImageQualityMetrics._prep(img)
        if mask is not None:
            img = img[mask]
        hist, _ = np.histogram(img.flatten(), bins=256, range=(0, 1))
        if hist.sum() == 0:
            return 0.0
        hist = hist / hist.sum()
        hist = hist[hist > 0]
        return float(-np.sum(hist * np.log2(hist)))

    @staticmethod
    def calculate_edge_energy(img: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
        """Calculate edge energy using Sobel operators."""
        img = ImageQualityMetrics._prep(img)
        sx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
        sy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(sx**2 + sy**2)
        return float(np.mean(mag[mask])) if mask is not None else float(np.mean(mag))

    @staticmethod
    def calculate_local_contrast(img: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
        """Calculate local contrast."""
        img = ImageQualityMetrics._prep(img)
        k = np.ones((9, 9), dtype=np.float32) / 81.0
        mu = cv2.filter2D(img, -1, k)
        var = cv2.filter2D(img**2, -1, k) - mu**2
        std = np.sqrt(np.clip(var, 0, None))
        return float(np.mean(std[mask])) if mask is not None else float(np.mean(std))


class ImageEnhancementProcessor:
    """Processor for image enhancement using GWO optimization."""

    def __init__(
        self,
        method: str = "adaptive",
        gwo_iterations: int = 30,
        num_wolves: int = 15,
        slice_types: Optional[List[str]] = None,
        splits: Optional[List[str]] = None,
        groups: Optional[List[str]] = None,
        required_visits: Optional[List[str]] = None,
        max_images_per_class: Optional[int] = None,
        seed: Optional[int] = None,
        verbose: bool = False
    ):
        """
        Initialize image enhancement processor.

        Args:
            method: Enhancement method ('adaptive', 'clahe', 'gabor', 'unsharp')
            gwo_iterations: Number of GWO iterations
            num_wolves: Number of wolves in GWO pack
            slice_types: List of slice types to process
            splits: List of splits to process
            groups: List of groups to process
            required_visits: List of required visits
            max_images_per_class: Maximum images per class (None = all)
            seed: Random seed for reproducibility
            verbose: Show detailed output
        """
        self.method = method
        self.gwo_iterations = gwo_iterations
        self.num_wolves = num_wolves
        self.slice_types = slice_types or ["axial", "coronal", "sagittal"]
        self.splits = splits or ["train", "val", "test"]
        self.groups = groups or ["AD", "CN"]
        self.required_visits = required_visits or ["sc", "m06", "m12"]
        self.max_images_per_class = max_images_per_class
        self.seed = seed
        self.verbose = verbose

        # Set seed
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

    def check_prerequisites(self) -> Tuple[bool, List[str]]:
        """Check if all prerequisites are met."""
        errors = []

        if not CV2_AVAILABLE:
            errors.append("OpenCV (cv2) is required. Install with: pip install opencv-python")
        if not NUMPY_AVAILABLE:
            errors.append("NumPy is required. Install with: pip install numpy")
        if not SCIPY_AVAILABLE:
            errors.append("SciPy is required. Install with: pip install scipy")
        if not SKIMAGE_AVAILABLE:
            errors.append("scikit-image is required. Install with: pip install scikit-image")

        return len(errors) == 0, errors

    def _get_bounds(self, method: str) -> List[Tuple[float, float]]:
        """Get parameter bounds for the given method."""
        if method == "adaptive":
            return [(1, 4), (0, 0.5), (0.5, 2), (1, 3)]
        elif method == "clahe":
            return [(1, 4), (4, 16)]
        elif method == "gabor":
            return [(5, 20), (0, 180), (2, 8), (2, 8)]
        elif method == "unsharp":
            return [(0.5, 2), (1, 3)]
        else:
            raise ValueError(f"Unknown method: {method}")

    def _create_fitness_function(self, orig_img: np.ndarray, brain_mask: np.ndarray) -> callable:
        """Create fitness function for optimization."""
        def fitness(params: np.ndarray) -> float:
            try:
                if self.method == "adaptive":
                    out = ImageEnhancer.adaptive_enhancement(orig_img, *params)
                elif self.method == "clahe":
                    clip, tile = params
                    out = ImageEnhancer.clahe_enhancement(orig_img, clip, (int(tile), int(tile)))
                elif self.method == "gabor":
                    out = ImageEnhancer.gabor_enhancement(orig_img, *params)
                elif self.method == "unsharp":
                    out = ImageEnhancer.unsharp_masking(orig_img, *params)
                else:
                    raise ValueError(f"Unknown method: {self.method}")

                out[~brain_mask] = 0  # background stays black

                ent = ImageQualityMetrics.calculate_entropy(out, brain_mask)
                edge = ImageQualityMetrics.calculate_edge_energy(out, brain_mask)
                ctr = ImageQualityMetrics.calculate_local_contrast(out, brain_mask)
                return 0.4 * ent + 0.4 * edge * 100 + 0.2 * ctr * 100
            except Exception:
                return -1000.0

        return fitness

    def _apply_best_enhancement(
        self,
        orig_img: np.ndarray,
        brain_mask: np.ndarray,
        params: np.ndarray
    ) -> np.ndarray:
        """Apply enhancement with optimized parameters."""
        if self.method == "adaptive":
            out = ImageEnhancer.adaptive_enhancement(orig_img, *params)
        elif self.method == "clahe":
            clip, tile = params
            out = ImageEnhancer.clahe_enhancement(orig_img, clip, (int(tile), int(tile)))
        elif self.method == "gabor":
            out = ImageEnhancer.gabor_enhancement(orig_img, *params)
        elif self.method == "unsharp":
            out = ImageEnhancer.unsharp_masking(orig_img, *params)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        out[~brain_mask] = 0
        return out

    def enhance_single_image(
        self,
        input_path: Path,
        output_path: Path
    ) -> Dict:
        """
        Enhance a single image using GWO optimization.

        Args:
            input_path: Path to input PNG file
            output_path: Path for output PNG file

        Returns:
            Dictionary with success status, parameters, fitness, and paths
        """
        # Check if output already exists
        if output_path.exists() and output_path.stat().st_size > 0:
            return {
                "success": True,
                "skipped": True,
                "input_path": str(input_path),
                "output_path": str(output_path)
            }

        try:
            # Load image
            img = cv2.imread(str(input_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Cannot load {input_path}")

            img = img.astype(np.float32) / 255.0
            mask = get_brain_mask(img)

            # Get bounds and create fitness function
            bounds = self._get_bounds(self.method)
            fit_func = self._create_fitness_function(img, mask)

            # Run GWO optimization
            gwo = GreyWolfOptimizer(
                fit_func,
                bounds,
                self.num_wolves,
                self.gwo_iterations,
                seed=self.seed,
                convergence_threshold=1e-6
            )
            best_params, best_fitness = gwo.optimize(verbose=self.verbose)

            # Apply enhancement
            enhanced = self._apply_best_enhancement(img, mask, best_params)

            # Save output
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), enhanced)

            # Verify output
            if not output_path.exists() or output_path.stat().st_size == 0:
                return {
                    "success": False,
                    "error": "Output file is empty or missing",
                    "input_path": str(input_path),
                    "output_path": str(output_path)
                }

            return {
                "success": True,
                "skipped": False,
                "best_params": best_params.tolist(),
                "best_fitness": float(best_fitness),
                "input_path": str(input_path),
                "output_path": str(output_path)
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "input_path": str(input_path),
                "output_path": str(output_path)
            }

    def scan_input_files(self, input_root: Path) -> Tuple[List[Path], Dict]:
        """
        Scan input directory for PNG files.

        Args:
            input_root: Root directory containing PNG files

        Returns:
            Tuple of (file_paths, statistics)
            file_paths: List of input PNG file paths
            statistics: Dictionary with temporal and subject statistics
        """
        png_paths = []
        temporal_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        subject_stats = defaultdict(lambda: defaultdict(int))

        for slice_t in self.slice_types:
            for split in self.splits:
                for cls in self.groups:
                    in_dir = input_root / slice_t / split / cls
                    if not in_dir.exists():
                        continue

                    pngs = list(in_dir.glob("*.png"))

                    # Count subjects and visits
                    subjects = set()
                    visit_counts = {visit: 0 for visit in self.required_visits}

                    for png in pngs:
                        # Extract subject ID and visit from filename
                        filename = png.name
                        parts = filename.split("_")

                        if len(parts) >= 3 and parts[1] == "S":
                            subject_id = f"{parts[0]}_S_{parts[2]}"
                            subjects.add(subject_id)

                            # Extract visit
                            for visit in self.required_visits:
                                if f"_{visit}_" in filename:
                                    visit_counts[visit] += 1
                                    temporal_stats[slice_t][f"{split}_{cls}"][visit] += 1
                                    break

                    subject_stats[slice_t][f"{split}_{cls}"] = len(subjects)

                    # Apply sampling if requested
                    if self.max_images_per_class and len(pngs) > self.max_images_per_class:
                        pngs = random.sample(pngs, self.max_images_per_class)

                    # Add to file list
                    png_paths.extend(pngs)

        return png_paths, {
            "temporal_stats": dict(temporal_stats),
            "subject_stats": dict(subject_stats)
        }

