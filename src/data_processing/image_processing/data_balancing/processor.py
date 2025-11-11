"""Data balancing processor using MRI augmentation."""

from __future__ import annotations

import json
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import cv2
    import numpy as np
    from scipy.ndimage import gaussian_filter, map_coordinates
    CV2_AVAILABLE = True
    NUMPY_AVAILABLE = True
    SCIPY_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    NUMPY_AVAILABLE = False
    SCIPY_AVAILABLE = False


def generate_alphabetical_id(index: int) -> str:
    """
    Generate 3-letter alphabetical ID (AAA, AAB, AAC, etc.).

    Args:
        index: Index to convert to alphabetical ID

    Returns:
        3-letter ID string
    """
    letters = []
    for i in range(3):
        letters.append(chr(65 + (index // (26 ** (2 - i))) % 26))
    return "".join(letters)


def get_subject_id(filename: str) -> Optional[str]:
    """
    Extract subject ID from filename.

    Args:
        filename: Filename (with or without .png extension)

    Returns:
        Subject ID string or None
    """
    # Handle augmented files: AUG_XXX_originalname.png
    if filename.startswith("AUG_"):
        parts = filename.split("_")
        if len(parts) >= 5:
            # Skip AUG and 3-letter ID, then reconstruct original subject ID
            return "_".join(parts[2:5])
    else:
        # Original format: extract subject ID (first 3 parts)
        parts = filename.split("_")
        if len(parts) >= 3 and parts[1] == "S":
            return "_".join(parts[:3])
    return None


def get_timepoint(filename: str, required_visits: List[str]) -> Optional[str]:
    """
    Extract timepoint/visit from filename.

    Args:
        filename: Filename
        required_visits: List of valid visit codes

    Returns:
        Visit code or None
    """
    # Handle both original and augmented filenames
    if filename.startswith("AUG_"):
        # AUG_XXX_002_S_0413_sc_coronal_y148.png
        parts = filename.split("_")
        if len(parts) >= 6:
            visit = parts[5]
            if visit in required_visits:
                return visit
    else:
        # 002_S_0413_sc_coronal_y148.png
        for visit in required_visits:
            if f"_{visit}_" in filename:
                return visit
    return None


class MRIAugmenter:
    """MRI-specific augmentation functions."""

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize MRI augmenter.

        Args:
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        self.seed = seed

    def elastic_deformation(self, image: np.ndarray, alpha: float = 30, sigma: float = 5) -> np.ndarray:
        """Apply elastic deformation to image."""
        shape = image.shape

        # Create random displacement fields
        dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
        dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha

        # Create meshgrid
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

        # Apply deformation
        deformed = map_coordinates(image, indices, order=1, mode="reflect")
        return deformed.reshape(shape)

    def rotation_translation(self, image: np.ndarray, angle: float, tx: float, ty: float) -> np.ndarray:
        """Apply rotation and translation."""
        rows, cols = image.shape
        center = (cols // 2, rows // 2)

        # Rotation matrix
        M_rot = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Add translation
        M_rot[0, 2] += tx
        M_rot[1, 2] += ty

        # Apply transformation
        transformed = cv2.warpAffine(
            image, M_rot, (cols, rows), borderMode=cv2.BORDER_REFLECT
        )
        return transformed

    def bias_field_simulation(self, image: np.ndarray, scale: float = 0.3) -> np.ndarray:
        """Simulate bias field artifact."""
        rows, cols = image.shape

        # Create smooth bias field
        x = np.linspace(-1, 1, cols)
        y = np.linspace(-1, 1, rows)
        X, Y = np.meshgrid(x, y)

        # Random polynomial bias field
        bias = 1 + scale * (
            np.random.randn() * X**2
            + np.random.randn() * Y**2
            + np.random.randn() * X * Y
        )

        # Apply bias field
        biased = image.astype(np.float32) * bias
        return np.clip(biased, 0, 255).astype(np.uint8)

    def motion_blur(self, image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """Apply motion blur to simulate motion artifacts."""
        # Create motion blur kernel
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
        kernel = kernel / kernel_size

        # Apply motion blur
        blurred = cv2.filter2D(image, -1, kernel)
        return blurred

    def intensity_inhomogeneity(self, image: np.ndarray, gamma_range: Tuple[float, float] = (0.8, 1.2)) -> np.ndarray:
        """Apply intensity inhomogeneity."""
        gamma = np.random.uniform(gamma_range[0], gamma_range[1])

        # Apply gamma correction
        inv_gamma = 1.0 / gamma
        table = np.array(
            [((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]
        ).astype("uint8")

        return cv2.LUT(image, table)

    def rician_noise(self, image: np.ndarray, sigma: float = 10) -> np.ndarray:
        """Add Rician noise (common in MRI)."""
        # Normalize image
        img_norm = image.astype(np.float32) / 255.0

        # Add Rician noise
        noise_real = np.random.normal(0, sigma / 255.0, image.shape)
        noise_imag = np.random.normal(0, sigma / 255.0, image.shape)

        noisy = np.sqrt((img_norm + noise_real) ** 2 + noise_imag**2)

        # Denormalize
        noisy = np.clip(noisy * 255, 0, 255).astype(np.uint8)
        return noisy

    def intensity_shift(self, image: np.ndarray, shift_range: Tuple[float, float] = (-20, 20)) -> np.ndarray:
        """Apply intensity shift to simulate scanner variability."""
        shift = np.random.uniform(shift_range[0], shift_range[1])
        shifted = image.astype(np.float32) + shift
        return np.clip(shifted, 0, 255).astype(np.uint8)


def generate_augmentation_params(
    rotation_range: Tuple[float, float] = (-10, 10),
    translation_range: Tuple[float, float] = (-10, 10),
    elastic_alpha_range: Tuple[float, float] = (20, 40),
    elastic_sigma_range: Tuple[float, float] = (4, 6),
    bias_scale_range: Tuple[float, float] = (0.2, 0.4),
    motion_kernels: List[int] = [3, 5, 7],
    gamma_range: Tuple[float, float] = (0.8, 1.2),
    noise_sigma_range: Tuple[float, float] = (5, 15),
    intensity_shift_range: Tuple[float, float] = (-15, 15),
    motion_probability: float = 0.5,
    elastic_probability: float = 0.7
) -> Dict:
    """
    Generate random augmentation parameters.

    Args:
        rotation_range: Rotation angle range in degrees
        translation_range: Translation range in pixels
        elastic_alpha_range: Elastic deformation alpha range
        elastic_sigma_range: Elastic deformation sigma range
        bias_scale_range: Bias field scale range
        motion_kernels: Available motion blur kernel sizes
        gamma_range: Gamma correction range
        noise_sigma_range: Rician noise sigma range
        intensity_shift_range: Intensity shift range
        motion_probability: Probability of applying motion blur
        elastic_probability: Probability of applying elastic deformation

    Returns:
        Dictionary of augmentation parameters
    """
    params = {
        "rotation_angle": float(np.random.uniform(rotation_range[0], rotation_range[1])),
        "translation_x": float(np.random.uniform(translation_range[0], translation_range[1])),
        "translation_y": float(np.random.uniform(translation_range[0], translation_range[1])),
        "elastic_alpha": float(np.random.uniform(elastic_alpha_range[0], elastic_alpha_range[1])),
        "elastic_sigma": float(np.random.uniform(elastic_sigma_range[0], elastic_sigma_range[1])),
        "bias_scale": float(np.random.uniform(bias_scale_range[0], bias_scale_range[1])),
        "motion_kernel": int(np.random.choice(motion_kernels)),
        "gamma": float(np.random.uniform(gamma_range[0], gamma_range[1])),
        "noise_sigma": float(np.random.uniform(noise_sigma_range[0], noise_sigma_range[1])),
        "intensity_shift": float(np.random.uniform(intensity_shift_range[0], intensity_shift_range[1])),
        "apply_motion": bool(np.random.random() > (1 - motion_probability)),
        "apply_elastic": bool(np.random.random() > (1 - elastic_probability)),
    }
    return params


def augment_image(image: np.ndarray, params: Dict, augmenter: MRIAugmenter) -> np.ndarray:
    """
    Apply augmentations to a single image using given parameters.

    Args:
        image: Input image array
        params: Augmentation parameters dictionary
        augmenter: MRIAugmenter instance

    Returns:
        Augmented image array
    """
    # Always apply rotation and translation
    augmented = augmenter.rotation_translation(
        image,
        params["rotation_angle"],
        params["translation_x"],
        params["translation_y"],
    )

    # Apply elastic deformation (if enabled)
    if params["apply_elastic"]:
        augmented = augmenter.elastic_deformation(
            augmented, alpha=params["elastic_alpha"], sigma=params["elastic_sigma"]
        )

    # Apply bias field
    augmented = augmenter.bias_field_simulation(augmented, scale=params["bias_scale"])

    # Apply motion blur (if enabled)
    if params["apply_motion"]:
        augmented = augmenter.motion_blur(
            augmented, kernel_size=params["motion_kernel"]
        )

    # Apply intensity inhomogeneity
    augmented = augmenter.intensity_inhomogeneity(
        augmented, gamma_range=(params["gamma"], params["gamma"])
    )

    # Apply Rician noise
    augmented = augmenter.rician_noise(augmented, sigma=params["noise_sigma"])

    # Apply intensity shift
    augmented = augmenter.intensity_shift(
        augmented, shift_range=(params["intensity_shift"], params["intensity_shift"])
    )

    return augmented


def convert_numpy_types(obj) -> any:
    """
    Convert numpy types to native Python types for JSON serialization.

    Args:
        obj: Object that may contain numpy types

    Returns:
        Object with numpy types converted to native Python types
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


class DataBalancingProcessor:
    """Processor for data balancing using MRI augmentation."""

    def __init__(
        self,
        slice_types: List[str],
        groups: List[str],
        required_visits: List[str],
        augmentation_targets: Dict[str, Dict[str, int]],
        augmentation_params: Optional[Dict] = None,
        seed: Optional[int] = None,
        verbose: bool = False
    ):
        """
        Initialize data balancing processor.

        Args:
            slice_types: List of slice types to process
            groups: List of groups/classes to process
            required_visits: List of required visits
            augmentation_targets: Dict mapping group -> {"current": int, "target": int}
            augmentation_params: Optional augmentation parameter ranges (from config)
            seed: Random seed for reproducibility
            verbose: Show detailed output
        """
        self.slice_types = slice_types
        self.groups = groups
        self.required_visits = required_visits
        self.augmentation_targets = augmentation_targets
        self.augmentation_params = augmentation_params or {}
        self.seed = seed
        self.verbose = verbose

        # Set seed
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # Initialize augmenter
        self.augmenter = MRIAugmenter(seed=seed)

    def check_prerequisites(self) -> Tuple[bool, List[str]]:
        """Check if all prerequisites are met."""
        errors = []

        if not CV2_AVAILABLE:
            errors.append("OpenCV (cv2) is required. Install with: pip install opencv-python")
        if not NUMPY_AVAILABLE:
            errors.append("NumPy is required. Install with: pip install numpy")
        if not SCIPY_AVAILABLE:
            errors.append("SciPy is required. Install with: pip install scipy")

        return len(errors) == 0, errors

    def organize_subjects_by_class(self, class_path: Path) -> Dict[str, List[str]]:
        """
        Organize images by subject.

        Args:
            class_path: Path to class directory containing PNG files

        Returns:
            Dictionary mapping subject_id -> list of filenames (sorted by timepoint)
        """
        subjects = defaultdict(list)

        if not class_path.exists():
            return {}

        for img_file in class_path.glob("*.png"):
            filename = img_file.name
            subject_id = get_subject_id(filename)
            if subject_id:
                subjects[subject_id].append(filename)

        # Sort images within each subject by timepoint
        for subject_id in subjects:
            subjects[subject_id].sort(
                key=lambda x: get_timepoint(x, self.required_visits) or ""
            )

        return dict(subjects)

    def generate_augmentation_params_for_class(self) -> Dict:
        """Generate augmentation parameters using configured ranges."""
        return generate_augmentation_params(
            rotation_range=tuple(self.augmentation_params.get("rotation_range", [-10, 10])),
            translation_range=tuple(self.augmentation_params.get("translation_range", [-10, 10])),
            elastic_alpha_range=tuple(self.augmentation_params.get("elastic_alpha_range", [20, 40])),
            elastic_sigma_range=tuple(self.augmentation_params.get("elastic_sigma_range", [4, 6])),
            bias_scale_range=tuple(self.augmentation_params.get("bias_scale_range", [0.2, 0.4])),
            motion_kernels=self.augmentation_params.get("motion_kernels", [3, 5, 7]),
            gamma_range=tuple(self.augmentation_params.get("gamma_range", [0.8, 1.2])),
            noise_sigma_range=tuple(self.augmentation_params.get("noise_sigma_range", [5, 15])),
            intensity_shift_range=tuple(self.augmentation_params.get("intensity_shift_range", [-15, 15])),
            motion_probability=self.augmentation_params.get("motion_probability", 0.5),
            elastic_probability=self.augmentation_params.get("elastic_probability", 0.7)
        )

    def augment_class(
        self,
        input_class_path: Path,
        output_class_path: Path,
        class_name: str,
        augmentation_params_dict: Optional[Dict] = None
    ) -> Dict:
        """
        Augment a single class to reach target subject count.

        Args:
            input_class_path: Input directory for the class
            output_class_path: Output directory for the class
            class_name: Name of the class (e.g., "AD", "CN")
            augmentation_params_dict: Pre-generated augmentation parameters (for consistency across planes)

        Returns:
            Dictionary with augmentation statistics
        """
        output_class_path.mkdir(parents=True, exist_ok=True)

        # Get current subjects
        subjects = self.organize_subjects_by_class(input_class_path)
        current_count = len(subjects)
        target_count = self.augmentation_targets.get(class_name, {}).get("target", current_count)

        # Copy all original images first
        for subject_id, images in subjects.items():
            for img_file in images:
                src = input_class_path / img_file
                dst = output_class_path / img_file
                if not dst.exists():
                    shutil.copy2(src, dst)

        # Calculate how many augmented subjects needed
        subjects_to_augment = max(0, target_count - current_count)
        augmentation_params_dict = augmentation_params_dict or {}

        if subjects_to_augment > 0:
            # Select subjects to augment (randomly sample with replacement)
            subject_ids = list(subjects.keys())
            selected_subjects = np.random.choice(
                subject_ids, size=subjects_to_augment, replace=True
            )

            # Generate augmentation parameters if not provided
            if not augmentation_params_dict:
                for i, source_subject in enumerate(selected_subjects):
                    alpha_id = generate_alphabetical_id(i)
                    augmentation_params_dict[alpha_id] = {
                        "source": source_subject,
                        "params": self.generate_augmentation_params_for_class(),
                    }

            # Apply augmentations
            for alpha_id, aug_info in augmentation_params_dict.items():
                source_subject = aug_info["source"]
                params = aug_info["params"]

                # Process each timepoint for this subject
                for img_file in subjects[source_subject]:
                    # Load image
                    img_path = input_class_path / img_file
                    image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

                    if image is None:
                        continue

                    # Apply augmentation
                    augmented = augment_image(image, params, self.augmenter)

                    # Save with new name format: AUG_XXX_originalfilename
                    new_filename = f"AUG_{alpha_id}_{img_file}"
                    cv2.imwrite(str(output_class_path / new_filename), augmented)

        return {
            "original_subjects": current_count,
            "augmented_subjects": subjects_to_augment,
            "total_subjects": target_count,
            "augmentation_params": augmentation_params_dict
        }

    def copy_non_train_sets(self, input_root: Path, output_root: Path) -> Dict:
        """
        Copy validation and test sets without augmentation.

        Args:
            input_root: Root input directory
            output_root: Root output directory

        Returns:
            Dictionary with copy statistics
        """
        stats = {"copied": 0, "skipped": 0}

        for slice_type in self.slice_types:
            for split in ["val", "test"]:
                for group in self.groups:
                    input_path = input_root / slice_type / split / group
                    output_path = output_root / slice_type / split / group

                    if input_path.exists():
                        output_path.mkdir(parents=True, exist_ok=True)

                        # Copy all files
                        for file in input_path.glob("*.png"):
                            dst = output_path / file.name
                            if not dst.exists():
                                shutil.copy2(file, dst)
                                stats["copied"] += 1
                            else:
                                stats["skipped"] += 1

        return stats

