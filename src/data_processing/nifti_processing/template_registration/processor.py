"""ANTs-based template registration processor."""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import ants
    ANTS_AVAILABLE = True
except ImportError:
    ANTS_AVAILABLE = False

try:
    import nibabel as nib
    import numpy as np
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False


class RegistrationProcessor:
    """Handles ANTs-based registration to MNI template and hippocampus ROI warping."""

    def __init__(
        self,
        mni_template_path: Path,
        hippocampus_roi_path: Path,
        registration_type: str = "SyNAggro",
        num_threads: int = 8,
        verbose: bool = False,
        min_hippo_volume: int = 100
    ):
        """
        Initialize registration processor.

        Args:
            mni_template_path: Path to MNI template
            hippocampus_roi_path: Path to hippocampus ROI mask
            registration_type: ANTs registration type
            num_threads: Number of threads for ANTs
            verbose: Show detailed output
            min_hippo_volume: Minimum hippocampus volume threshold
        """
        self.mni_template_path = Path(mni_template_path)
        self.hippocampus_roi_path = Path(hippocampus_roi_path)
        self.registration_type = registration_type
        self.num_threads = num_threads
        self.verbose = verbose
        self.min_hippo_volume = min_hippo_volume

        # Create temp directory for intermediate files
        self.temp_dir = Path(tempfile.gettempdir()) / "ants_registration"
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        # Set ANTs environment
        os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = str(num_threads)

    def check_prerequisites(self) -> Tuple[bool, List[str]]:
        """
        Check if all prerequisites are met.

        Returns:
            (success, list of error messages)
        """
        errors = []

        if not ANTS_AVAILABLE:
            errors.append("ANTs Python package not installed. Install with: pip install antspyx")

        if not NIBABEL_AVAILABLE:
            errors.append("Nibabel not installed. Install with: pip install nibabel")

        if not self.mni_template_path.exists():
            errors.append(f"MNI template not found: {self.mni_template_path}")

        if not self.hippocampus_roi_path.exists():
            errors.append(f"Hippocampus ROI not found: {self.hippocampus_roi_path}")

        return len(errors) == 0, errors

    def process_subject(
        self,
        brain_path: Path,
        output_brain_slices: Dict[str, Path],
        output_mask_3d: Optional[Path] = None,
        cleanup_temp: bool = True
    ) -> Dict:
        """
        Process a single subject: register to MNI, warp ROI, extract optimal slices.

        Args:
            brain_path: Path to skull-stripped brain
            output_brain_slices: Dict of plane->output path for brain slices
            output_mask_3d: Optional path to save 3D hippocampus mask
            cleanup_temp: Clean up temporary files

        Returns:
            Result dictionary with status and details
        """
        result = {
            'status': 'pending',
            'error': None,
            'slices': {},
            'hippo_mask_3d': None,
            'hippo_volume': 0,
            'processing_time': 0,
            'registration_quality': None
        }

        start_time = time.time()
        temp_files = []

        try:
            # Load and reorient brain to RAS
            brain_nifti = self._load_and_reorient(brain_path)
            brain_data = brain_nifti.get_fdata()

            # Convert to ANTs image
            brain_ants = ants.from_numpy(
                brain_data,
                spacing=brain_nifti.header.get_zooms()[:3]
            )

            # Load templates
            mni_template = ants.image_read(str(self.mni_template_path))
            hippo_roi = ants.image_read(str(self.hippocampus_roi_path))

            # Perform registration to MNI space
            registration_params = {
                "type_of_transform": self.registration_type,
                "num_threads": self.num_threads,
                "verbose": self.verbose
            }

            registration = ants.registration(
                fixed=mni_template,
                moving=brain_ants,
                **registration_params
            )

            # Get transforms
            forward_transforms = registration['fwdtransforms']
            inverse_transforms = registration['invtransforms']
            temp_files.extend(forward_transforms)
            temp_files.extend(inverse_transforms)

            # Calculate registration quality (optional)
            if 'warpedfixout' in registration:
                warped = registration['warpedfixout']
                result['registration_quality'] = float(
                    ants.image_similarity(mni_template, warped, metric_type='MI')
                )

            # Warp hippocampus ROI from MNI to subject space
            hippo_in_subject = ants.apply_transforms(
                fixed=brain_ants,
                moving=hippo_roi,
                transformlist=inverse_transforms,
                interpolator='nearestNeighbor'
            )

            # Convert to numpy and ensure binary mask
            hippo_mask = hippo_in_subject.numpy()
            hippo_mask = (hippo_mask > 0.5).astype(np.float32)

            # Check hippocampus volume
            hippo_volume = np.sum(hippo_mask)
            result['hippo_volume'] = float(hippo_volume)

            if hippo_volume < self.min_hippo_volume:
                result['status'] = 'error'
                result['error'] = f'Hippocampus volume too small: {hippo_volume:.0f} voxels'
                return result

            # Save 3D hippocampus mask if requested
            if output_mask_3d:
                output_mask_3d.parent.mkdir(parents=True, exist_ok=True)
                self._save_3d_mask(hippo_mask, brain_nifti.affine, output_mask_3d)
                result['hippo_mask_3d'] = str(output_mask_3d)

            # Extract optimal slices for each plane
            for plane, output_path in output_brain_slices.items():
                # Find optimal slice (max hippocampus area)
                slice_idx, area = self._find_optimal_slice(hippo_mask, plane)

                # Extract brain slice
                brain_slice = self._extract_slice(brain_data, slice_idx, plane)

                # Save slice
                output_path.parent.mkdir(parents=True, exist_ok=True)
                self._save_slice_as_nifti(
                    brain_slice,
                    brain_nifti.affine,
                    slice_idx,
                    plane,
                    output_path
                )

                result['slices'][plane] = {
                    'slice_idx': int(slice_idx),
                    'hippo_area': float(area),
                    'output_path': str(output_path)
                }

            result['status'] = 'success'

        except Exception as e:
            result['status'] = 'error'
            result['error'] = str(e)

        finally:
            # Clean up temporary files
            if cleanup_temp:
                for temp_file in temp_files:
                    try:
                        Path(temp_file).unlink()
                    except:
                        pass

            result['processing_time'] = time.time() - start_time

        return result

    def _load_and_reorient(self, filepath: Path) -> nib.Nifti1Image:
        """Load NIfTI file and reorient to RAS."""
        img = nib.load(filepath)
        return self._reorient_to_ras(img)

    def _reorient_to_ras(self, image: nib.Nifti1Image) -> nib.Nifti1Image:
        """Reorient a nibabel image to RAS orientation."""
        orig_ornt = nib.io_orientation(image.affine)
        ras_ornt = nib.orientations.axcodes2ornt(('R', 'A', 'S'))
        ornt_trans = nib.orientations.ornt_transform(orig_ornt, ras_ornt)

        if not np.array_equal(orig_ornt, ras_ornt):
            data = nib.orientations.apply_orientation(
                image.get_fdata(), ornt_trans
            )
            affine = image.affine @ nib.orientations.inv_ornt_aff(
                ornt_trans, image.shape
            )
            reoriented_img = nib.Nifti1Image(data, affine, image.header)
            return reoriented_img

        return image

    def _find_optimal_slice(self, mask_data: np.ndarray, plane: str) -> Tuple[int, float]:
        """Find the slice with maximum hippocampus area in the specified plane."""
        if plane == 'axial':
            areas = np.sum(mask_data, axis=(0, 1))
        elif plane == 'sagittal':
            areas = np.sum(mask_data, axis=(1, 2))
        elif plane == 'coronal':
            areas = np.sum(mask_data, axis=(0, 2))
        else:
            raise ValueError(f"Invalid plane: {plane}")

        slice_idx = np.argmax(areas)
        max_area = areas[slice_idx]

        return int(slice_idx), float(max_area)

    def _extract_slice(self, image_data: np.ndarray, slice_idx: int, plane: str) -> np.ndarray:
        """Extract a 2D slice from 3D volume."""
        if plane == 'axial':
            return image_data[:, :, slice_idx]
        elif plane == 'sagittal':
            return image_data[slice_idx, :, :]
        elif plane == 'coronal':
            return image_data[:, slice_idx, :]
        else:
            raise ValueError(f"Invalid plane: {plane}")

    def _save_slice_as_nifti(
        self,
        slice_2d: np.ndarray,
        original_affine: np.ndarray,
        slice_idx: int,
        plane: str,
        output_path: Path
    ):
        """Save a 2D slice as a NIfTI file with proper header information."""
        # Add dimension for NIfTI format
        if plane == 'axial':
            slice_3d = slice_2d[:, :, np.newaxis]
        elif plane == 'sagittal':
            slice_3d = slice_2d[np.newaxis, :, :]
        elif plane == 'coronal':
            slice_3d = slice_2d[:, np.newaxis, :]

        nifti_img = nib.Nifti1Image(slice_3d, original_affine)
        nib.save(nifti_img, str(output_path))

    def _save_3d_mask(
        self,
        mask_3d: np.ndarray,
        original_affine: np.ndarray,
        output_path: Path
    ):
        """Save a 3D mask as a NIfTI file."""
        nifti_img = nib.Nifti1Image(mask_3d, original_affine)
        nib.save(nifti_img, str(output_path))