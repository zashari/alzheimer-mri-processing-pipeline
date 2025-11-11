# Hippocampus ROI Mask

This directory should contain the hippocampus ROI mask file required for template registration.

## Required File

- **File name**: `hippho50.nii.gz`
- **Description**: Harvard-Oxford bilateral hippocampus mask at 50 percent threshold

## Download Instructions

The required file is **not included** in this repository due to file size constraints.

### Download from NeuroVault

1. Visit: https://neurovault.org/images/448213/
2. Click the **"Download"** button to download the hippocampus mask
3. Save the file as `hippho50.nii.gz` in this directory

### Citation

If you use this hippocampus mask, please cite:

- **NeuroVault Image**: https://identifiers.org/neurovault.image:448213
- **Collection**: Reactivation of Single-Episode Pain Patterns in the Hippocampus and Decision Making

### Additional Notes

- The mask is at 50% threshold (Harvard-Oxford atlas)
- For better visualization, you may need to adjust viewer settings to disable smoothing
- This mask is used in the template registration stage for hippocampus ROI extraction

## Verification

After downloading, verify the file exists:
```bash
ls hippho50.nii.gz
```

The file should be present before running the template registration stage.

