# MNI Brain Template

This directory should contain the MNI152 T1-weighted brain template required for template registration.

## Required File

- **File name**: `MNI152_T1_1mm_brain.nii.gz`
- **Description**: MNI152 T1-weighted 1mm brain template (skull-stripped)

## Download Instructions

The required file is **not included** in this repository due to file size constraints.

### Download from FSL Data Standard

1. Visit: https://git.fmrib.ox.ac.uk/fsl/data_standard/-/blob/master/MNI152_T1_1mm_brain.nii.gz?ref_type=heads
2. Click the **"Download"** button (or right-click and "Save As")
3. Save the file as `MNI152_T1_1mm_brain.nii.gz` in this directory

### Alternative Download Sources

The MNI152 template is also available from:

- **FSL Data Repository**: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/Atlases
- **MNI Official Website**: http://www.bic.mni.mcgill.ca/ServicesAtlases/ICBM152NLin2009

### Citation

If you use the MNI152 template, please cite:

- **FSL**: Jenkinson, M., Beckmann, C.F., Behrens, T.E., Woolrich, M.W., Smith, S.M. (2012). FSL. NeuroImage, 62:782-90
- **MNI152**: Fonov, V., Evans, A.C., Botteron, K., Almli, C.R., McKinstry, R.C., Collins, D.L. (2011). Unbiased average age-appropriate atlases for pediatric studies. NeuroImage, 54(1):313-327

### Additional Notes

- This is the standard MNI152 T1-weighted brain template at 1mm resolution
- The template is already skull-stripped (brain-only)
- This template is used in the template registration stage for spatial normalization

## Verification

After downloading, verify the file exists:
```bash
ls MNI152_T1_1mm_brain.nii.gz
```

The file should be present before running the template registration stage.

