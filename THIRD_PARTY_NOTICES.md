# Third-Party Notices

This project uses and integrates with several third-party tools and libraries. This file acknowledges their contributions and provides license information.

## Tools and Libraries

### HD-BET (Brain Extraction)

- **Name**: HD-BET
- **Repository**: [https://github.com/MIC-DKFZ/HD-BET](https://github.com/MIC-DKFZ/HD-BET)
- **License**: Apache License 2.0
- **Usage**: Used for automated brain extraction (skull stripping) from MRI images
- **Citation**: Isensee, F., Schell, M., Tursunova, I., *et al.* (2019). *Automated brain extraction of multi-sequence MRI using artificial neural networks.* Human Brain Mapping. [https://doi.org/10.1002/hbm.24750](https://doi.org/10.1002/hbm.24750)

### ANTs (Advanced Normalization Tools)

- **Name**: ANTs
- **Repository**: [https://github.com/ANTsX/ANTs](https://github.com/ANTsX/ANTs)
- **License**: Apache License 2.0
- **Usage**: Used for template registration and image normalization
- **Citation**: Avants, B. B., Epstein, C. L., Grossman, M., & Gee, J. C. (2008). *Symmetric diffeomorphic image registration with cross-correlation: evaluating automated labeling of elderly and neurodegenerative brain.* Medical Image Analysis, 12(1), 26-41. [https://doi.org/10.1016/j.media.2007.06.004](https://doi.org/10.1016/j.media.2007.06.004)

### Python Dependencies

This project uses various Python packages. See `requirements.txt` for the complete list. Key dependencies include:

- **nibabel**: For NIfTI file handling
- **numpy**: For numerical operations
- **pandas**: For data manipulation
- **rich**: For terminal output formatting
- **pyyaml**: For configuration file parsing
- **Pillow**: For image processing

All Python dependencies are listed in `requirements.txt` with their respective versions. Please refer to each package's repository for their specific licenses.

## License Compliance

- All third-party tools and libraries retain their original licenses
- This project does not modify third-party code
- Users must comply with all third-party licenses when using this pipeline
- See individual repositories for full license texts

## Attribution Requirements

When publishing work that uses this pipeline, please cite:

1. **This repository** (see CITATION.cff or README Citation section)
2. **HD-BET** (if used for skull stripping)
3. **ANTs** (if used for template registration)
4. **ADNI Dataset** (if using ADNI data - follow ADNI citation requirements)

See README.md References section for complete citation information.

## Disclaimer

This project is not affiliated with, endorsed by, or sponsored by any of the third-party tools or libraries mentioned above. All trademarks and registered trademarks are the property of their respective owners.

## Questions

For questions about third-party licenses or usage, please:

1. Check the respective repository's LICENSE file
2. Contact the original maintainers
3. Open an issue in this repository for integration-specific questions

---

*Last updated: 2025-11-09*

