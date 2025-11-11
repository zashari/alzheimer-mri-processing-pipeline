[![License](https://img.shields.io/badge/license-MIT-informational)](./LICENSE)
[![Status](https://img.shields.io/badge/status-WIP-%23ffaa00)](#)
[![Cite](https://img.shields.io/badge/cite-CITATION.cff-blue)](./CITATION.cff)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](#)

## Table of Contents

- [ADNI/IDA Compliance Notice](#adniida-compliance-notice)
  - [Scope of This Repository](#scope-of-this-repository)
  - [Requirements for ADNI Data Users](#requirements-for-adni-data-users)
  - [Repository Compliance Statement](#repository-compliance-statement)
  - [Resources](#resources)
- [About This Project](#about-this-project)
- [What's Included](#whats-included)
- [Platforms](#platforms)
- [Getting Started](#getting-started)
- [Contributing](#contributing)
- [Citation](#citation)
  - [Quick Citation](#quick-citation)
  - [Citation File](#citation-file)
  - [References](#references)
- [Acknowledgments](#acknowledgments)
- [License](#license)

---

# ADNI/IDA Compliance Notice

> **Project status:** Stable release. Current version: v1.0.0.

> **This project is an independent, open-source effort. It is not affiliated with, endorsed by, or sponsored by the Alzheimer’s Disease Neuroimaging Initiative ([ADNI](https://adni.loni.usc.edu/)) or the Imaging Data Archive ([IDA](https://ida.loni.usc.edu/)) operated by the Laboratory of Neuro Imaging ([LONI](https://loni.usc.edu/)). “ADNI” and “IDA-LONI” are trademarks of their respective owners and are used solely to indicate data compatibility.**
>
> **This repository contains code only. It does not host or distribute ADNI data. Access to ADNI/IDA is governed by their Data Use Agreements (DUAs). Users are solely responsible for compliance with all applicable terms.**
>
> **Before requesting data access, read the [ADNI Data Use Agreement](https://ida.loni.usc.edu/collaboration/access/appLicense.jsp) and then follow the [ADNI data access instructions](https://adni.loni.usc.edu/data-samples/adni-data/#AccessData).**

---

## Scope of This Repository

### Included
- **Processing framework and utilities** (code only)
- **Configuration templates** for minor experiments
- **Documentation and minimal examples** for educational purposes

### Explicitly Excluded
- ❌ ADNI participant-level data  
- ❌ ADNI-derived datasets or processed outputs  
- ❌ Participant identifiers or metadata

---

## Requirements for ADNI Data Users

If you use this pipeline with **ADNI data**, you must comply with the ADNI DUA:  <https://ida.loni.usc.edu/collaboration/access/appLicense.jsp>

---

## Repository Compliance Statement

This project is designed to support DUA compliance:
- ✅ **Code only**; no data included
- ✅ No participant-level information
- ✅ No derived datasets
- ✅ `.gitignore` excludes data artifacts (e.g., `*.nii`, `*.nii.gz`, `outputs/`, `.reports/`)
- ✅ Prominent disclaimers and usage guidance

---

## Resources

- **ADNI Website:** <https://adni.loni.usc.edu/>
- **ADNI Data Use Agreement:** <https://ida.loni.usc.edu/collaboration/access/appLicense.jsp>
- **IDA-LONI Access Portal:** <https://ida.loni.usc.edu/>
- **ADNI Publication & Citation Guidelines:** <https://adni.loni.usc.edu/wp-content/uploads/how_to_apply/ADNI_Manuscript_Citations.pdf>

**By using this pipeline with ADNI data, you confirm that you have read, understood, and agree to comply with all terms of the ADNI Data Use Agreement.**

---

## About This Project

This repository provides a **complete 3D NIfTI preprocessing pipeline** for **ADNI T1-weighted MRI** to accelerate experimentation in Alzheimer’s disease research. The pipeline is designed for **Windows** (primary development target) and is expected to run on **Unix-based systems**; portability improvements are welcome via issues and pull requests.

The initial workflow was developed in **early 2025** over several months of research, experimentation, and iteration as part of a **final-year thesis**. The aim is to reduce time spent on data wrangling so you can focus on modeling.

> **Important:** This repository contains **code only**. It does **not** host or distribute ADNI data. Users are responsible for obtaining authorized access and complying with the ADNI/IDA DUA.

---

## What’s Included

- End-to-end **3D NIfTI preprocessing pipeline** (Python-only implementation)
- Modular, reusable components with clear, minimal APIs
- **Configuration templates** for small-scale experiments
- **Documentation and minimal examples** to get started quickly

If you prefer a **Jupyter Notebook (.ipynb)** walkthrough, contact **izzat.zaky@gmail.com**.

---

## Platforms

- **Windows** — primary development and testing platform  
- **Unix-based systems (Linux/macOS)** — expected support; please report and/or fix portability issues

---

## Getting Started

1. Obtain ADNI access and review the **ADNI DUA**.  
2. Follow the **README** for environment setup and path configuration.  
3. Execute pipeline modules step by step or integrate them into your training workflow.

---

## Contributing

Contributions are welcome. Typical flow:
1. Fork and clone the repository  
2. Create a feature branch  
3. Implement changes (and update docs/tests where helpful)  
4. Open a Pull Request with rationale, approach, and any assumptions

---

## Citation

If you use this pipeline in your research, please cite it appropriately. This helps track the impact of this work and supports future development.

### Quick Citation

**BibTeX format:**
```bibtex
@misc{alzheimer_mri_processing_pipeline_2025,
  title        = {alzheimer-mri-processing-pipeline},
  author       = {Ashari, Zaky and contributors},
  year         = {2025},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/zashari/alzheimer-mri-processing-pipeline}},
  note         = {Version v1.0.0}
}
```

**APA format:**
```
Ashari, Z. (2025). alzheimer-mri-processing-pipeline [Computer software]. GitHub. 
https://github.com/zashari/alzheimer-mri-processing-pipeline
```

### Citation File

- **[`CITATION.cff`](CITATION.cff)** — Citation File Format for automatic citation (GitHub will display a "Cite this repository" button)

### References

This pipeline is built upon and informed by the following research papers and methods. When publishing work that uses this pipeline, please cite the relevant references:

#### Skull Stripping
- Druzhinina, P., & Kondrateva, E. (2022). *The effect of skull-stripping on transfer learning for 3D MRI models: ADNI data.* Medical Imaging with Deep Learning (MIDL). [https://openreview.net/forum?id=IS1yeyiAFZS](https://openreview.net/forum?id=IS1yeyiAFZS)
- Isensee, F., Schell, M., Tursunova, I., *et al.* (2019). *Automated brain extraction of multi-sequence MRI using artificial neural networks.* Human Brain Mapping. [https://doi.org/10.1002/hbm.24750](https://doi.org/10.1002/hbm.24750) — **HD-BET tool**: [https://github.com/MIC-DKFZ/HD-BET](https://github.com/MIC-DKFZ/HD-BET)

#### Alzheimer's Disease ROI
- Hassouneh, A., Bazuin, B., Danna-Dos-Santos, A., Acar, I., Abdel-Qader, I., & ADNI. (2024). *Feature Importance Analysis and Machine Learning for Alzheimer's Disease Early Detection: Feature Fusion of the Hippocampus, Entorhinal Cortex, and Standardized Uptake Value Ratio.* [https://doi.org/10.1159/000538486](https://doi.org/10.1159/000538486)

#### Data Augmentation / Domain Adaptation
- Llambias, S. N., Nielsen, M., & Mehdipour Ghazi, M. (2023). *Data Augmentation-Based Unsupervised Domain Adaptation In Medical Imaging.* arXiv preprint arXiv:2308.04395. [https://doi.org/10.48550/arXiv.2308.04395](https://doi.org/10.48550/arXiv.2308.04395)

#### Image Enhancement / Optimization
- Mirjalili, S., Mirjalili, S. M., & Lewis, A. (2014). *Grey Wolf Optimizer.* Advances in Engineering Software, 69, 46–61. [https://doi.org/10.1016/j.advengsoft.2013.12.007](https://doi.org/10.1016/j.advengsoft.2013.12.007)

#### Dataset
- **ADNI Dataset**: If using ADNI data, follow [ADNI citation requirements](https://adni.loni.usc.edu/wp-content/uploads/how_to_apply/ADNI_Manuscript_Citations.pdf)
- Imaging Data Archive (IDA) at LONI: [https://ida.loni.usc.edu/](https://ida.loni.usc.edu/)

---

## Acknowledgments

This work was developed as part of a final-year thesis project. Special thanks to:

- **Dr. Dani Suandi, S.Si., M.Si.** — Lecturer in Mathematics, Binus University; Lecturer, School of Computer Science, Binus University — for guidance and supervision throughout this project.
  - [Google Scholar](https://scholar.google.com/citations?user=LKUVKGEAAAAJ&hl=id)
  - [ResearchGate](https://www.researchgate.net/profile/Dani-Suandi)

---

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

**Third-party dependencies** (e.g., HD-BET, ANTs) have their own licenses. Please review and comply with their respective license terms. See [docs/THIRD_PARTY_NOTICES.md](docs/THIRD_PARTY_NOTICES.md) for details.
