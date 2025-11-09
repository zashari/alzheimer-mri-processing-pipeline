# ADNI/IDA Compliance Notice

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

If this repository is useful in your work, please cite it:

```bibtex
@misc{alzheimer_mri_processing_pipeline_2025,
  title        = {alzheimer-mri-processing-pipeline},
  author       = {Zaky, Izzat and contributors},
  year         = {2025},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/zashari/alzheimer-mri-processing-pipeline}},
  note         = {Version <tag-or-commit>},
}
