## 📚 References & Acknowledgments

**How to cite this repository**

If you use this code, please cite:

```
@misc{your_repo_2025,
  title        = {<Your Project Title>},
  author       = {<Your Name> and <Co-authors>},
  year         = {2025},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/<user>/<repo>}},
  note         = {Version <tag/commit>},
}
```

**References used in this work**

1. **Skull-stripping**

* Druzhinina P, Kondrateva E. *The effect of skull-stripping on transfer learning for 3D MRI models: ADNI data.* MIDL 2022. [https://openreview.net/forum?id=IS1yeyiAFZS](https://openreview.net/forum?id=IS1yeyiAFZS)
* MIC-DKFZ **HD-BET**: [https://github.com/MIC-DKFZ/HD-BET](https://github.com/MIC-DKFZ/HD-BET)
  Isensee F, Schell M, Tursunova I, *et al.* *Automated brain extraction of multi-sequence MRI using artificial neural networks.* **Human Brain Mapping** (2019). [https://doi.org/10.1002/hbm.24750](https://doi.org/10.1002/hbm.24750)

2. **Alzheimer’s Disease ROI**

* Hassouneh A, Bazuin B, Danna-Dos-Santos A, Acar I, Abdel-Qader I; ADNI. *Feature Importance Analysis and Machine Learning for Alzheimer's Disease Early Detection: Feature Fusion of the Hippocampus, Entorhinal Cortex, and Standardized Uptake Value Ratio.* DOI: 10.1159/000538486 (PMID: 38650695; PMCID: PMC11034932).

3. **Data Augmentation / Domain Adaptation**

* Llambias SN, Nielsen M, Mehdipour Ghazi M. *Data Augmentation-Based Unsupervised Domain Adaptation In Medical Imaging.* arXiv:2308.04395 (2023). [https://doi.org/10.48550/arXiv.2308.04395](https://doi.org/10.48550/arXiv.2308.04395)

4. **Image Enhancement / Optimization**

* Mirjalili S, Mirjalili SM, Lewis A. *Grey Wolf Optimizer.* **Advances in Engineering Software** 69:46–61 (2014). [https://doi.org/10.1016/j.advengsoft.2013.12.007](https://doi.org/10.1016/j.advengsoft.2013.12.007)

5. **Dataset**

* Imaging Data Archive (IDA) at LONI: [https://ida.loni.usc.edu/](https://ida.loni.usc.edu/)
  *(ADNI data accessed via IDA.)*

**Acknowledgments**

* With gratitude to **Dr. Dani Suandi, S.Si., M.Si.** (Lecturer in Mathematics, Binus University; Lecturer, School of Computer Science, Universitas Bina Nusantara) for guidance and supervision.
  Google Scholar: [https://scholar.google.com/citations?user=LKUVKGEAAAAJ&hl=id](https://scholar.google.com/citations?user=LKUVKGEAAAAJ&hl=id) — ResearchGate: [https://www.researchgate.net/profile/Dani-Suandi](https://www.researchgate.net/profile/Dani-Suandi)

**Third-party code notice**

This repository may depend on third-party tools (e.g., HD-BET). Please consult and comply with their respective licenses in their repositories.

---

## 🧾 `references.bib` (BibTeX)

```bibtex
@inproceedings{druzhinina2022the,
  title        = {The effect of skull-stripping on transfer learning for 3D {MRI} models: {ADNI} data},
  author       = {Druzhinina, Polina and Kondrateva, Ekaterina},
  booktitle    = {Medical Imaging with Deep Learning (MIDL) -- Short Papers},
  year         = {2022},
  url          = {https://openreview.net/forum?id=IS1yeyiAFZS},
  keywords     = {3D CNN, ADNI, skull-stripping, interpretability, transfer learning}
}

@article{isensee2019hdbet,
  title        = {Automated brain extraction of multi-sequence MRI using artificial neural networks},
  author       = {Isensee, Fabian and Schell, M and Tursunova, I and Brugnara, G and Bonekamp, D and Neuberger, U and Wick, A and Schlemmer, H.-P. and Heiland, S and Wick, W and Bendszus, M and Maier-Hein, K. H. and Kickingereder, P},
  journal      = {Human Brain Mapping},
  year         = {2019},
  pages        = {1--13},
  doi          = {10.1002/hbm.24750},
  note         = {Early View},
  url          = {https://doi.org/10.1002/hbm.24750}
}

@article{hassouneh2024feature,
  title        = {Feature Importance Analysis and Machine Learning for Alzheimer's Disease Early Detection: Feature Fusion of the Hippocampus, Entorhinal Cortex, and Standardized Uptake Value Ratio},
  author       = {Hassouneh, Aya and Bazuin, Bradley and Danna-Dos-Santos, Alessander and Acar, Ilgin and Abdel-Qader, Ikhlas and Alzheimer's Disease Neuroimaging Initiative},
  year         = {2024},
  doi          = {10.1159/000538486},
  pmid         = {38650695},
  pmcid        = {PMC11034932},
  note         = {Journal info not specified in source; see DOI for publisher record}
}

@article{llambias2023data,
  title        = {Data Augmentation-Based Unsupervised Domain Adaptation In Medical Imaging},
  author       = {Llambias, Sebastian N{\o}rgaard and Nielsen, Mads and Mehdipour Ghazi, Mostafa},
  journal      = {arXiv preprint arXiv:2308.04395},
  year         = {2023},
  doi          = {10.48550/arXiv.2308.04395},
  url          = {https://arxiv.org/abs/2308.04395},
  archivePrefix= {arXiv},
  eprint       = {2308.04395},
  primaryClass = {eess.IV}
}

@article{mirjalili2014gwo,
  title        = {Grey Wolf Optimizer},
  author       = {Mirjalili, Seyedali and Mirjalili, Seyed Mohammad and Lewis, Andrew},
  journal      = {Advances in Engineering Software},
  volume       = {69},
  pages        = {46--61},
  year         = {2014},
  issn         = {0965-9978},
  doi          = {10.1016/j.advengsoft.2013.12.007},
  url          = {https://www.sciencedirect.com/science/article/pii/S0965997813001853}
}

@misc{adni_ida_usc,
  title        = {Alzheimer's Disease Neuroimaging Initiative (ADNI) via the LONI Imaging Data Archive (IDA)},
  howpublished = {\url{https://ida.loni.usc.edu/}},
  note         = {Data used in this research were obtained from the IDA at the USC Laboratory of Neuro Imaging. Access subject to IDA/ADNI data use policies.},
  year         = {2025}
}

@misc{suandi_ack_2025,
  title        = {Acknowledgment of supervision and guidance},
  author       = {Suandi, Dani},
  year         = {2025},
  note         = {Dr. Dani Suandi, S.Si., M.Si., Lecturer in Mathematics at Binus University; Lecturer in School of Computer Science, Universitas Bina Nusantara. Scholar: \url{https://scholar.google.com/citations?user=LKUVKGEAAAAJ\&hl=id}; ResearchGate: \url{https://www.researchgate.net/profile/Dani-Suandi}}
}

@misc{hdbet_code,
  title        = {HD-BET: Brain extraction tool},
  author       = {MIC-DKFZ},
  year         = {2019},
  howpublished = {\url{https://github.com/MIC-DKFZ/HD-BET}},
  note         = {Code repository; see repo for license}
}
```

---

## 🔖 Optional: `CITATION.cff` (GitHub’s citation badge)

Create a `CITATION.cff` in your repo root for nice “Cite this repository” support:

```yaml
cff-version: 1.2.0
message: "If you use this software, please cite it as below."
title: "<Your Project Title>"
version: "<tag/commit>"
date-released: "2025-11-05"
repository-code: "https://github.com/<user>/<repo>"
authors:
  - family-names: "<LastName>"
    given-names: "<FirstName>"
  - name: "and contributors"
preferred-citation:
  type: misc
  title: "<Your Project Title>"
  authors:
    - family-names: "<LastName>"
      given-names: "<FirstName>"
  year: 2025
  notes: "Version <tag/commit>. See README for third-party references."
```

---

### Quick tips

* Keep both the **Markdown References** section and the **`references.bib`** in sync.
* If you vendor or call external tools (e.g., HD-BET), include their LICENSE files or link to them, and state their licenses in a `NOTICE` section if required.