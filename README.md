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

If you use this pipeline with **ADNI data**, you must comply with the ADNI DUA:  
<https://ida.loni.usc.edu/collaboration/access/appLicense.jsp>

1. **Data Access**
   - Obtain ADNI data directly through the official ADNI/IDA portal. This repository does not provide access to ADNI data.

2. **Use of AI/Cloud Tools**
   - **Do not** upload ADNI data to public-facing AI tools (e.g., ChatGPT or unmanaged cloud AI services).
   - Use only tools/platforms that provide explicit, enforceable data containment and confidentiality guarantees.
   - Refer to **ADNI DUA Appendix A** for detailed restrictions.

3. **Redistribution**
   - **Do not** redistribute ADNI data in any form.
   - Sharing of participant-level data is restricted to your authorized research team.
   - Distribution of derived datasets must follow ADNI policies and, where required, be coordinated with the ADNI study PI.

4. **Publications**
   - Cite ADNI according to the official guidelines.
   - Submit manuscripts to the ADNI Data and Publications Committee prior to journal submission.
   - Follow all acknowledgment and citation requirements.

5. **Security and Incident Reporting**
   - Implement appropriate administrative, physical, and technical safeguards.
   - Report any unauthorized disclosure or breach to ADNI within the timeframe specified in the DUA (typically within 15 days).

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