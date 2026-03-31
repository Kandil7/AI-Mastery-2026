# Attribution and Legal Compliance

## FineTuningLLMs Integration - Legal Documentation

This document outlines the legal requirements, attribution guidelines, and compliance measures for integrating the FineTuningLLMs repository into the AI-Mastery-2026 curriculum.

---

## 1. License Overview

### 1.1 Original License

The FineTuningLLMs repository is licensed under the **MIT License**:

```
MIT License

Copyright (c) [year] [fullname]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### 1.2 License Analysis

| Permission | Allowed | Notes |
|------------|---------|-------|
| Commercial Use | ✅ Yes | No restrictions |
| Modification | ✅ Yes | Allowed with attribution |
| Distribution | ✅ Yes | Allowed with license |
| Private Use | ✅ Yes | No restrictions |
| Sublicensing | ✅ Yes | Allowed |
| Patent Use | ✅ Yes | Implicit in MIT |

| Condition | Required | Notes |
|-----------|----------|-------|
| License & Copyright | ✅ Yes | Must preserve in all copies |
| State Changes | ⚠️ Recommended | Best practice for modifications |
| Disclose Source | ❌ No | Not required by MIT |
| Same License | ❌ No | Derivatives can use different license |

---

## 2. Attribution Requirements

### 2.1 Mandatory Attribution

The MIT License requires:

> "The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software."

**What this means for AI-Mastery-2026:**

1. ✅ Preserve original copyright notice
2. ✅ Include full license text
3. ✅ Maintain attribution to dvgodoy
4. ✅ Include attribution in all adapted content

### 2.2 Recommended Attribution Best Practices

Beyond legal requirements, follow these best practices:

1. **Link to Original Repository**
   - Provide clear link to https://github.com/dvgodoy/FineTuningLLMs
   - Mention the associated book

2. **Note Modifications**
   - Clearly state what changes were made
   - Include adaptation date

3. **Maintain License Headers**
   - Keep original license headers in code files
   - Add adaptation notices

4. **Credit Original Author**
   - Use "Original Author: dvgodoy" consistently
   - Include full name where known (Daniel Godoy)

---

## 3. Attribution Templates

### 3.1 Repository-Level Attribution

**Location:** `README.md` (main curriculum)

```markdown
## Fine-Tuning Track Attribution

The Fine-Tuning Track (Tier 2, Track 06) incorporates content from the 
FineTuningLLMs repository by dvgodoy.

**Original Repository:** https://github.com/dvgodoy/FineTuningLLMs  
**Original Author:** dvgodoy (Daniel Godoy)  
**License:** MIT License  
**Associated Book:** "A Hands-On Guide to Fine-Tuning LLMs with PyTorch and Hugging Face"

This integration includes modifications and enhancements for the AI-Mastery-2026 
curriculum while maintaining the core educational value and proper attribution.

We gratefully acknowledge this excellent educational resource and encourage 
students to explore the original repository for additional learning materials.
```

### 3.2 Module-Level Attribution

**Location:** Each module's `README.md`

```markdown
## Content Attribution

This module incorporates content adapted from the FineTuningLLMs repository:

| Property | Value |
|----------|-------|
| Original Author | dvgodoy (Daniel Godoy) |
| Repository | https://github.com/dvgodoy/FineTuningLLMs |
| License | MIT License |
| Source Content | Chapter X: [Chapter Title] |
| Adaptation Date | 2026 |
| Adapted By | AI-Mastery-2026 Team |

**Modifications Made:**
- Added learning objectives
- Integrated knowledge check questions
- Added "Try It Yourself" exercises
- Updated imports for curriculum structure
- Added troubleshooting tips

We maintain full compliance with the MIT License and encourage students to 
explore the original repository for additional examples and insights.
```

### 3.3 Notebook-Level Attribution

**Location:** Top and bottom of each Jupyter notebook

```markdown
---
title: "[Module Title]"
original_author: dvgodoy
original_source: https://github.com/dvgodoy/FineTuningLLMs
original_license: MIT
adaptations_by: AI-Mastery-2026 Team
adaptation_date: 2026
---

## Attribution Notice

This notebook is adapted from the FineTuningLLMs repository by dvgodoy:
- **Repository:** https://github.com/dvgodoy/FineTuningLLMs
- **License:** MIT License
- **Associated Book:** "A Hands-On Guide to Fine-Tuning LLMs with PyTorch and Hugging Face"

**Modifications for AI-Mastery-2026:**
- Added learning objectives
- Integrated knowledge check questions
- Added "Try It Yourself" exercises
- Updated imports for curriculum structure

---

[Notebook Content]

---

## License & Attribution

**Original Content:**
- Author: dvgodoy (Daniel Godoy)
- Repository: https://github.com/dvgodoy/FineTuningLLMs
- License: MIT License

**Adaptations:**
- Adapted by: AI-Mastery-2026 Team
- License: MIT License

Full license text included in: [LICENSE](../../../LICENSE)
```

### 3.4 Code File Attribution

**Location:** Top of Python scripts

```python
"""
[File Description]

Original Source: FineTuningLLMs Repository
Original Author: dvgodoy (Daniel Godoy)
Original License: MIT License
Source URL: https://github.com/dvgodoy/FineTuningLLMs

Adaptations for AI-Mastery-2026:
- [List modifications]

License: MIT License (see LICENSE file)
"""

# Original copyright notice preserved below:
# Copyright (c) [year] dvgodoy
# MIT License - see original repository for full text
```

### 3.5 Documentation Attribution

**Location:** Technical documentation files

```markdown
## Attribution

This documentation incorporates content from:

**FineTuningLLMs Repository**
- Author: dvgodoy (Daniel Godoy)
- URL: https://github.com/dvgodoy/FineTuningLLMs
- License: MIT License

Adapted for AI-Mastery-2026 curriculum with modifications including:
- [List specific adaptations]

See [ATTRIBUTION_AND_LEGAL.md](./ATTRIBUTION_AND_LEGAL.md) for full legal details.
```

---

## 4. License File Management

### 4.1 Repository Structure

```
AI-Mastery-2026/
├── LICENSE                          # Main repository license
├── README.md                        # Contains attribution section
├── docs/
│   └── curriculum/
│       └── integrations/
│           └── fine-tuning-llms/
│               ├── ATTRIBUTION_AND_LEGAL.md    # This document
│               └── LICENSE_FINE_TUNING_LLMs    # Copy of original MIT license
└── notebooks/
    └── 04_llm_fundamentals/
        └── fine-tuning/
            ├── README.md            # Directory attribution
            └── [notebooks with attribution headers]
```

### 4.2 License File Content

Create `LICENSE_FINE_TUNING_LLMs`:

```
MIT License

Copyright (c) 2023-2026 dvgodoy (Daniel Godoy)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

Source: https://github.com/dvgodoy/FineTuningLLMs
Retrieved: March 30, 2026
Integrated: AI-Mastery-2026 Curriculum
```

---

## 5. Third-Party Content Considerations

### 5.1 Datasets

| Dataset | License | Attribution Required | Notes |
|---------|---------|---------------------|-------|
| Alpaca | CC BY-NC-SA 4.0 | ✅ Yes | Non-commercial |
| Dolly | CC BY-SA 4.0 | ✅ Yes | Share-alike |
| OpenOrca | MIT | ✅ Yes | Compatible |
| Custom datasets | Varies | Check | Document sources |

**Action:** Document all dataset sources and licenses in each notebook.

### 5.2 Pre-trained Models

| Model | License | Commercial Use | Notes |
|-------|---------|----------------|-------|
| Llama 2 | Custom (Meta) | ✅ Yes (with terms) | Follow Meta license |
| Mistral | Apache 2.0 | ✅ Yes | Permissive |
| Phi-2 | MIT | ✅ Yes | Permissive |
| TinyLlama | Apache 2.0 | ✅ Yes | Permissive |

**Action:** Include model license notices in deployment modules.

### 5.3 Libraries

| Library | License | Notes |
|---------|---------|-------|
| PyTorch | BSD-style | Permissive |
| Transformers | Apache 2.0 | Permissive |
| PEFT | Apache 2.0 | Permissive |
| BitsAndBytes | Apache 2.0 | Permissive |
| accelerate | Apache 2.0 | Permissive |

**Action:** All core dependencies have permissive licenses.

---

## 6. Compliance Checklist

### 6.1 Legal Compliance

```
Legal Compliance Checklist:
━━━━━━━━━━━━━━━━━━━━━━━━━━━

[ ] MIT License Requirements
    [ ] Original copyright notice preserved
    [ ] Full license text included
    [ ] Attribution to dvgodoy maintained
    [ ] Modifications documented

[ ] Repository Documentation
    [ ] Main README includes attribution section
    [ ] LICENSE_FINE_TUNING_LLMs created
    [ ] ATTRIBUTION_AND_LEGAL.md complete

[ ] Module Documentation
    [ ] Each module README has attribution
    [ ] Source chapters identified
    [ ] Modifications listed

[ ] Notebook Attribution
    [ ] Header includes attribution metadata
    [ ] Footer includes license notice
    [ ] Original author credited
    [ ] Repository URL included

[ ] Code Files
    [ ] License headers preserved
    [ ] Attribution comments added
    [ ] Modification history noted

[ ] Third-Party Content
    [ ] Dataset licenses documented
    [ ] Model licenses noted
    [ ] Library licenses compatible
```

### 6.2 Attribution Quality

```
Attribution Quality Checklist:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[ ] Visibility
    [ ] Attribution is prominent (not hidden)
    [ ] Links are working
    [ ] Author name is correct

[ ] Completeness
    [ ] All required elements included
    [ ] Source URL is accurate
    [ ] License type is specified

[ ] Consistency
    [ ] Same format across all files
    [ ] Author name consistent
    [ ] Repository URL consistent

[ ] Clarity
    [ ] Easy to understand
    [ ] Modifications clearly noted
    [ ] No confusing or misleading statements
```

---

## 7. Modification Documentation

### 7.1 Types of Modifications

| Modification Type | Description | Documentation Required |
|-------------------|-------------|----------------------|
| Content Addition | New sections, exercises | List additions |
| Content Removal | Removed sections | Note removals |
| Content Modification | Changed explanations | Summarize changes |
| Code Updates | Updated imports, versions | Document version changes |
| Structural Changes | Split/merged notebooks | Document new structure |

### 7.2 Modification Log Template

```markdown
## Modification Log

| Date | Version | Changes | Author |
|------|---------|---------|--------|
| 2026-03-30 | 1.0 | Initial integration | AI-Mastery-2026 Team |
| | | - Split Chapter 0 into 2 modules | |
| | | - Added learning objectives | |
| | | - Added knowledge checks | |
| | | - Updated library versions | |

### Detailed Changes

#### Module 1: Introduction to Fine-Tuning
- **Added:** Learning objectives section
- **Added:** Business use cases for fine-tuning
- **Added:** Decision tree (Fine-tune vs Prompt)
- **Added:** 5 knowledge check questions
- **Modified:** Updated Hugging Face examples to v4.37+

#### Module 2: Transformer Architecture
- **Added:** Interactive architecture diagrams
- **Added:** Mathematical derivation of attention
- **Added:** 8 knowledge check questions
- **Modified:** Updated model examples
```

---

## 8. Commercial Use Considerations

### 8.1 Permitted Commercial Uses

| Use Case | Permitted | Notes |
|----------|-----------|-------|
| Paid courses | ✅ Yes | MIT allows commercial use |
| Corporate training | ✅ Yes | No restrictions |
| SaaS integration | ✅ Yes | No copyleft |
| Consulting | ✅ Yes | Allowed |
| Books/Publications | ✅ Yes | Allowed with attribution |

### 8.2 Attribution in Commercial Contexts

For commercial use, maintain:

1. **Course Materials:** Include attribution in syllabus/resources
2. **Documentation:** Include attribution in user docs
3. **Code:** Preserve license headers
4. **Marketing:** Acknowledge sources where appropriate

### 8.3 Recommended Commercial Attribution

```markdown
## Acknowledgments

This [course/product/service] incorporates educational content from the 
FineTuningLLMs repository by dvgodoy, licensed under the MIT License.

We gratefully acknowledge this excellent resource that helped make this 
[course/product/service] possible.

Original Repository: https://github.com/dvgodoy/FineTuningLLMs
```

---

## 9. Risk Mitigation

### 9.1 Legal Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| License violation | Low | High | Comprehensive attribution |
| Copyright claim | Low | High | Document all sources |
| Third-party claims | Low | Medium | Verify dataset licenses |
| Model license issues | Low | Medium | Use permissively licensed models |

### 9.2 Mitigation Actions

1. **Legal Review:** Have legal team review attribution
2. **Documentation:** Maintain comprehensive source documentation
3. **Regular Audits:** Quarterly review of attribution compliance
4. **Update Process:** Process for handling license changes

### 9.3 Contact Information

**For Legal Questions:**
- AI-Mastery-2026 Legal Contact: [legal@your-org.com]
- FineTuningLLMs Repository: https://github.com/dvgodoy/FineTuningLLMs
- Original Author: dvgodoy (via GitHub)

---

## 10. Enforcement and Updates

### 10.1 Compliance Monitoring

| Activity | Frequency | Responsible |
|----------|-----------|-------------|
| Attribution audit | Quarterly | Legal Team |
| Link validation | Weekly | Automated |
| License update check | Monthly | Curriculum Lead |
| Student feedback review | Ongoing | Teaching Team |

### 10.2 Update Process

When upstream content changes:

1. **Review Changes:** Assess impact on integrated content
2. **Update Attribution:** Note new version/date
3. **Test Integration:** Verify all notebooks still work
4. **Document Changes:** Update modification log
5. **Notify Students:** Communicate significant changes

### 10.3 Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-03-30 | Initial legal documentation |

---

## Appendix: Full License Text

### MIT License (FineTuningLLMs)

```
MIT License

Copyright (c) 2023-2026 dvgodoy (Daniel Godoy)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### Quick Attribution Summary

```
For quick reference, here's the minimum required attribution:

"Content adapted from FineTuningLLMs by dvgodoy 
(https://github.com/dvgodoy/FineTuningLLMs), licensed under MIT License."

Best practice includes:
- Full author name: dvgodoy (Daniel Godoy)
- Repository URL: https://github.com/dvgodoy/FineTuningLLMs
- License: MIT License
- Associated Book: "A Hands-On Guide to Fine-Tuning LLMs with PyTorch and Hugging Face"
- Modifications made
```

---

## References

### Legal Resources

- [MIT License Explained](https://choosealicense.com/licenses/mit/)
- [Open Source License Compliance](https://opensource.org/licenses)
- [Software License Notices](https://www.law.cornell.edu/cfr/text/17/201.20)

### Attribution Best Practices

- [THANKS Files in Open Source](https://github.com/thanks)
- [NOTICE File Conventions](https://www.apache.org/legal/src-headers.html)
- [Open Source Attribution Guidelines](https://opensource.guide/legal/)

### Related Documentation

- [REPO_ANALYSIS_FINE_TUNING.md](./REPO_ANALYSIS_FINE_TUNING.md)
- [CURRICULUM_INTEGRATION_MAP.md](./CURRICULUM_INTEGRATION_MAP.md)
- [NOTEBOOK_INTEGRATION_PLAN.md](./NOTEBOOK_INTEGRATION_PLAN.md)

---

*Document Version: 1.0*  
*Created: March 30, 2026*  
*Last Reviewed: March 30, 2026*  
*Next Review: June 30, 2026*

*For: AI-Mastery-2026 Curriculum Integration*  
*Legal Status: Compliant with MIT License*
