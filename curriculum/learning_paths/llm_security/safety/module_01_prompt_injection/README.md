# Module 1: Prompt Injection Attacks

**Track:** LLM Security & Safety  
**Module ID:** SEC-SAFETY-001  
**Version:** 1.0  
**Last Updated:** March 30, 2026  
**Status:** ✅ Production Ready

---

## 📋 Module Overview

### Description

Prompt injection attacks represent one of the most critical security vulnerabilities in LLM-powered applications. This module provides comprehensive coverage of prompt injection techniques, real-world attack vectors, and defense strategies. Students will learn to identify, exploit (in controlled environments), and defend against various forms of prompt injection attacks.

### Why This Matters

> ⚠️ **Critical Security Risk:** Prompt injection is listed as **#1** in the OWASP Top 10 for LLM Applications (LLM01:2025). Understanding and mitigating these attacks is essential for any production LLM system.

**Real-World Impact:**
- Data exfiltration from RAG systems
- Unauthorized access to system functions
- Manipulation of AI-generated content
- Compliance violations (GDPR, HIPAA)
- Reputation damage and financial loss

---

## 🎯 Learning Objectives

By the end of this module, students will be able to:

| Level | Bloom's Taxonomy | Objective |
|-------|------------------|-----------|
| **Remember** | Recall | Define prompt injection and identify its various types (direct, indirect, multi-turn) |
| **Understand** | Comprehend | Explain how prompt injection attacks exploit LLM instruction-following behavior |
| **Apply** | Execute | Implement prompt injection attacks in controlled lab environments |
| **Analyze** | Differentiate | Analyze attack vectors and assess severity of potential vulnerabilities |
| **Create** | Design | Design and implement comprehensive defense strategies including input sanitization, output filtering, and architectural safeguards |

---

## 📚 Prerequisites

### Required Knowledge

| Topic | Proficiency Level | Verification |
|-------|-------------------|--------------|
| Python Programming | Intermediate | Complete `part1_fundamentals/module_1_2_python/` |
| LLM Fundamentals | Basic | Understand how LLMs process prompts and generate responses |
| API Integration | Basic | Experience calling REST APIs |
| Security Basics | Basic | Understanding of common web vulnerabilities (helpful but not required) |

### Technical Requirements

```bash
# Python 3.10+ required
python --version  # Should be 3.10 or higher

# Required packages (install via requirements.txt)
pip install -r requirements.txt

# Environment variables needed
export OPENAI_API_KEY="your-key-here"  # Or use .env file
export ANTHROPIC_API_KEY="your-key-here"  # Optional, for multi-provider labs
```

---

## ⏱️ Time Estimates

| Component | Estimated Time | Description |
|-----------|---------------|-------------|
| **Theory Content** | 2.5 hours | Reading, videos, and concept review |
| **Hands-On Labs** | 4.0 hours | Three guided lab exercises with code |
| **Knowledge Check** | 1.0 hour | 10 multiple-choice questions |
| **Coding Challenges** | 3.5 hours | Three progressively difficult challenges |
| **Total Module Time** | **11.0 hours** | Complete module completion |

### Suggested Schedule

```
Day 1: Theory Content (2.5 hours)
  - Read all theory sections
  - Review case studies
  - Watch embedded videos (if available)

Day 2: Labs 1 & 2 (2.5 hours)
  - Complete direct injection lab
  - Complete indirect injection lab

Day 3: Lab 3 & Defenses (2.0 hours)
  - Complete defense implementation lab
  - Test all defenses

Day 4: Assessment (1.0 hour)
  - Complete knowledge check quiz
  - Review explanations

Day 5: Coding Challenges (3.0 hours)
  - Complete all three challenges
  - Submit for evaluation
```

---

## ✅ Success Criteria

To complete this module successfully, students must:

### Minimum Requirements

- [ ] Score **80% or higher** on knowledge check quiz (8/10 correct)
- [ ] Complete **all three labs** with working code
- [ ] Submit **at least one coding challenge** (any difficulty level)

### Excellence Criteria (Recommended)

- [ ] Score **100%** on knowledge check quiz
- [ ] Complete **all three coding challenges** (easy, medium, hard)
- [ ] Implement **additional defense mechanisms** beyond required labs
- [ ] Document **lessons learned** in a security journal

### Competency Validation

After completing this module, you should be able to:

1. ✅ Identify prompt injection attempts in user inputs
2. ✅ Explain the difference between direct and indirect injection
3. ✅ Implement at least 3 different defense strategies
4. ✅ Conduct basic security testing on LLM applications
5. ✅ Document security findings and recommendations

---

## 📁 Module Structure

```
module_01_prompt_injection/
├── README.md                      # This file - module overview
├── 01_theory.md                   # Theory content and concepts
├── 02_case_studies.md             # Real-world examples and analysis
├── requirements.txt               # Python dependencies
├── labs/
│   ├── lab_01_direct_injection.py # Direct injection attack lab
│   ├── lab_02_indirect_injection.py # Indirect injection via RAG
│   └── lab_03_building_defenses.py # Defense implementation lab
├── assessments/
│   ├── knowledge_check.md         # 10 quiz questions
│   └── coding_challenges.md       # 3 coding challenges
├── solutions/
│   ├── lab_solutions.py           # Complete lab solutions
│   └── challenge_solutions.py     # Challenge reference solutions
└── resources/
    ├── further_reading.md         # Additional resources
    ├── tools_frameworks.md        # Security tools and frameworks
    └── best_practices.md          # Industry best practices
```

---

## 🚀 Quick Start

### Option 1: Guided Learning Path (Recommended)

```bash
# 1. Navigate to module directory
cd curriculum/learning_paths/llm_security/safety/module_01_prompt_injection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start with theory
cat 01_theory.md

# 4. Run labs in order
python labs/lab_01_direct_injection.py
python labs/lab_02_indirect_injection.py
python labs/lab_03_building_defenses.py

# 5. Complete assessment
cat assessments/knowledge_check.md
cat assessments/coding_challenges.md
```

### Option 2: Assessment-First Approach

```bash
# 1. Try the knowledge check first to gauge understanding
cat assessments/knowledge_check.md

# 2. Review theory based on knowledge gaps
cat 01_theory.md

# 3. Complete labs for hands-on practice
python labs/lab_01_direct_injection.py

# 4. Challenge yourself with coding exercises
cat assessments/coding_challenges.md
```

---

## 📊 Assessment Breakdown

| Assessment Type | Weight | Passing Score | Attempts Allowed |
|-----------------|--------|---------------|------------------|
| Knowledge Check Quiz | 30% | 80% | Unlimited (best score counts) |
| Lab Completion | 40% | All labs complete | Unlimited revisions |
| Coding Challenges | 30% | 1+ submitted | Unlimited revisions |

### Grading Rubric

| Component | Excellent (A) | Proficient (B) | Developing (C) | Needs Improvement (D/F) |
|-----------|---------------|----------------|----------------|------------------------|
| **Knowledge Check** | 100% (10/10) | 90-100% (9-10/10) | 80-89% (8/10) | <80% (<8/10) |
| **Lab Completion** | All labs + extensions | All labs complete | Labs with minor issues | Incomplete labs |
| **Coding Challenges** | All 3 challenges | 2 challenges | 1 challenge (hard) | 1 challenge (easy) |
| **Code Quality** | Production-ready, documented | Good structure, some docs | Basic functionality | Significant issues |

---

## 🆘 Getting Help

### Support Channels

| Channel | Response Time | Best For |
|---------|---------------|----------|
| **GitHub Discussions** | 24-48 hours | General questions, peer support |
| **Office Hours** | Weekly (see schedule) | Live Q&A, code review |
| **Email Support** | 48 hours | Private concerns, accommodations |
| **Discord Community** | Variable | Quick questions, community help |

### Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| API key errors | Verify `.env` file configuration, check key validity |
| Lab code not running | Ensure all dependencies installed (`pip install -r requirements.txt`) |
| Quiz not loading | Clear browser cache, try incognito mode |
| Challenge tests failing | Review test output carefully, check edge cases |

---

## 📜 Academic Integrity

### Allowed

- ✅ Discussing concepts with peers
- ✅ Using official documentation
- ✅ Searching for general programming help
- ✅ Reviewing provided hints and scaffolding

### Not Allowed

- ❌ Copying solutions from other students
- ❌ Using AI to complete assessments (ironic, we know!)
- ❌ Sharing solution code publicly
- ❌ Submitting work that is not your own

**Note:** For this security module, understanding *how* attacks work is essential. All lab exercises should be performed **only in controlled, isolated environments** with proper authorization.

---

## 🔄 Version History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0 | March 30, 2026 | Initial release | AI-Mastery-2026 Security Team |

---

## 📞 Module Authors

- **Lead Author:** AI-Mastery-2026 Security Team
- **Technical Reviewer:** Security & Compliance Engineer
- **Pedagogy Reviewer:** Content Writer (Technical)

---

## 🔗 Related Modules

| Module | Relationship |
|--------|--------------|
| `SEC-SAFETY-002` | Content Moderation & Toxicity Detection (Next in sequence) |
| `SEC-SAFETY-003` | PII Detection & Data Privacy (Follow-up module) |
| `SEC-AUTH-001` | Authentication & Authorization for LLM Systems (Parallel track) |
| `PROD-MONITOR-001` | Security Monitoring & Incident Response (Advanced) |

---

## 📄 License

This module content is licensed under **CC BY-NC-SA 4.0** (Creative Commons Attribution-NonCommercial-ShareAlike).

Code examples are licensed under **MIT License** for educational use.

---

**Ready to begin? Start with [01_theory.md](01_theory.md)** →
