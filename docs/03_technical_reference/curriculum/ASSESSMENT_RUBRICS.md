# 📋 AI-Mastery-2026: Assessment Rubrics & Evaluation Framework

**Companion Document to:** [REDESIGNED_LLM_CURRICULUM_2026.md](./REDESIGNED_LLM_CURRICULUM_2026.md)  
**Version:** 1.0  
**Last Updated:** March 29, 2026

---

## 📊 Table of Contents

1. [Project Rubrics](#project-rubrics)
2. [Quiz Specifications](#quiz-specifications)
3. [Capstone Evaluation](#capstone-evaluation)
4. [Code Review Rubrics](#code-review-rubrics)
5. [Peer Review Guidelines](#peer-review-guidelines)
6. [Knowledge Check Templates](#knowledge-check-templates)
7. [Grading Calibration](#grading-calibration)

---

## 📝 Project Rubrics

### Standard Project Rubric (100 points)

Used for all hands-on projects across all tracks.

| Criterion | Points | Excellent (90-100%) | Good (80-89%) | Satisfactory (70-79%) | Needs Improvement (60-69%) | Unsatisfactory (<60%) |
|-----------|--------|---------------------|---------------|----------------------|---------------------------|----------------------|
| **Functionality** | 30 | All features work flawlessly; exceeds requirements | All required features work correctly | Most features work; minor issues | Some features incomplete or buggy | Major functionality missing |
| **Code Quality** | 20 | Clean, readable, well-structured; follows all best practices | Good structure; follows most best practices | Acceptable structure; some violations | Poor structure; many violations | Unmaintainable code |
| **Testing** | 15 | Comprehensive tests (>90% coverage); edge cases covered | Good tests (>80% coverage); most cases covered | Adequate tests (>70% coverage) | Limited tests (<70% coverage) | No or minimal tests |
| **Documentation** | 15 | Excellent README; API docs; comments; examples | Good documentation; minor gaps | Adequate documentation | Poor documentation | No documentation |
| **Performance** | 10 | Exceeds all performance targets | Meets all performance targets | Mostly meets targets | Below targets | Significant performance issues |
| **Innovation** | 10 | Creative solutions; novel approaches; extra features | Some creative elements | Standard implementation | Minimal effort | Copy/template only |

### Grading Scale

| Score Range | Grade | Status | Action |
|-------------|-------|--------|--------|
| 90-100 | A | Exceeds Expectations | Pass with distinction |
| 80-89 | B | Meets Expectations | Pass |
| 70-79 | C | Approaching Expectations | Pass (remediation recommended) |
| 60-69 | D | Below Expectations | Retry required |
| <60 | F | Unsatisfactory | Retry required with support |

---

## 📝 Project-Type Specific Rubrics

### Implementation Project Rubric

Used for coding implementation projects (e.g., "Build Transformer from Scratch")

| Criterion | Weight | Evaluation Criteria |
|-----------|--------|---------------------|
| **Correctness** | 35% | • Algorithm produces correct outputs<br>• Handles edge cases properly<br>• No critical bugs |
| **Code Quality** | 25% | • Clean, readable code<br>• Proper modularization<br>• Consistent style |
| **Testing** | 20% | • Unit tests for all components<br>• Integration tests<br>• Edge case coverage |
| **Documentation** | 10% | • Clear README<br>• Inline comments<br>• Usage examples |
| **Performance** | 10% | • Meets latency requirements<br>• Efficient memory usage<br>• Scalable design |

### Analysis Project Rubric

Used for analysis projects (e.g., "Compare Embedding Models")

| Criterion | Weight | Evaluation Criteria |
|-----------|--------|---------------------|
| **Methodology** | 30% | • Clear experimental design<br>• Appropriate metrics<br>• Controlled variables |
| **Analysis Depth** | 25% | • Thorough investigation<br>• Multiple perspectives<br>• Root cause analysis |
| **Data Quality** | 20% | • Representative datasets<br>• Sufficient sample size<br>• Proper preprocessing |
| **Conclusions** | 15% | • Evidence-based claims<br>• Clear recommendations<br>• Limitations acknowledged |
| **Presentation** | 10% | • Clear visualizations<br>• Well-structured report<br>• Executive summary |

### Design Project Rubric

Used for design projects (e.g., "Architect RAG System")

| Criterion | Weight | Evaluation Criteria |
|-----------|--------|---------------------|
| **Requirements** | 20% | • Clear problem definition<br>• User needs identified<br>• Constraints documented |
| **Architecture** | 30% | • Appropriate patterns<br>• Scalability considered<br>• Trade-offs analyzed |
| **Technical Depth** | 25% | • Sound technical decisions<br>• Industry best practices<br>• Security considered |
| **Documentation** | 15% | • Architecture diagrams<br>• Component specifications<br>• API contracts |
| **Feasibility** | 10% | • Realistic implementation plan<br>• Resource estimates<br>• Risk assessment |

---

## 📝 Capstone Evaluation Rubric

Used for track capstone projects (weighted more heavily than regular projects)

### Capstone Scoring Matrix (200 points)

| Category | Sub-Category | Points | Evaluation Criteria |
|----------|--------------|--------|---------------------|
| **Problem Definition** | Clarity | 10 | Clear, specific problem statement |
| | Relevance | 10 | Addresses real-world need |
| | Scope | 5 | Appropriate complexity for track |
| **Technical Implementation** | Functionality | 40 | All features working correctly |
| | Code Quality | 25 | Production-quality code |
| | Architecture | 15 | Sound design patterns |
| | Testing | 15 | Comprehensive test coverage |
| **Documentation** | README | 10 | Complete setup and usage guide |
| | Architecture Docs | 10 | System design documentation |
| | API Documentation | 10 | Complete API reference |
| **Demonstration** | Live Demo | 15 | Working demonstration |
| | Performance | 10 | Meets performance targets |
| | User Guide | 5 | Clear usage instructions |
| **Reflection** | Lessons Learned | 10 | Insightful self-assessment |
| | Future Work | 10 | Thoughtful improvement ideas |
| | Ethical Considerations | 10 | Addresses ethical implications |
| **Total** | | **200** | |

### Capstone Defense Rubric

| Criterion | Points | Evaluation Criteria |
|-----------|--------|---------------------|
| **Presentation Quality** | 25 | • Clear, engaging presentation<br>• Well-organized slides<br>• Appropriate timing |
| **Technical Q&A** | 40 | • Accurate technical responses<br>• Deep understanding demonstrated<br>• Handles challenging questions |
| **Design Justification** | 20 | • Can explain design decisions<br>• Discusses trade-offs<br>• Defends architectural choices |
| **Future Vision** | 15 | • Clear roadmap for improvements<br>• Understands limitations<br>• Realistic scaling plans |

---

## 📝 Quiz Specifications

### Quiz Structure Template

Each track has 3 quizzes with the following structure:

| Quiz | Modules Covered | Questions | Time Limit | Passing Score |
|------|-----------------|-----------|------------|---------------|
| Quiz 1 | Modules 1-4 | 25-30 | 45 minutes | 80% |
| Quiz 2 | Modules 5-7 | 25-30 | 45 minutes | 80% |
| Quiz 3 | Modules 8-10 | 20-25 | 40 minutes | 80% |

### Question Type Distribution

| Question Type | Percentage | Points per Question | Example |
|---------------|------------|---------------------|---------|
| Multiple Choice | 40% | 1 point | "Which of the following is NOT a component of the transformer architecture?" |
| True/False | 15% | 1 point | "True or False: Self-attention allows each position to attend to all positions." |
| Code Completion | 20% | 2 points | "Complete the attention score calculation: `scores = Q @ K.T / ___`" |
| Short Answer | 15% | 3 points | "Explain the purpose of layer normalization in transformers." |
| Diagram Labeling | 10% | 2 points | "Label the components of the transformer architecture diagram." |

### Bloom's Taxonomy Distribution

| Bloom's Level | Percentage | Question Types | Example Verbs |
|---------------|------------|----------------|---------------|
| Remember | 15% | Multiple Choice, T/F | Define, List, Recall, Identify |
| Understand | 20% | Multiple Choice, Short Answer | Explain, Describe, Summarize |
| Apply | 30% | Code Completion, Short Answer | Implement, Use, Calculate |
| Analyze | 20% | Short Answer, Diagram | Compare, Contrast, Debug |
| Evaluate | 10% | Short Answer | Critique, Justify, Recommend |
| Create | 5% | Short Answer, Code | Design, Build, Architect |

### Quiz Quality Checklist

Before publishing any quiz, verify:

- [ ] Questions align with learning objectives
- [ ] Difficulty is appropriate for module level
- [ ] No ambiguous or trick questions
- [ ] All correct answers are verifiable
- [ ] Distractors (wrong answers) are plausible
- [ ] Code snippets are syntactically correct
- [ ] Diagrams are clear and labeled
- [ ] Time limit is appropriate (avg. 1.5 min/question)
- [ ] Passing score is achievable but meaningful
- [ ] Questions are randomized from question bank

---

## 📝 Knowledge Check Templates

### Module Knowledge Check (5 questions)

Each module includes a knowledge check with 5 questions for formative assessment.

**Template:**

```markdown
### Module X.Y: Knowledge Check

**Instructions:** Answer all 5 questions. You have unlimited attempts. 
Minimum passing score: 70%.

---

**Question 1** (Remember - 1 point)
[Multiple choice question testing recall of key concept]

A) Option A
B) Option B
C) Option C
D) Option D

**Correct Answer:** B
**Explanation:** [Brief explanation of why B is correct]

---

**Question 2** (Understand - 1 point)
[True/False or multiple choice testing comprehension]

**Correct Answer:** [Answer]
**Explanation:** [Explanation]

---

**Question 3** (Apply - 2 points)
[Code completion or calculation]

**Correct Answer:** [Answer]
**Explanation:** [Explanation]

---

**Question 4** (Analyze - 2 points)
[Comparison or debugging question]

**Correct Answer:** [Answer]
**Explanation:** [Explanation]

---

**Question 5** (Evaluate - 2 points)
[Short answer requiring judgment]

**Sample Answer:** [Sample response]
**Grading Rubric:** [Criteria for full/partial credit]

---

**Total Points:** 8
**Passing Score:** 6 (70%)
```

---

## 📝 Code Review Rubrics

### Peer Code Review Rubric

Used for peer review of project submissions.

| Criterion | Rating | Comments |
|-----------|--------|----------|
| **Code Readability** | ⭐⭐⭐⭐⭐ | Code is easy to read and understand |
| | ⭐⭐⭐⭐ | Mostly readable with minor issues |
| | ⭐⭐⭐ | Some readability concerns |
| | ⭐⭐ | Difficult to follow |
| | ⭐ | Very hard to understand |
| **Code Structure** | ⭐⭐⭐⭐⭐ | Well-organized, modular design |
| | ⭐⭐⭐⭐ | Good structure with minor issues |
| | ⭐⭐⭐ | Acceptable organization |
| | ⭐⭐ | Poor modularization |
| | ⭐ | No clear structure |
| **Error Handling** | ⭐⭐⭐⭐⭐ | Comprehensive error handling |
| | ⭐⭐⭐⭐ | Good error handling |
| | ⭐⭐⭐ | Basic error handling |
| | ⭐⭐ | Minimal error handling |
| | ⭐ | No error handling |
| **Testing** | ⭐⭐⭐⭐⭐ | Thorough test coverage |
| | ⭐⭐⭐⭐ | Good test coverage |
| | ⭐⭐⭐ | Adequate tests |
| | ⭐⭐ | Limited tests |
| | ⭐ | No tests |
| **Documentation** | ⭐⭐⭐⭐⭐ | Excellent documentation |
| | ⭐⭐⭐⭐ | Good documentation |
| | ⭐⭐⭐ | Adequate documentation |
| | ⭐⭐ | Poor documentation |
| | ⭐ | No documentation |

### Reviewer Guidelines

**When reviewing code:**

1. **Be Constructive**
   - Focus on the code, not the person
   - Suggest improvements, not just criticisms
   - Acknowledge what works well

2. **Be Specific**
   - Point to specific lines or sections
   - Explain why something is an issue
   - Suggest concrete improvements

3. **Be Thorough**
   - Review all major components
   - Check edge cases
   - Verify tests cover key scenarios

4. **Be Timely**
   - Complete reviews within 48 hours
   - Provide actionable feedback
   - Follow up on revisions

---

## 📝 Peer Review Guidelines

### Peer Review Process

1. **Assignment**
   - Learners are assigned 2-3 peer projects to review
   - Reviews are anonymous (reviewer identity hidden)
   - Reviews are due within 48 hours of assignment

2. **Review Components**
   - Complete rubric evaluation
   - Provide written feedback (minimum 100 words)
   - Suggest 2-3 specific improvements
   - Highlight 1-2 strengths

3. **Quality Standards**
   - Reviews must be constructive and respectful
   - Feedback must be specific and actionable
   - Reviews are themselves evaluated for quality

### Peer Review Quality Rubric

| Criterion | Points | Evaluation Criteria |
|-----------|--------|---------------------|
| **Completeness** | 30 | All rubric sections completed |
| **Specificity** | 30 | Feedback references specific code sections |
| **Constructiveness** | 25 | Suggestions are actionable and helpful |
| **Professionalism** | 15 | Tone is respectful and supportive |

---

## 📝 Grading Calibration

### Grader Training Requirements

All graders (TAs, instructors) must complete:

1. **Calibration Exercise**
   - Grade 5 sample submissions independently
   - Compare scores with master grades
   - Discuss discrepancies until alignment achieved
   - Must achieve ≥90% agreement with master grades

2. **Rubric Familiarization**
   - Review all rubric criteria in detail
   - Practice applying rubric to sample work
   - Discuss edge cases and borderline scenarios

3. **Bias Awareness**
   - Complete unconscious bias training
   - Understand common grading biases
   - Implement strategies to mitigate bias

### Inter-Rater Reliability

**Target:** ≥85% agreement between graders

**Measurement:**
- Randomly select 10% of submissions for double-grading
- Calculate correlation between grader scores
- Investigate and resolve significant discrepancies
- Retrain graders if reliability drops below target

### Grade Appeal Process

1. **Submission**
   - Learner submits appeal within 7 days of grade
   - Must specify grounds for appeal
   - Must include supporting evidence

2. **Review**
   - Independent grader re-evaluates submission
   - Original grader provides rationale
   - Third party (instructor) makes final decision

3. **Resolution**
   - Decision communicated within 5 business days
   - Grade may be adjusted up, down, or unchanged
   - Decision is final

---

## 📝 Assessment Analytics

### Track-Level Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Quiz Pass Rate** | ≥85% | % of learners passing each quiz |
| **Project Completion Rate** | ≥80% | % of learners completing all projects |
| **Average Project Score** | ≥75 | Mean project score across track |
| **Time-to-Completion** | Within 20% of estimate | Average days to complete track |
| **Satisfaction Score** | ≥4.0/5.0 | Post-track survey results |

### Question-Level Analytics

| Metric | Target | Action if Below Target |
|--------|--------|------------------------|
| **Difficulty Index** | 0.5-0.8 | Revise if <0.3 (too hard) or >0.9 (too easy) |
| **Discrimination Index** | ≥0.3 | Revise if <0.2 (doesn't differentiate) |
| **Point-Biserial** | ≥0.2 | Revise if negative (wrong key?) |
| **Response Time** | Within expected range | Revise if significantly longer |

### Continuous Improvement

**Monthly Review:**
- Analyze assessment performance data
- Identify problematic questions/items
- Update question bank
- Refine rubrics based on grading patterns

**Quarterly Review:**
- Comprehensive curriculum assessment
- Industry alignment check
- Graduate outcome analysis
- Major updates as needed

---

## 📝 Accessibility Considerations

### Assessment Accessibility

All assessments must:

- [ ] Be compatible with screen readers
- [ ] Provide alternative text for images/diagrams
- [ ] Use color-blind friendly palettes
- [ ] Allow extended time for learners with accommodations
- [ ] Provide transcripts for audio/video content
- [ ] Support keyboard navigation
- [ ] Use clear, simple language

### Accommodation Process

1. **Request**
   - Learner submits accommodation request
   - Provides documentation if required
   - Specifies needed accommodations

2. **Approval**
   - Accessibility team reviews request
   - Determines appropriate accommodations
   - Communicates decision within 5 business days

3. **Implementation**
   - LMS configured for accommodations
   - Instructors notified of approved accommodations
   - Regular check-ins to ensure effectiveness

---

## 📝 Academic Integrity

### Plagiarism Detection

**Tools Used:**
- Code similarity detection (MOSS, JPlag)
- Text similarity detection (Turnitin)
- AI-generated content detection

**Thresholds:**
- Code similarity >30% triggers review
- Text similarity >20% triggers review
- AI detection >50% triggers review

### Violation Consequences

| Violation Level | First Offense | Second Offense | Third Offense |
|-----------------|---------------|----------------|---------------|
| **Minor** (unintentional) | Warning + education | Grade reduction | Track failure |
| **Moderate** (significant copying) | Grade reduction | Track failure | Program dismissal |
| **Severe** (contract cheating) | Track failure | Program dismissal | N/A |

### Integrity Pledge

All learners must acknowledge:

> "I pledge that all work submitted is my own. I have not copied from others, used unauthorized resources, or had someone else complete my work. I understand the consequences of academic dishonesty."

---

**Document Status:** ✅ Production Ready  
**Last Updated:** March 29, 2026  
**Companion Documents:** 
- [REDESIGNED_LLM_CURRICULUM_2026.md](./REDESIGNED_LLM_CURRICULUM_2026.md)
- [LEARNING_PATH_VISUAL.md](./LEARNING_PATH_VISUAL.md)
