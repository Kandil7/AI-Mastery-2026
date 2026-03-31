# Enhanced Fine-Tuning Track

## AI-Mastery-2026: Tier 2, Track 06

This document defines the enhanced fine-tuning track structure, incorporating content from the FineTuningLLMs repository by dvgodoy.

---

## Track Overview

**Track Title:** Fine-Tuning LLMs with PyTorch and Hugging Face  
**Track Duration:** 6-8 weeks (40-60 hours)  
**Difficulty Level:** Intermediate to Advanced  
**Prerequisites:** Python, PyTorch basics, Transformer fundamentals  
**Hardware:** GPU with 8GB+ VRAM or Colab Free tier

### Learning Outcomes

Upon completing this track, students will be able to:

1. ✅ Explain the theory behind LLM fine-tuning and when to use it
2. ✅ Prepare datasets for fine-tuning with proper formatting
3. ✅ Implement quantization techniques (8-bit, 4-bit) for memory efficiency
4. ✅ Apply PEFT and LoRA for parameter-efficient fine-tuning
5. ✅ Use SFTTrainer for supervised fine-tuning workflows
6. ✅ Optimize training with Flash Attention
7. ✅ Convert models to GGUF format for efficient inference
8. ✅ Deploy fine-tuned models locally with Ollama
9. ✅ Evaluate fine-tuned model performance
10. ✅ Troubleshoot common fine-tuning issues

---

## Track Structure: 12 Modules

```
Module 01: Introduction to Fine-Tuning          (Chapter 0)
Module 02: Transformer Architecture Review      (Chapter 0)
Module 03: Quantization Fundamentals            (Chapter 1)
Module 04: Setting Up GPU Environment           (Appendix A)
Module 05: Data Preparation & Formatting        (Chapter 2)
Module 06: Tokenization & Chat Templates        (Chapter 2)
Module 07: PEFT & LoRA Theory                   (Chapter 3)
Module 08: LoRA Implementation                  (Chapter 3)
Module 09: SFTTrainer Deep Dive                 (Chapter 4)
Module 10: Flash Attention & Optimization       (Chapter 5)
Module 11: GGUF Conversion                      (Chapter 6)
Module 12: Local Deployment with Ollama         (Chapter 6)
```

---

## Module Details

### Module 01: Introduction to Fine-Tuning

**Source:** FineTuningLLMs Chapter 0  
**Duration:** 2-3 hours  
**Difficulty:** Beginner

#### Learning Objectives

By the end of this module, students will be able to:
- Define fine-tuning and explain its purpose
- Distinguish between pre-training, fine-tuning, and prompt engineering
- Identify use cases where fine-tuning is appropriate
- Understand the fine-tuning workflow at a high level

#### Theory Content

| Topic | Source | Enhancement |
|-------|--------|-------------|
| What is Fine-Tuning? | Chapter 0 | Add business use cases |
| Fine-Tuning vs Prompt Engineering | Chapter 0 | Add comparison table |
| When to Fine-Tune | Chapter 0 | Add decision tree |
| Fine-Tuning Workflow Overview | Chapter 0 | Add workflow diagram |
| Types of Fine-Tuning | Chapter 0 | Add full/PEFT comparison |

#### Hands-On Lab

**Lab 1.1:** Environment Setup and First Model Load
```python
# Students will:
# 1. Set up Hugging Face environment
# 2. Load a pre-trained model
# 3. Run basic inference
# 4. Compare model outputs before fine-tuning
```

#### Knowledge Check (5 Questions)

1. What is the primary purpose of fine-tuning?
2. When should you choose fine-tuning over prompt engineering?
3. What are the main types of fine-tuning?
4. What hardware is typically required for fine-tuning?
5. What is the difference between full fine-tuning and PEFT?

#### Resources

- 📖 Reading: FineTuningLLMs Chapter 0 (Sections 1-3)
- 📺 Video: Introduction to Fine-Tuning (supplemental)
- 🔗 Reference: Hugging Face Fine-Tuning Documentation

---

### Module 02: Transformer Architecture Review

**Source:** FineTuningLLMs Chapter 0  
**Duration:** 3-4 hours  
**Difficulty:** Intermediate

#### Learning Objectives

By the end of this module, students will be able to:
- Explain the transformer architecture components
- Describe how attention mechanisms work
- Understand the role of embeddings in LLMs
- Identify key architectural choices in modern LLMs

#### Theory Content

| Topic | Source | Enhancement |
|-------|--------|-------------|
| Transformer Architecture | Chapter 0 | Add interactive diagrams |
| Self-Attention Mechanism | Chapter 0 | Add mathematical derivation |
| Positional Encodings | Chapter 0 | Add visualization |
| Feed-Forward Networks | Chapter 0 | Add role explanation |
| Layer Normalization | Chapter 0 | Add before/after comparison |
| Modern LLM Architectures | Chapter 0 | Add Llama, Mistral comparison |

#### Hands-On Lab

**Lab 2.1:** Exploring Model Architecture
```python
# Students will:
# 1. Load different model architectures
# 2. Inspect model configuration
# 3. Visualize attention patterns
# 4. Compare architecture parameters
```

#### Knowledge Check (8 Questions)

1. What are the main components of a transformer?
2. How does self-attention work?
3. What is the purpose of positional encoding?
4. What is the difference between encoder and decoder?
5. How do modern LLMs differ from original transformers?
6. What is causal attention?
7. What role does layer normalization play?
8. How do you inspect a model's architecture in code?

#### Resources

- 📖 Reading: FineTuningLLMs Chapter 0 (Sections 4-6)
- 📺 Video: Transformer Architecture Explained (supplemental)
- 🔗 Reference: "Attention Is All You Need" paper

---

### Module 03: Quantization Fundamentals

**Source:** FineTuningLLMs Chapter 1  
**Duration:** 3-4 hours  
**Difficulty:** Intermediate

#### Learning Objectives

By the end of this module, students will be able to:
- Explain the purpose and benefits of quantization
- Distinguish between 8-bit, 4-bit, and other quantization levels
- Implement quantization using BitsAndBytes
- Understand the trade-offs between precision and performance

#### Theory Content

| Topic | Source | Enhancement |
|-------|--------|-------------|
| What is Quantization? | Chapter 1 | Add visual comparison |
| 8-bit Quantization | Chapter 1 | Add use cases |
| 4-bit Quantization | Chapter 1 | Add memory savings calc |
| BitsAndBytes Library | Chapter 1 | Add installation guide |
| Quantization-Aware Training | Chapter 1 | Add advanced note |
| Trade-offs and Considerations | Chapter 1 | Add decision matrix |

#### Hands-On Lab

**Lab 3.1:** Quantization Comparison
```python
# Students will:
# 1. Load a model in full precision
# 2. Load the same model in 8-bit
# 3. Load the same model in 4-bit
# 4. Compare memory usage
# 5. Compare inference speed
# 6. Compare output quality
```

#### Knowledge Check (8 Questions)

1. What is quantization and why is it important?
2. What is the memory savings from 8-bit vs 16-bit?
3. What is the memory savings from 4-bit vs 16-bit?
4. What library is commonly used for LLM quantization?
5. What are the trade-offs of lower-bit quantization?
6. When should you use 8-bit vs 4-bit?
7. How does quantization affect model quality?
8. What is quantization-aware training?

#### Resources

- 📖 Reading: FineTuningLLMs Chapter 1
- 📺 Video: Quantization Explained (supplemental)
- 🔗 Reference: BitsAndBytes Documentation
- 📊 Tool: Memory Calculator (supplemental)

---

### Module 04: Setting Up GPU Environment

**Source:** FineTuningLLMs Appendix A  
**Duration:** 2-3 hours  
**Difficulty:** Beginner

#### Learning Objectives

By the end of this module, students will be able to:
- Set up a GPU-enabled Python environment
- Install required libraries for fine-tuning
- Verify GPU availability and configuration
- Choose appropriate hardware options (local vs cloud)

#### Theory Content

| Topic | Source | Enhancement |
|-------|--------|-------------|
| Hardware Requirements | Appendix A | Add cost comparison |
| Local GPU Setup | Appendix A | Add troubleshooting |
| Cloud Options (Colab, Runpod) | Appendix A | Add pricing table |
| Python Environment Setup | Appendix A | Add conda/venv options |
| Library Installation | Appendix A | Add version pinning |
| Verification Steps | Appendix A | Add test script |

#### Hands-On Lab

**Lab 4.1:** Environment Setup
```python
# Students will:
# 1. Create virtual environment
# 2. Install required packages
# 3. Verify GPU availability
# 4. Run test inference
# 5. Document environment configuration
```

#### Knowledge Check (5 Questions)

1. What is the minimum VRAM for fine-tuning?
2. What are the advantages of Colab for learning?
3. How do you verify GPU availability in Python?
4. What are the key libraries for fine-tuning?
5. What cloud options are available for fine-tuning?

#### Resources

- 📖 Reading: FineTuningLLMs Appendix A
- 📺 Video: Setting Up Your Environment (supplemental)
- 🔗 Reference: CUDA Installation Guide
- 💰 Tool: Cloud Cost Calculator (supplemental)

---

### Module 05: Data Preparation & Formatting

**Source:** FineTuningLLMs Chapter 2  
**Duration:** 4-5 hours  
**Difficulty:** Intermediate

#### Learning Objectives

By the end of this module, students will be able to:
- Identify appropriate datasets for fine-tuning
- Format data for instruction tuning
- Create custom datasets from raw text
- Split data into train/validation/test sets
- Handle common data quality issues

#### Theory Content

| Topic | Source | Enhancement |
|-------|--------|-------------|
| Dataset Types | Chapter 2 | Add taxonomy |
| Instruction Format | Chapter 2 | Add templates |
| Data Quality Considerations | Chapter 2 | Add checklist |
| Data Preprocessing | Chapter 2 | Add pipeline |
| Train/Val/Test Splitting | Chapter 2 | Add best practices |
| Custom Dataset Creation | Chapter 2 | Add step-by-step |

#### Hands-On Lab

**Lab 5.1:** Data Preparation Pipeline
```python
# Students will:
# 1. Load a raw dataset
# 2. Clean and preprocess data
# 3. Format for instruction tuning
# 4. Create train/val/test splits
# 5. Save in appropriate format
# 6. Validate data quality
```

#### Knowledge Check (10 Questions)

1. What are common dataset formats for fine-tuning?
2. What is the instruction format structure?
3. How do you handle missing data?
4. What is a good train/val/test split ratio?
5. How do you ensure data quality?
6. What are common data preprocessing steps?
7. How do you create a custom dataset?
8. What is data deduplication and why is it important?
9. How do you handle imbalanced datasets?
10. What tools are available for data preparation?

#### Resources

- 📖 Reading: FineTuningLLMs Chapter 2 (Sections 1-4)
- 📺 Video: Data Preparation Best Practices (supplemental)
- 🔗 Reference: Hugging Face Datasets Documentation
- 📊 Dataset: Sample datasets for practice

---

### Module 06: Tokenization & Chat Templates

**Source:** FineTuningLLMs Chapter 2  
**Duration:** 3-4 hours  
**Difficulty:** Intermediate

#### Learning Objectives

By the end of this module, students will be able to:
- Explain the tokenization process
- Use appropriate tokenizers for different models
- Apply chat templates for conversational models
- Handle special tokens correctly
- Debug tokenization issues

#### Theory Content

| Topic | Source | Enhancement |
|-------|--------|-------------|
| Tokenization Basics | Chapter 2 | Add visual examples |
| Different Tokenizer Types | Chapter 2 | Add comparison |
| Model-Specific Tokenizers | Chapter 2 | Add examples |
| Chat Templates | Chapter 2 | Add template examples |
| Special Tokens | Chapter 2 | Add reference table |
| Tokenization Troubleshooting | Chapter 2 | Add common issues |

#### Hands-On Lab

**Lab 6.1:** Tokenization Practice
```python
# Students will:
# 1. Load different tokenizers
# 2. Tokenize sample text
# 3. Decode tokens back to text
# 4. Apply chat templates
# 5. Handle special tokens
# 6. Compare tokenization across models
```

#### Knowledge Check (8 Questions)

1. What is tokenization and why is it needed?
2. What are the main types of tokenizers?
3. How do you load a tokenizer for a specific model?
4. What are chat templates used for?
5. What are special tokens and when are they used?
6. How do you handle unknown tokens?
7. What is the difference between encode and decode?
8. How do you debug tokenization issues?

#### Resources

- 📖 Reading: FineTuningLLMs Chapter 2 (Sections 5-7)
- 📺 Video: Tokenization Deep Dive (supplemental)
- 🔗 Reference: Hugging Face Tokenizers Documentation
- 🛠️ Tool: Tokenizer Visualizer (supplemental)

---

### Module 07: PEFT & LoRA Theory

**Source:** FineTuningLLMs Chapter 3  
**Duration:** 3-4 hours  
**Difficulty:** Advanced

#### Learning Objectives

By the end of this module, students will be able to:
- Explain the concept of Parameter-Efficient Fine-Tuning
- Describe how LoRA (Low-Rank Adaptation) works
- Understand the mathematical foundation of LoRA
- Compare LoRA to other PEFT methods
- Configure LoRA hyperparameters

#### Theory Content

| Topic | Source | Enhancement |
|-------|--------|-------------|
| What is PEFT? | Chapter 3 | Add taxonomy |
| LoRA Mathematical Foundation | Chapter 3 | Add derivation |
| LoRA Architecture | Chapter 3 | Add diagrams |
| LoRA Hyperparameters | Chapter 3 | Add tuning guide |
| Comparison with Other PEFT Methods | Chapter 3 | Add comparison table |
| When to Use LoRA | Chapter 3 | Add decision tree |

#### Hands-On Lab

**Lab 7.1:** LoRA Configuration Analysis
```python
# Students will:
# 1. Calculate parameter counts for different LoRA configs
# 2. Compare memory requirements
# 3. Analyze rank impact on performance
# 4. Plan LoRA configuration for use case
```

#### Knowledge Check (10 Questions)

1. What is Parameter-Efficient Fine-Tuning?
2. How does LoRA reduce trainable parameters?
3. What is the mathematical formulation of LoRA?
4. What are the key LoRA hyperparameters?
5. What does the rank (r) parameter control?
6. What is LoRA alpha and how is it used?
7. Which layers are typically adapted with LoRA?
8. How does LoRA compare to adapter methods?
9. What are the memory savings with LoRA?
10. When should you use LoRA vs full fine-tuning?

#### Resources

- 📖 Reading: FineTuningLLMs Chapter 3 (Sections 1-3)
- 📺 Video: LoRA Explained (supplemental)
- 🔗 Reference: LoRA Original Paper
- 📊 Tool: Parameter Calculator (supplemental)

---

### Module 08: LoRA Implementation

**Source:** FineTuningLLMs Chapter 3  
**Duration:** 4-5 hours  
**Difficulty:** Advanced

#### Learning Objectives

By the end of this module, students will be able to:
- Configure LoRA using PEFT library
- Apply LoRA to a pre-trained model
- Set up training with LoRA adapters
- Monitor LoRA training progress
- Save and load LoRA adapters

#### Theory Content

| Topic | Source | Enhancement |
|-------|--------|-------------|
| PEFT Library Setup | Chapter 3 | Add installation |
| LoRA Configuration | Chapter 3 | Add config examples |
| Applying LoRA to Models | Chapter 3 | Add code patterns |
| Training Loop with LoRA | Chapter 3 | Add complete example |
| Saving LoRA Adapters | Chapter 3 | Add best practices |
| Loading and Merging Adapters | Chapter 3 | Add deployment notes |

#### Hands-On Lab

**Lab 8.1:** LoRA Fine-Tuning
```python
# Students will:
# 1. Configure LoRA with PEFT
# 2. Apply LoRA to a base model
# 3. Set up training arguments
# 4. Run fine-tuning loop
# 5. Save LoRA adapters
# 6. Load and test adapters
```

#### Knowledge Check (8 Questions)

1. How do you install and import the PEFT library?
2. What is LoraConfig and what parameters does it take?
3. How do you apply LoRA to a pre-trained model?
4. What is get_peft_model()?
5. How do you save LoRA adapters after training?
6. How do you load LoRA adapters for inference?
7. What is adapter merging and when is it useful?
8. How do you evaluate a LoRA fine-tuned model?

#### Resources

- 📖 Reading: FineTuningLLMs Chapter 3 (Sections 4-6)
- 📺 Video: LoRA Implementation Walkthrough (supplemental)
- 🔗 Reference: PEFT Library Documentation
- 💻 Code: Complete LoRA example notebook

---

### Module 09: SFTTrainer Deep Dive

**Source:** FineTuningLLMs Chapter 4  
**Duration:** 5-6 hours  
**Difficulty:** Advanced

#### Learning Objectives

By the end of this module, students will be able to:
- Explain the purpose of SFTTrainer
- Configure SFTTrainer for fine-tuning
- Set up training arguments and callbacks
- Monitor training metrics
- Handle common training issues

#### Theory Content

| Topic | Source | Enhancement |
|-------|--------|-------------|
| What is SFTTrainer? | Chapter 4 | Add use cases |
| SFTTrainer Configuration | Chapter 4 | Add config reference |
| Training Arguments | Chapter 4 | Add parameter guide |
| Data Collators | Chapter 4 | Add examples |
| Callbacks and Logging | Chapter 4 | Add monitoring setup |
| Checkpointing and Resume | Chapter 4 | Add best practices |

#### Hands-On Lab

**Lab 9.1:** Complete Fine-Tuning Pipeline
```python
# Students will:
# 1. Prepare dataset for SFTTrainer
# 2. Configure SFTTrainer
# 3. Set up TrainingArguments
# 4. Run training with monitoring
# 5. Save trained model
# 6. Evaluate results
```

#### Knowledge Check (8 Questions)

1. What is SFTTrainer and when should you use it?
2. What are the key SFTTrainer parameters?
3. How do you configure TrainingArguments?
4. What is a data collator and why is it needed?
5. How do you monitor training progress?
6. What callbacks are available for SFTTrainer?
7. How do you save and load checkpoints?
8. How do you resume training from a checkpoint?

#### Resources

- 📖 Reading: FineTuningLLMs Chapter 4
- 📺 Video: SFTTrainer Complete Guide (supplemental)
- 🔗 Reference: Hugging Face Trainer Documentation
- 💻 Code: Complete training pipeline notebook

---

### Module 10: Flash Attention & Optimization

**Source:** FineTuningLLMs Chapter 5  
**Duration:** 3-4 hours  
**Difficulty:** Advanced

#### Learning Objectives

By the end of this module, students will be able to:
- Explain the benefits of Flash Attention
- Install and configure Flash Attention
- Optimize training with memory-efficient techniques
- Profile and benchmark training performance
- Apply additional optimization techniques

#### Theory Content

| Topic | Source | Enhancement |
|-------|--------|-------------|
| Attention Computational Complexity | Chapter 5 | Add complexity analysis |
| Flash Attention Algorithm | Chapter 5 | Add visual explanation |
| Memory-Efficient Attention | Chapter 5 | Add comparison |
| Flash Attention Installation | Chapter 5 | Add platform guide |
| Performance Benchmarking | Chapter 5 | Add benchmark scripts |
| Additional Optimizations | Chapter 5 | Add gradient checkpointing |

#### Hands-On Lab

**Lab 10.1:** Optimization Comparison
```python
# Students will:
# 1. Run training without optimizations
# 2. Run training with Flash Attention
# 3. Compare memory usage
# 4. Compare training speed
# 5. Profile training performance
# 6. Document optimization impact
```

#### Knowledge Check (6 Questions)

1. What is the computational complexity of standard attention?
2. How does Flash Attention improve efficiency?
3. What are the memory savings with Flash Attention?
4. How do you install Flash Attention?
5. What other optimization techniques complement Flash Attention?
6. How do you profile training performance?

#### Resources

- 📖 Reading: FineTuningLLMs Chapter 5
- 📺 Video: Flash Attention Explained (supplemental)
- 🔗 Reference: Flash Attention Paper
- 📊 Tool: Training Profiler (supplemental)

---

### Module 11: GGUF Conversion

**Source:** FineTuningLLMs Chapter 6  
**Duration:** 3-4 hours  
**Difficulty:** Advanced

#### Learning Objectives

By the end of this module, students will be able to:
- Explain the GGUF format and its benefits
- Convert fine-tuned models to GGUF
- Understand quantization options for GGUF
- Optimize GGUF models for inference
- Troubleshoot conversion issues

#### Theory Content

| Topic | Source | Enhancement |
|-------|--------|-------------|
| What is GGUF? | Chapter 6 | Add format comparison |
| GGUF Benefits | Chapter 6 | Add use cases |
| llama.cpp Overview | Chapter 6 | Add architecture |
| Conversion Process | Chapter 6 | Add step-by-step |
| GGUF Quantization Options | Chapter 6 | Add quality comparison |
| Conversion Troubleshooting | Chapter 6 | Add common issues |

#### Hands-On Lab

**Lab 11.1:** Model Conversion
```python
# Students will:
# 1. Prepare fine-tuned model for conversion
# 2. Install llama.cpp
# 3. Convert model to GGUF
# 4. Apply GGUF quantization
# 5. Test converted model
# 6. Compare with original
```

#### Knowledge Check (8 Questions)

1. What is the GGUF format?
2. What are the benefits of GGUF over other formats?
3. What is llama.cpp and what is its role?
4. How do you convert a model to GGUF?
5. What quantization options are available for GGUF?
6. How does GGUF quantization affect quality?
7. What are common conversion issues?
8. How do you test a converted GGUF model?

#### Resources

- 📖 Reading: FineTuningLLMs Chapter 6 (Sections 1-3)
- 📺 Video: GGUF Conversion Guide (supplemental)
- 🔗 Reference: llama.cpp Documentation
- 🛠️ Tool: GGUF Conversion Script

---

### Module 12: Local Deployment with Ollama

**Source:** FineTuningLLMs Chapter 6  
**Duration:** 4-5 hours  
**Difficulty:** Advanced

#### Learning Objectives

By the end of this module, students will be able to:
- Install and configure Ollama
- Import GGUF models into Ollama
- Create custom model configurations
- Serve models via API
- Integrate fine-tuned models into applications

#### Theory Content

| Topic | Source | Enhancement |
|-------|--------|-------------|
| Ollama Overview | Chapter 6 | Add architecture |
| Ollama Installation | Chapter 6 | Add platform guide |
| Model Import Process | Chapter 6 | Add step-by-step |
| Modelfile Configuration | Chapter 6 | Add template examples |
| API Usage | Chapter 6 | Add code examples |
| Application Integration | Chapter 6 | Add integration patterns |

#### Hands-On Lab

**Lab 12.1:** Complete Deployment
```python
# Students will:
# 1. Install Ollama
# 2. Import fine-tuned GGUF model
# 3. Create Modelfile configuration
# 4. Test model via CLI
# 5. Test model via API
# 6. Build simple application integration
```

#### Knowledge Check (8 Questions)

1. What is Ollama and what is it used for?
2. How do you install Ollama?
3. How do you import a GGUF model into Ollama?
4. What is a Modelfile and what does it contain?
5. How do you test a model with Ollama CLI?
6. How do you use the Ollama API?
7. How do you integrate Ollama into an application?
8. What are best practices for production deployment?

#### Resources

- 📖 Reading: FineTuningLLMs Chapter 6 (Sections 4-6)
- 📺 Video: Ollama Deployment Guide (supplemental)
- 🔗 Reference: Ollama Documentation
- 💻 Code: Application integration examples

---

## Capstone Project

### Project: End-to-End Fine-Tuning Pipeline

**Duration:** 2-3 weeks  
**Difficulty:** Advanced

#### Project Description

Students will complete a full fine-tuning pipeline:
1. Select a use case and dataset
2. Prepare and format data
3. Configure and run fine-tuning with LoRA
4. Optimize with Flash Attention
5. Convert to GGUF format
6. Deploy locally with Ollama
7. Evaluate and document results

#### Deliverables

1. **Code Repository** - Complete fine-tuning pipeline
2. **Technical Report** - Documentation of approach and results
3. **Demo Application** - Working application using fine-tuned model
4. **Presentation** - 10-minute presentation of project

#### Evaluation Rubric

| Criterion | Weight | Excellent (4) | Good (3) | Satisfactory (2) | Needs Improvement (1) |
|-----------|--------|---------------|----------|------------------|----------------------|
| Data Preparation | 15% | Clean, well-formatted, documented | Good formatting | Basic formatting | Poor formatting |
| Fine-Tuning Implementation | 25% | Optimal config, efficient training | Good config | Basic training | Issues with training |
| Optimization | 15% | Multiple optimizations applied | Flash Attention used | Basic optimization | No optimization |
| Deployment | 20% | Production-ready deployment | Working deployment | Basic deployment | Deployment issues |
| Evaluation | 15% | Comprehensive evaluation | Good evaluation | Basic evaluation | Limited evaluation |
| Documentation | 10% | Excellent documentation | Good documentation | Basic documentation | Poor documentation |

---

## Assessment Strategy

### Formative Assessments

| Assessment Type | Frequency | Purpose |
|-----------------|-----------|---------|
| Knowledge Checks | Per module | Verify understanding |
| Lab Completion | Per module | Hands-on practice |
| Code Reviews | Modules 8, 9, 10 | Code quality feedback |

### Summative Assessments

| Assessment | Weight | Timing |
|------------|--------|--------|
| Module Quizzes | 30% | After each module |
| Lab Assignments | 30% | After modules 5, 8, 9, 11, 12 |
| Capstone Project | 40% | End of track |

### Passing Criteria

- Minimum 70% overall score
- Complete all required labs
- Submit capstone project

---

## Resource Summary

### Primary Resources (FineTuningLLMs)

| Resource | Modules | URL |
|----------|---------|-----|
| Chapter 0 | 1, 2 | https://github.com/dvgodoy/FineTuningLLMs |
| Chapter 1 | 3 | https://github.com/dvgodoy/FineTuningLLMs |
| Chapter 2 | 5, 6 | https://github.com/dvgodoy/FineTuningLLMs |
| Chapter 3 | 7, 8 | https://github.com/dvgodoy/FineTuningLLMs |
| Chapter 4 | 9 | https://github.com/dvgodoy/FineTuningLLMs |
| Chapter 5 | 10 | https://github.com/dvgodoy/FineTuningLLMs |
| Chapter 6 | 11, 12 | https://github.com/dvgodoy/FineTuningLLMs |
| Appendix A | 4 | https://github.com/dvgodoy/FineTuningLLMs |

### Supplemental Resources

| Resource | Purpose | URL |
|----------|---------|-----|
| Hugging Face Docs | Reference | https://huggingface.co/docs |
| PEFT Documentation | LoRA reference | https://huggingface.co/docs/peft |
| llama.cpp | GGUF conversion | https://github.com/ggerganov/llama.cpp |
| Ollama | Local deployment | https://ollama.ai |
| Flash Attention | Optimization | https://github.com/Dao-AILab/flash-attention |

---

## Attribution

This track incorporates content from the FineTuningLLMs repository:

- **Original Author:** dvgodoy (Daniel Godoy)
- **Repository:** https://github.com/dvgodoy/FineTuningLLMs
- **License:** MIT License
- **Associated Book:** "A Hands-On Guide to Fine-Tuning LLMs with PyTorch and Hugging Face"

All adaptations maintain proper attribution and comply with the MIT License.

---

*Document Version: 1.0*  
*Created: March 30, 2026*  
*For: AI-Mastery-2026 Curriculum*
