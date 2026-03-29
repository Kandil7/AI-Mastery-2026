# The LLM Scientist - Production-Ready Implementation

A comprehensive collection of production-grade implementations covering the complete LLM development lifecycle, from architecture to evaluation.

## 📚 Table of Contents

- [Module 2.1: LLM Architecture](#module-21-llm-architecture)
- [Module 2.2: Pre-Training Models](#module-22-pre-training-models)
- [Module 2.3: Post-Training Datasets](#module-23-post-training-datasets)
- [Module 2.4: Supervised Fine-Tuning](#module-24-supervised-fine-tuning)
- [Module 2.5: Preference Alignment](#module-25-preference-alignment)
- [Module 2.6: Evaluation](#module-26-evaluation)
- [Module 2.7: Quantization](#module-27-quantization)
- [Module 2.8: New Trends](#module-28-new-trends)

## 🏗️ Module 2.1: LLM Architecture

Core building blocks of Large Language Models.

### Files

| File | Description |
|------|-------------|
| `attention.py` | Multi-head, self, masked, cross, and flash attention |
| `transformer.py` | Complete transformer encoder-decoder with positional encodings |
| `tokenization.py` | BPE, WordPiece, SentencePiece tokenizers |
| `sampling.py` | Greedy, beam, temperature, top-p, top-k, contrastive sampling |

### Key Components

```python
from llm_scientist.module_2_1_llm_architecture import (
    MultiHeadAttention,
    Transformer,
    BPETokenizer,
    TopPSampler,
)

# Multi-head attention
mha = MultiHeadAttention(d_model=512, num_heads=8)

# Complete transformer
model = Transformer(vocab_size=30522, d_model=512, num_heads=8)

# BPE tokenizer
tokenizer = BPETokenizer(vocab_size=30000)
tokenizer.train(texts)

# Top-p sampling
sampler = TopPSampler(p=0.9, temperature=0.8)
```

## 🎯 Module 2.2: Pre-Training Models

Large-scale pre-training components.

### Files

| File | Description |
|------|-------------|
| `data_prep.py` | Data collection, cleaning, deduplication (MinHash/LSH), quality scoring |
| `distributed_training.py` | Data/pipeline/tensor parallelism, FSDP, DeepSpeed ZeRO |
| `optimization.py` | Mixed precision, gradient checkpointing, LR scheduling |
| `monitoring.py` | Loss tracking, GPU monitoring, memory profiling, W&B integration |

### Key Components

```python
from llm_scientist.module_2_2_pretraining import (
    PreTrainingDataPipeline,
    FSDPTrainer,
    MixedPrecisionTrainer,
    TrainingMonitor,
)

# Data pipeline
pipeline = PreTrainingDataPipeline()
documents = pipeline.run(sources=['wikipedia', 'books'])

# FSDP training
trainer = FSDPTrainer(model, config)
model = trainer.prepare_model()

# Monitoring
monitor = TrainingMonitor(use_wandb=True)
```

## 📝 Module 2.3: Post-Training Datasets

Dataset preparation for post-training.

### Files

| File | Description |
|------|-------------|
| `formats.py` | ShareGPT, ChatML, Alpaca formats, conversation templates |
| `synthetic_data.py` | Instruction generation, Self-Instruct pipeline |
| `enhancement.py` | Chain-of-thought, Branch-Solve-Merge, self-reflection |
| `quality_filtering.py` | Reward model filtering, perplexity filtering, diversity scoring |

### Key Components

```python
from llm_scientist.module_2_3_post_training import (
    FormatConverter,
    SelfInstruct,
    DataEnhancementPipeline,
    QualityFilterPipeline,
)

# Format conversion
converter = FormatConverter()
conversations = converter.convert('sharegpt', 'chatml', input_path='data.jsonl')

# Self-Instruct
self_instruct = SelfInstruct(model, tokenizer)
instructions = self_instruct.run()

# Enhancement
enhancer = DataEnhancementPipeline(model, tokenizer)
enhanced = enhancer.enhance_dataset(examples, methods=['cot', 'correction'])
```

## 🔧 Module 2.4: Supervised Fine-Tuning

SFT implementation with LoRA and QLoRA.

### Files

| File | Description |
|------|-------------|
| `sft.py` | Full fine-tuning pipeline with checkpointing |
| `lora.py` | LoRA implementation from scratch |
| `qlora.py` | 4-bit quantization + LoRA (QLoRA) |
| `distributed.py` | Multi-GPU fine-tuning with FSDP/DeepSpeed |

### Key Components

```python
from llm_scientist.module_2_4_sft import (
    SFTTrainer,
    LoRAModel,
    QLoRATrainer,
    FSDPSFTTrainer,
)

# Full SFT
trainer = SFTTrainer(model, config, tokenizer, dataset)
trainer.train()

# LoRA
lora_model = LoRAModel(model, LoRAConfig(r=8, alpha=16))
lora_model.enable_adapter_layers()

# QLoRA
qlora_trainer = QLoRATrainer(qlora_model, config, tokenizer, dataset)
qlora_trainer.train()
```

## 🎲 Module 2.5: Preference Alignment

Preference alignment techniques.

### Files

| File | Description |
|------|-------------|
| `rejection_sampling.py` | Multi-response generation, preference pair creation |
| `dpo.py` | Direct Preference Optimization implementation |
| `rlhf.py` | PPO for RLHF with reward/value models |
| `reward_modeling.py` | Reward model training with Bradley-Terry loss |

### Key Components

```python
from llm_scientist.module_2_5_preference import (
    RejectionSampler,
    DPOTrainer,
    PPOTrainer,
    RewardModelTrainer,
)

# Rejection sampling
sampler = RejectionSampler(model, tokenizer, reward_model)
pairs = sampler.sample(prompts)

# DPO
dpo_trainer = DPOTrainer(policy_model, reference_model, config, tokenizer, dataset)
dpo_trainer.train()

# RLHF with PPO
ppo_trainer = PPOTrainer(policy, value, reward, config, tokenizer)
ppo_trainer.train(prompts)
```

## 📊 Module 2.6: Evaluation

Comprehensive evaluation framework.

### Files

| File | Description |
|------|-------------|
| `benchmarks.py` | MMLU, TruthfulQA, GSM8K, HumanEval implementations |
| `human_eval.py` | Human evaluation interface with quality control |
| `model_based_eval.py` | LLM-as-judge, pairwise comparison |
| `feedback_analysis.py` | Error categorization, pattern detection |

### Key Components

```python
from llm_scientist.module_2_6_evaluation import (
    BenchmarkRunner,
    HumanEvaluator,
    ModelBasedEvaluator,
    FeedbackAnalyzer,
)

# Benchmark evaluation
runner = BenchmarkRunner(model, tokenizer)
results = runner.run_all()  # MMLU, TruthfulQA, GSM8K, HumanEval

# Human evaluation
human_eval = HumanEvaluator()
human_eval.create_tasks(prompts, responses)

# Model-based evaluation
judge = ModelBasedEvaluator(judge_model, tokenizer)
results = judge.evaluate_all(prompts, responses)
```

## 🔢 Module 2.7: Quantization

Model quantization techniques.

### Files

| File | Description |
|------|-------------|
| `base_quant.py` | FP32, FP16, INT8, zero-point, absmax quantization |
| `gguf.py` | GGUF format conversion for llama.cpp |
| `gptq.py` | GPTQ layer-by-layer quantization |
| `awq.py` | Activation-aware weight quantization |
| `exl2.py` | EXL2 2-8 bit quantization |

### Key Components

```python
from llm_scientist.module_2_7_quantization import (
    quantize_model,
    GPTQQuantizer,
    AWQQuantizer,
    EXL2Quantizer,
)

# Basic quantization
config = QuantizationConfig(quant_type=QuantizationType.INT8)
quantized_model = quantize_model(model, config)

# GPTQ
gptq = GPTQQuantizer(GPTQConfig(bits=4))
quantized = gptq.quantize_model(model, calibration_data)

# AWQ
awq = AWQQuantizer(AWQConfig(bits=4))
quantized = awq.quantize_model(model, calibration_data)
```

## 🚀 Module 2.8: New Trends

Cutting-edge LLM techniques.

### Files

| File | Description |
|------|-------------|
| `model_merging.py` | SLERP, DARE, TIES, task arithmetic, model soups |
| `multimodal.py` | CLIP-style vision-language, LLaVA architecture |
| `interpretability.py` | Sparse autoencoders, feature visualization, activation patching |
| `test_time_compute.py` | Chain-of-thought, self-consistency, verification |

### Key Components

```python
from llm_scientist.module_2_8_new_trends import (
    merge_models,
    LLaVAModel,
    InterpretabilityAnalyzer,
    TestTimeCompute,
)

# Model merging
merged = merge_models([model1, model2], method='slerp', weights=[0.5, 0.5])

# Multimodal
vlm = LLaVAModel(config, language_model)
outputs = vlm(input_ids, attention_mask, images)

# Interpretability
analyzer = InterpretabilityAnalyzer(model, config)
results = analyzer.analyze_module('layer.0', data)

# Test-time compute
ttc = TestTimeCompute(model, tokenizer, config)
result = ttc.solve(problem, use_self_consistency=True)
```

## 📦 Installation

```bash
# Install dependencies
pip install torch>=2.0.0
pip install transformers>=4.30.0
pip install datasets>=2.14.0
pip install accelerate>=0.20.0
pip install wandb>=0.15.0
pip install tensorboard>=2.13.0

# Optional: Flash Attention
pip install flash-attn --no-build-isolation

# Optional: DeepSpeed
pip install deepspeed>=0.10.0
```

## 🧪 Usage Examples

### Complete Fine-Tuning Pipeline

```python
from llm_scientist import (
    SFTConfig, SFTTrainer,
    LoRAConfig, LoRAModel,
    PreTrainingDataPipeline,
)

# 1. Prepare data
pipeline = PreTrainingDataPipeline()
documents = pipeline.run(sources=['alpaca'])

# 2. Load model
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf')
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')

# 3. Apply LoRA
lora_model = LoRAModel(model, LoRAConfig(r=16, alpha=32))

# 4. Train
config = SFTConfig(
    output_dir='./sft_output',
    num_train_epochs=3,
    per_device_train_batch_size=4,
)
trainer = SFTTrainer(lora_model, config, tokenizer, dataset)
trainer.train()
```

### DPO Fine-Tuning

```python
from llm_scientist import (
    DPOConfig, DPOTrainer,
    RejectionSampler,
)

# 1. Generate preference data
sampler = RejectionSampler(model, tokenizer, reward_model)
pairs = sampler.sample(prompts)

# 2. Train with DPO
config = DPOConfig(
    beta=0.1,
    learning_rate=5e-7,
    num_train_epochs=3,
)
trainer = DPOTrainer(policy_model, reference_model, config, tokenizer, dataset)
trainer.train()
```

### Model Evaluation

```python
from llm_scientist import BenchmarkRunner

runner = BenchmarkRunner(model, tokenizer, output_dir='./eval')

# Run all benchmarks
results = runner.run_all()

# Access results
print(f"MMLU Accuracy: {results['mmlu'].accuracy:.4f}")
print(f"GSM8K Accuracy: {results['gsm8k'].accuracy:.4f}")
print(f"HumanEval Pass@1: {results['humaneval'].pass_at_k[1]:.4f}")
```

## 📄 License

MIT License

## 🙏 Acknowledgments

This implementation draws from numerous research papers and open-source projects. Key references are cited in each module's documentation.
