# Arabic LLM Autonomous Research System - Improvements

## تحسينات نظام البحث الآلي للغة العربية

**Based on**: Latest Karpathy autoresearch patterns  
**Date**: March 26, 2026  
**Version**: 2.1.0  

---

## 🔍 Current State Analysis

### What We Have ✅
- Complete package structure (arabic_llm/ with 26 modules)
- Autonomous research agent (agents/researcher.py)
- Experiment proposals system (agents/proposals.py)
- Experiment evaluator (agents/evaluator.py)
- Training infrastructure (models/qlora.py)
- 8,424 books processed
- 61,500+ training examples

### What We Can Improve 🎯

Based on Karpathy's autoresearch pattern, here are key improvements:

---

## 🚀 Key Improvements

### 1. Simplified File Structure

**Current Issue**: Too many files, complex structure  
**Improvement**: Follow autoresearch's minimal 3-file pattern

```
arabic-llm/
├── prepare_data.py       # Fixed: Data prep, constants (DO NOT MODIFY)
├── train_model.py        # Agent modifies: Model, optimizer, training
├── program.md            # Human modifies: Agent instructions
├── run_agent.py          # Agent loop controller
└── analysis.ipynb        # Results analysis
```

**Benefits**:
- Clearer separation of concerns
- Easier for agent to understand scope
- Faster iteration cycle
- Simpler debugging

---

### 2. Fixed Time Budget Training

**Current Issue**: No fixed time budget  
**Improvement**: Implement 5-minute wall-clock budget

```python
# train_model.py
TIME_BUDGET_SECONDS = 300  # 5 minutes

def train():
    start_time = time.time()
    
    for epoch in range(max_epochs):
        # Check time budget
        if time.time() - start_time > TIME_BUDGET_SECONDS:
            print(f"⏰ Time budget reached")
            break
        
        # Training step
        ...
    
    # Evaluate even if interrupted
    val_loss = evaluate()
    return val_loss
```

**Benefits**:
- Comparable experiments regardless of changes
- Finds optimal model for your hardware
- ~12 experiments/hour, ~100/night

---

### 3. Better Primary Metric

**Current Issue**: Using val_loss (vocab-size dependent)  
**Improvement**: Use val_bpb (bits per byte) or perplexity per token

```python
def compute_val_bpb(model, val_loader):
    """
    Compute validation bits per byte.
    Vocab-size-independent metric.
    Lower is better.
    """
    model.eval()
    total_bits = 0
    total_bytes = 0
    
    with torch.no_grad():
        for batch in val_loader:
            # Compute loss
            loss = model(...).loss
            
            # Convert to bits per byte
            bits = loss.item() * np.log2(2)  # nats to bits
            bytes_count = sum(len(text) for text in batch['text'])
            
            total_bits += bits * bytes_count
            total_bytes += bytes_count
    
    return total_bits / total_bytes
```

**Benefits**:
- Vocab-size-independent
- Fair comparison across architectures
- Standard metric in autoresearch

---

### 4. Smaller Compute Support

**Current Issue**: Optimized for H100/RTX 3090  
**Improvement**: Add configs for smaller GPUs (MacBook, etc.)

```python
# configs/small_compute.yaml
# For MacBooks / smaller GPUs

model:
  depth: 4              # Reduced from 8
  hidden_size: 256      # Reduced from 512
  num_heads: 4          # Reduced from 8

data:
  vocab_size: 4096      # Reduced from 151936
  max_seq_len: 512      # Reduced from 2048
  batch_size: 8         # Increased (smaller seq = more fit in memory)

training:
  total_batch_size: 16384  # Reduced from 2^16
  time_budget: 300         # Keep 5 minutes
```

**Benefits**:
- Accessible to more researchers
- Faster iteration on consumer hardware
- Still finds optimal model for platform

---

### 5. Better Experiment Proposals

**Current Issue**: Fixed list of 40+ proposals  
**Improvement**: Dynamic proposal generation based on results

```python
class AdaptiveProposalGenerator:
    """Generate proposals based on previous results"""
    
    def __init__(self, experiment_log):
        self.log = experiment_log
        self.best_params = self._get_best_params()
    
    def generate_next_proposal(self):
        """Generate next proposal based on trends"""
        
        # Analyze what's working
        improving_params = self._analyze_improvements()
        
        # Propose variations of successful params
        if 'LORA_R' in improving_params:
            # Try nearby values
            current_r = self.best_params.get('LORA_R', 64)
            return ExperimentProposal(
                change=f"Fine-tune LoRA rank around {current_r}",
                modifications={'LORA_R': current_r + 16}
            )
        
        # If stuck, try different category
        return self._explore_new_category()
```

**Benefits**:
- Learns from previous experiments
- Explores promising directions
- Avoids repeating failed experiments

---

### 6. Better Logging & Visualization

**Current Issue**: Basic JSONL logging  
**Improvement**: Real-time progress visualization

```python
class ExperimentTracker:
    """Track and visualize experiment progress"""
    
    def __init__(self, log_dir="experiments"):
        self.log_dir = Path(log_dir)
        self.experiments = []
    
    def log_experiment(self, exp_result):
        """Log experiment and update visualization"""
        self.experiments.append(exp_result)
        self._save_log()
        self._update_progress_plot()
    
    def _update_progress_plot(self):
        """Generate progress.png showing val_loss over time"""
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 4))
        
        # Plot val_loss over experiments
        plt.subplot(1, 2, 1)
        plt.plot([e['val_loss'] for e in self.experiments])
        plt.xlabel('Experiment')
        plt.ylabel('val_loss')
        plt.title('Progress Over Time')
        
        # Plot histogram of improvements
        plt.subplot(1, 2, 2)
        improvements = [e['improvement'] for e in self.experiments if e['improved']]
        plt.hist(improvements, bins=20)
        plt.xlabel('Improvement')
        plt.ylabel('Count')
        plt.title('Improvement Distribution')
        
        plt.tight_layout()
        plt.savefig(self.log_dir / 'progress.png')
        plt.close()
```

**Benefits**:
- Visual progress tracking
- Identify trends quickly
- Share results easily

---

### 7. Analysis Notebook

**Current Issue**: No analysis tools  
**Improvement**: Add analysis.ipynb

```python
# analysis.ipynb

# Load experiment log
import json
import pandas as pd

with open('experiments/experiment_log.jsonl') as f:
    experiments = [json.loads(line) for line in f]

df = pd.DataFrame(experiments)

# Analyze top improvements
top_improvements = df[df['improved']].nlargest(10, 'improvement')
print("Top 10 Improvements:")
print(top_improvements[['experiment', 'change', 'val_loss', 'improvement']])

# Analyze by category
df['category'] = df['change'].apply(extract_category)
category_stats = df.groupby('category').agg({
    'val_loss': 'mean',
    'improved': 'sum',
    'experiment': 'count'
})
print("\nPerformance by Category:")
print(category_stats)

# Plot learning curve
plt.figure(figsize=(10, 4))
plt.plot(df['val_loss'])
plt.xlabel('Experiment')
plt.ylabel('val_loss')
plt.title('Learning Curve')
plt.show()
```

**Benefits**:
- Deep dive into results
- Identify patterns
- Generate insights

---

### 8. Human-in-the-Loop Improvements

**Current Issue**: Fully autonomous  
**Improvement**: Add human review checkpoints

```python
class HumanReviewCheckpoint:
    """Pause for human review at key milestones"""
    
    def __init__(self, review_interval=20):
        self.review_interval = review_interval
    
    def check(self, tracker):
        """Check if human review is needed"""
        if len(tracker.experiments) % self.review_interval == 0:
            print(f"\n{'='*70}")
            print(f"HUMAN REVIEW CHECKPOINT - Experiment {len(tracker.experiments)}")
            print(f"{'='*70}")
            print(f"Best val_loss: {tracker.best_loss:.4f}")
            print(f"Improvements: {tracker.improved_count}/{len(tracker.experiments)}")
            print(f"\nReview progress.png and experiment_log.jsonl")
            print(f"Update program.md if needed, then press Enter to continue...")
            input()
```

**Benefits**:
- Human intuition guides research
- Catch agent mistakes early
- Iterate on "research org code" (program.md)

---

### 9. Better program.md Structure

**Current Issue**: Long, complex instructions  
**Improvement**: Structured, modular instructions

```markdown
# Arabic LLM Research Program v2.1

## Goal
Minimize val_bpb on Arabic text while maintaining:
- Arabic ratio > 70%
- Training stability (no divergence)
- Time budget: 5 minutes per experiment

## Current Best
- val_bpb: 1.234
- Configuration: LoRA r=128, lr=2e-4, depth=8

## Recent Learnings (Last 20 Experiments)
✅ LoRA rank 128 > 64 (consistent improvement)
✅ Learning rate 2e-4 > 1e-4 (faster convergence)
⚠️ Depth > 12 causes overfitting
⚠️ Batch size > 32 no improvement

## Next Directions to Explore
1. Fine-tune LoRA rank around 128 (try 144, 160)
2. Try AdamW betas (0.9, 0.95) vs (0.9, 0.999)
3. Experiment with warmup ratio (0.05, 0.10)

## Constraints
- Modify train_model.py ONLY
- Keep time budget at 300 seconds
- Monitor for divergence (val_loss > 2.0)
- Log all experiments to experiments/experiment_log.jsonl
```

**Benefits**:
- Clear, actionable instructions
- Recent learnings guide agent
- Constraints prevent mistakes

---

### 10. Integration with Arabic LLM Dataset

**Current Issue**: Generic training loop  
**Improvement**: Arabic-specific optimizations

```python
# train_model.py - Arabic-specific features

class ArabicLLMTrainer:
    """Trainer optimized for Arabic language"""
    
    def __init__(self, config):
        # Arabic-specific defaults
        self.config = {
            'vocab_size': 151936,  # Qwen2.5 Arabic vocab
            'arabic_ratio_threshold': 0.70,
            'diacritics_ratio_threshold': 0.30,
            **config
        }
    
    def validate_arabic_quality(self, val_loader):
        """Ensure model maintains Arabic quality"""
        arabic_ratios = []
        
        for batch in val_loader:
            text = batch['text']
            arabic_ratio = get_arabic_ratio(text)
            arabic_ratios.append(arabic_ratio)
        
        avg_ratio = sum(arabic_ratios) / len(arabic_ratios)
        
        if avg_ratio < self.config['arabic_ratio_threshold']:
            print(f"⚠️ WARNING: Arabic ratio {avg_ratio:.1%} below threshold!")
            return False
        
        return True
```

**Benefits**:
- Maintains Arabic quality
- Prevents degradation
- Domain-specific optimization

---

## 📋 Implementation Plan

### Phase 1: Core Improvements (Week 1)
- [ ] Simplify file structure (3-file pattern)
- [ ] Implement fixed time budget
- [ ] Add val_bpb metric
- [ ] Create analysis.ipynb

### Phase 2: Smaller Compute Support (Week 1)
- [ ] Add small_compute.yaml config
- [ ] Test on CPU/Mac
- [ ] Document compute requirements

### Phase 3: Better Proposals (Week 2)
- [ ] Implement adaptive proposal generator
- [ ] Add learning from results
- [ ] Prevent repeated failures

### Phase 4: Visualization (Week 2)
- [ ] Add experiment tracker
- [ ] Generate progress.png
- [ ] Real-time logging

### Phase 5: Human-in-the-Loop (Week 3)
- [ ] Add review checkpoints
- [ ] Improve program.md structure
- [ ] Document iteration process

### Phase 6: Arabic Optimization (Week 3)
- [ ] Add Arabic quality validation
- [ ] Domain-specific metrics
- [ ] Test on full dataset

---

## 🎯 Expected Results

### Before Improvements
- ~12 experiments/hour
- ~100 experiments/night
- Manual analysis
- Fixed proposal list
- No visualization

### After Improvements
- ~12 experiments/hour (same speed)
- ~100 experiments/night (same throughput)
- **Automated analysis** (better insights)
- **Adaptive proposals** (smarter exploration)
- **Real-time visualization** (better tracking)
- **Human checkpoints** (better guidance)
- **Smaller compute support** (more accessible)

**Expected Improvement**: 2-3x faster convergence to optimal configuration

---

## 📊 Comparison with Karpathy's autoresearch

| Feature | Karpathy's Version | Our Current | Our Improved |
|---------|-------------------|-------------|--------------|
| **File Structure** | 3 files | 33 files | 5 files |
| **Time Budget** | 5 min | None | 5 min ✅ |
| **Metric** | val_bpb | val_loss | val_bpb ✅ |
| **Proposals** | Dynamic | Fixed list | Adaptive ✅ |
| **Visualization** | progress.png | None | progress.png ✅ |
| **Analysis** | analysis.ipynb | None | analysis.ipynb ✅ |
| **Human Review** | Yes | No | Yes ✅ |
| **Arabic Support** | No | Yes | Optimized ✅ |
| **Small Compute** | Yes (forks) | No | Yes ✅ |

---

## ✅ Next Steps

1. **Immediate**: Create simplified file structure
2. **This Week**: Implement time budget and val_bpb
3. **Next Week**: Add adaptive proposals and visualization
4. **Week 3**: Human-in-the-loop and Arabic optimization

**Target Version**: 2.1.0  
**Status**: Ready for Implementation  
**Priority**: High

---

**Version**: 2.1.0 (planned)  
**Date**: March 26, 2026  
**Status**: 📋 **IMPLEMENTATION PLAN**  
**Next Action**: Begin Phase 1 implementation
