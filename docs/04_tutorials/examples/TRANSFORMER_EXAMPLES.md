# Transformer Module Usage Examples

Complete guide to using BERT and GPT-2 from scratch (`src/llm/transformer.py`).

---

## Table of Contents
1. [BERT - Bidirectional Encoder](#bert-usage)
2. [GPT-2 - Autoregressive Decoder](#gpt-2-usage)
3. [Text Classification with BERT](#text-classification)
4. [Text Generation with GPT-2](#text-generation)
5. [Question Answering](#question-answering)
6. [Fine-Tuning Examples](#fine-tuning)

---

## BERT Usage

### Basic BERT Forward Pass

```python
import numpy as np
from src.llm.transformer import BERT

# Create BERT model (BERT-base configuration)
bert = BERT(
    vocab_size=30522,      # WordPiece vocabulary
    max_seq_len=512,       # Maximum sequence length
    d_model=768,           # Hidden size
    num_layers=12,         # Number of transformer layers
    num_heads=12,          # Number of attention heads
    d_ff=3072,            # Feed-forward dimension
    dropout=0.1
)

# Input: tokenized text
# Shape: (batch_size, seq_len)
input_ids = np.random.randint(0, 30522, (4, 128))  # Batch of 4, seq_len 128

# Optional: segment IDs for sentence pair tasks
segment_ids = np.zeros((4, 128), dtype=np.int32)
segment_ids[:, 64:] = 1  # Second sentence starts at position 64

# Forward pass
hidden_states = bert.forward(
    input_ids=input_ids,
    segment_ids=segment_ids,
    training=False
)

print(f"Hidden states shape: {hidden_states.shape}")  # (4, 128, 768)

# Extract [CLS] token representation (for classification)
cls_output = hidden_states[:, 0, :]  # (4, 768)
print(f"[CLS] output shape: {cls_output.shape}")
```

### Masked Language Modeling (MLM)

```python
# Create BERT model
bert = BERT(vocab_size=30522, max_seq_len=512)

# Input with [MASK] tokens
input_ids = np.random.randint(0, 30522, (2, 64))
# Suppose position 10 and 25 are masked (token_id = 103)
input_ids[:, [10, 25]] = 103  # [MASK] token

# Get hidden states
hidden_states = bert.forward(input_ids, training=False)

# Predict masked tokens
logits = bert.predict_masked_tokens(hidden_states, training=False)
print(f"MLM logits shape: {logits.shape}")  # (2, 64, 30522)

# Get predictions for masked positions
masked_predictions = np.argmax(logits[:, [10, 25], :], axis=-1)
print(f"Predicted tokens at masked positions: {masked_predictions}")
```

---

## GPT-2 Usage

### Basic Text Generation

```python
from src.llm.transformer import GPT2

# Create GPT-2 model (GPT-2 small configuration)
gpt2 = GPT2(
    vocab_size=50257,      # GPT-2 vocabulary (BPE)
    max_seq_len=1024,      # Context length
    d_model=768,           # Hidden size
    num_layers=12,         # Transformer layers
    num_heads=12,          # Attention heads
    d_ff=3072,            # Feed-forward dimension
    dropout=0.1
)

# Input prompt (tokenized)
prompt = np.array([[15496, 995, 318, 257]])  # "Hello world is a" (example tokens)

# Generate next token
logits = gpt2.forward(prompt, training=False)
next_token_logits = logits[:, -1, :]  # Get logits for next token

# Sample next token
probabilities = np.exp(next_token_logits) / np.sum(np.exp(next_token_logits))
next_token = np.random.choice(50257, p=probabilities[0])

print(f"Generated next token: {next_token}")
```

### Autoregressive Generation

```python
def generate_text(gpt2_model, prompt_ids, max_new_tokens=50, temperature=1.0):
    """
    Generate text autoregressively.
    
    Args:
        gpt2_model: GPT2 instance
        prompt_ids: (1, seq_len) initial tokens
        max_new_tokens: Number of tokens to generate
        temperature: Sampling temperature (higher = more random)
    
    Returns:
        Generated sequence
    """
    generated = prompt_ids.copy()
    
    for _ in range(max_new_tokens):
        # Get logits for next token
        logits = gpt2_model.forward(generated, training=False)
        next_token_logits = logits[:, -1, :] / temperature
        
        # Softmax
        exp_logits = np.exp(next_token_logits - np.max(next_token_logits))
        probabilities = exp_logits / np.sum(exp_logits)
        
        # Sample
        next_token = np.random.choice(50257, p=probabilities[0])
        
        # Append to sequence
        generated = np.concatenate([generated, [[next_token]]], axis=1)
        
        # Stop if max length reached
        if generated.shape[1] >= 1024:
            break
    
    return generated

# Usage
prompt = np.array([[15496, 995]])  # "Hello world"
generated_sequence = generate_text(gpt2, prompt, max_new_tokens=30)
print(f"Generated {generated_sequence.shape[1]} tokens")
```

### Top-K and Top-P Sampling

```python
def top_k_sampling(logits, k=50):
    """Sample from top-k most likely tokens."""
    top_k_indices = np.argsort(logits)[-k:]
    top_k_logits = logits[top_k_indices]
    
    # Softmax over top-k
    exp_logits = np.exp(top_k_logits - np.max(top_k_logits))
    probabilities = exp_logits / np.sum(exp_logits)
    
    # Sample
    sampled_idx = np.random.choice(len(probabilities), p=probabilities)
    return top_k_indices[sampled_idx]

def top_p_sampling(logits, p=0.9):
    """Nucleus sampling (top-p)."""
    sorted_indices = np.argsort(logits)[::-1]
    sorted_logits = logits[sorted_indices]
    
    # Softmax
    exp_logits = np.exp(sorted_logits - np.max(sorted_logits))
    probabilities = exp_logits / np.sum(exp_logits)
    
    # Cumulative probability
    cumsum = np.cumsum(probabilities)
    
    # Find cutoff point
    cutoff_idx = np.searchsorted(cumsum, p)
    
    # Sample from nucleus
    nucleus_probs = probabilities[:cutoff_idx + 1]
    nucleus_probs = nucleus_probs / np.sum(nucleus_probs)
    sampled_idx = np.random.choice(len(nucleus_probs), p=nucleus_probs)
    
    return sorted_indices[sampled_idx]

# Usage in generation
logits = gpt2.forward(prompt, training=False)[:, -1, :]
next_token = top_k_sampling(logits[0], k=50)
# or
next_token = top_p_sampling(logits[0], p=0.9)
```

---

## Text Classification

### Sentiment Analysis with BERT

```python
class BERTClassifier:
    """BERT-based text classifier."""
    
    def __init__(self, num_classes=2):
        # BERT encoder
        self.bert = BERT(
            vocab_size=30522,
            max_seq_len=512,
            d_model=768,
            num_layers=12,
            num_heads=12
        )
        
        # Classification head
        from src.ml.deep_learning import Dense, Activation, Dropout
        self.dropout = Dropout(0.1)
        self.classifier = Dense(768, num_classes)
    
    def forward(self, input_ids, segment_ids=None, training=True):
        # Get BERT outputs
        hidden_states = self.bert.forward(
            input_ids,
            segment_ids=segment_ids,
            training=training
        )
        
        # Use [CLS] token
        cls_output = hidden_states[:, 0, :]  # (batch_size, 768)
        
        # Classification
        pooled = self.dropout.forward(cls_output, training)
        logits = self.classifier.forward(pooled, training)
        
        return logits
    
    def predict(self, input_ids, segment_ids=None):
        """Predict class probabilities."""
        logits = self.forward(input_ids, segment_ids, training=False)
        
        # Softmax
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probabilities = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        
        return probabilities

# Usage
classifier = BERTClassifier(num_classes=2)  # Binary sentiment

# Example: "This movie is great!"
input_ids = np.array([[2023, 3185, 2003, 2307, 999]])  # Tokenized
probabilities = classifier.predict(input_ids)

print(f"Positive probability: {probabilities[0, 1]:.2%}")
print(f"Negative probability: {probabilities[0, 0]:.2%}")
```

---

## Text Generation

### Story Generation with GPT-2

```python
class StoryGenerator:
    """GPT-2 based story generator."""
    
    def __init__(self):
        self.gpt2 = GPT2(
            vocab_size=50257,
            max_seq_len=1024,
            d_model=768,
            num_layers=12,
            num_heads=12
        )
    
    def generate_story(self, prompt, max_length=200, temperature=0.8, top_p=0.9):
        """
        Generate a story from a prompt.
        
        Args:
            prompt: Starting text (tokenized)
            max_length: Maximum story length in tokens
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
        
        Returns:
            Generated token sequence
        """
        current_sequence = prompt.copy()
        
        for _ in range(max_length):
            # Get next token logits
            logits = self.gpt2.forward(current_sequence, training=False)
            next_logits = logits[:, -1, :] / temperature
            
            # Top-p sampling
            next_token = top_p_sampling(next_logits[0], p=top_p)
            
            # Append
            current_sequence = np.concatenate(
                [current_sequence, [[next_token]]], axis=1
            )
            
            # Stop at end token (e.g., token_id = 50256)
            if next_token == 50256:
                break
        
        return current_sequence

# Usage
generator = StoryGenerator()

# Prompt: "Once upon a time"
prompt = np.array([[7454, 2402, 257, 640]])
story = generator.generate_story(prompt, max_length=100)

print(f"Generated story with {story.shape[1]} tokens")
```

---

## Question Answering

### BERT-based QA System

```python
class BERTQuestionAnswering:
    """BERT for extractive question answering."""
    
    def __init__(self):
        self.bert = BERT(vocab_size=30522, max_seq_len=512)
        
        # Span prediction heads
        from src.ml.deep_learning import Dense
        self.start_logits = Dense(768, 1)
        self.end_logits = Dense(768, 1)
    
    def forward(self, input_ids, segment_ids, training=True):
        """
        Predict answer span in context.
        
        Args:
            input_ids: [CLS] question [SEP] context [SEP]
            segment_ids: 0 for question, 1 for context
        
        Returns:
            start_logits, end_logits: (batch_size, seq_len)
        """
        # Get BERT outputs
        hidden_states = self.bert.forward(
            input_ids,
            segment_ids=segment_ids,
            training=training
        )
        
        # Predict start and end positions
        start_logits = self.start_logits.forward(hidden_states, training)
        end_logits = self.end_logits.forward(hidden_states, training)
        
        return start_logits.squeeze(-1), end_logits.squeeze(-1)
    
    def extract_answer(self,input_ids, segment_ids, tokens):
        """Extract answer span from context."""
        start_logits, end_logits = self.forward(
            input_ids, segment_ids, training=False
        )
        
        # Find best span
        start_idx = np.argmax(start_logits[0])
        end_idx = np.argmax(end_logits[0])
        
        # Ensure valid span
        if end_idx < start_idx:
            end_idx = start_idx
        
        # Extract answer tokens
        answer_tokens = tokens[start_idx:end_idx + 1]
        
        return {
            'answer': ''.join(answer_tokens),
            'start_idx': int(start_idx),
            'end_idx': int(end_idx),
            'confidence': float(start_logits[0, start_idx] + end_logits[0, end_idx])
        }

# Usage
qa_model = BERTQuestionAnswering()

# Example: Question + Context
# input_ids = [CLS] + question_tokens + [SEP] + context_tokens + [SEP]
input_ids = np.random.randint(0, 30522, (1, 256))
segment_ids = np.zeros((1, 256))
segment_ids[:, 64:] = 1  # Context starts at position 64

tokens = ["what", "is", "AI", "?", "[SEP]", "AI", "is", "intelligence", "..."]
answer = qa_model.extract_answer(input_ids, segment_ids, tokens)

print(f"Answer: {answer['answer']}")
print(f"Span: [{answer['start_idx']}, {answer['end_idx']}]")
```

---

## Fine-Tuning

### Fine-Tuning BERT on Custom Dataset

```python
def fine_tune_bert(model, train_data, val_data, epochs=3, batch_size=16):
    """
    Fine-tune BERT on custom task.
    
    Args:
        model: BERTClassifier instance
        train_data: List of (input_ids, label) tuples
        val_data: Validation data
        epochs: Number of epochs
        batch_size: Batch size
    
    Returns:
        Training history
    """
    from src.ml.deep_learning import CrossEntropyLoss
    
    loss_fn = CrossEntropyLoss()
    history = {'train_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        epoch_loss = 0
        n_batches = len(train_data) // batch_size
        
        for i in range(n_batches):
            # Get batch
            batch_start = i * batch_size
            batch_end = batch_start + batch_size
            batch = train_data[batch_start:batch_end]
            
            # Prepare inputs
            input_ids = np.stack([item[0] for item in batch])
            labels = np.array([item[1] for item in batch])
            
            # Forward pass
            logits = model.forward(input_ids, training=True)
            loss = loss_fn.forward(logits, labels)
            
            # Backward pass (simplified)
            # In full implementation: compute gradients and update
            
            epoch_loss += loss
        
        # Validation
        val_input_ids = np.stack([item[0] for item in val_data])
        val_labels = np.array([item[1] for item in val_data])
        val_probs = model.predict(val_input_ids)
        val_preds = np.argmax(val_probs, axis=1)
        val_acc = np.mean(val_preds == val_labels)
        
        history['train_loss'].append(epoch_loss / n_batches)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}: Loss={epoch_loss/n_batches:.4f}, Val Acc={val_acc:.2%}")
    
    return history

# Example usage
# train_data = [(input_ids, label), ...]
# history = fine_tune_bert(classifier, train_data, val_data, epochs=3)
```

---

## Advanced Topics

### Attention Visualization

```python
def visualize_attention(bert_model, input_ids, layer_idx=11, head_idx=0):
    """
    Extract and visualize attention weights.
    
    Note: Requires modifying forward pass to return attention weights.
    """
    hidden_states = bert_model.forward(input_ids, training=False)
    
    # Would extract attention weights from specific layer/head
    # attention_weights shape: (batch_size, num_heads, seq_len, seq_len)
    
    # Visualization with matplotlib
    import matplotlib.pyplot as plt
    
    # plt.imshow(attention_weights[0, head_idx], cmap='viridis')
    # plt.colorbar()
    # plt.title(f'Layer {layer_idx}, Head {head_idx}')
    # plt.show()
    
    pass
```

### Model Saving and Loading

```python
# Save BERT model
bert_model = BERT(vocab_size=30522, max_seq_len=512)
bert_model.save('bert_model.npz')

# Load BERT model
loaded_model = BERT(vocab_size=30522, max_seq_len=512)
loaded_model.load('bert_model.npz')
```

---

## Performance Tips

1. **Batch Size**: Use 8-32 for training, larger for inference
2. **Sequence Length**: Limit to 128-256 for most tasks (512 max for BERT)
3. **Mixed Precision**: Convert to PyTorch/TensorFlow for FP16
4. **Gradient Accumulation**: Simulate larger batches with limited memory
5. **Learning Rate**: 2e-5 to 5e-5 typical for fine-tuning

---

## See Also

- [BERT Paper](https://arxiv.org/abs/1810.04805)
- [GPT-2 Paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [Attention is All You Need](https://arxiv.org/abs/1706.03762)
- [HuggingFace Transformers](https://huggingface.co/transformers/) (production library)
