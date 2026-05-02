# Chapter 7: NLP Fundamentals - From Text to Numbers

> **Learning Duration:** 4 Days  
> **Difficulty:** Intermediate  
> **Prerequisites:** Python, basic linear algebra

---

## 🎯 Learning Objectives

By the end of this chapter, you will:
- Clean and normalize text data
- Implement various tokenization strategies
- Understand word embeddings and their training
- Build simple NLP pipelines
- Apply NLP techniques to real problems

---

## 7.1 Why NLP Is Different

### Text vs. Numbers

Images → pixels are already numbers
Audio → waveforms are already numbers
Text → **characters/symbols** need to be converted to numbers

```
Image:      [0.2, 0.5, 0.1, ...]    Already numeric!
Audio:      [0.1, -0.3, 0.8, ...]    Already numeric!
Text:       "hello world"           Needs conversion
```

### The Challenge

**Challenge 1: Discrete Symbols**
- No natural numeric representation
- Need encoding schemes

**Challenge 2: Variable Length**
- Sentences have different lengths
- Need fixed-length representations

**Challenge 3: Ambiguity**
- Same word can mean different things
- Context matters

---

## 7.2 Text Preprocessing

### The Pipeline

```
Raw Text → Cleaning → Normalization → Tokenization → Vectorization
                              ↓
                      Stop words, stemming
```

### Implementation

```python
import re
import string
from typing import List

class TextPreprocessor:
    """
    Complete text preprocessing pipeline.
    """
    
    def __init__(self, lowercase: bool = True,
                 remove_punctuation: bool = False,
                 remove_numbers: bool = False,
                 remove_urls: bool = True,
                 remove_emails: bool = True):
        """
        Initialize preprocessor.
        
        Args:
            lowercase: Convert to lowercase
            remove_punctuation: Remove punctuation
            remove_numbers: Remove numeric characters
            remove_urls: Remove URLs
            remove_emails: Remove email addresses
        """
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_numbers = remove_numbers
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        
    def clean(self, text: str) -> str:
        """
        Basic cleaning operations.
        """
        if self.remove_urls:
            text = re.sub(r'http\S+|www\.\S+', '', text)
            
        if self.remove_emails:
            text = re.sub(r'\S+@\S+', '', text)
            
        if self.remove_numbers:
            text = re.sub(r'\d+', '', text)
            
        if self.remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
            
        if self.lowercase:
            text = text.lower()
            
        return text
    
    def remove_whitespace(self, text: str) -> str:
        """Remove extra whitespace."""
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def expand_contractions(self, text: str) -> str:
        """
        Expand English contractions.
        
        Example: "don't" → "do not"
        """
        contractions = {
            "n't": " not",
            "'re": " are",
            "'s": " is",
            "'d": " would",
            "'ll": " will",
            "'ve": " have",
            "'m": " am"
        }
        
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
            
        return text
    
    def remove_special_characters(self, text: str) -> str:
        """Keep only alphanumeric and spaces."""
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        return text
    
    def preprocess(self, text: str) -> str:
        """
        Complete preprocessing pipeline.
        """
        # Step 1: Clean
        text = self.clean(text)
        
        # Step 2: Expand contractions
        text = self.expand_contractions(text)
        
        # Step 3: Remove special characters
        text = self.remove_special_characters(text)
        
        # Step 4: Remove extra whitespace
        text = self.remove_whitespace(text)
        
        return text
    
    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """Preprocess a list of texts."""
        return [self.preprocess(text) for text in texts]
```

### Usage Example

```python
preprocessor = TextPreprocessor(
    lowercase=True,
    remove_punctuation=True,
    remove_urls=True,
    remove_emails=True
)

texts = [
    "Check out https://example.com for more info!",
    "Don't forget to email me at test@test.com",
    "Hello World!!! How are you?"
]

for text in texts:
    cleaned = preprocessor.preprocess(text)
    print(f"Original: {text}")
    print(f"Cleaned:  {cleaned}")
    print()
```

---

## 7.3 Tokenization

### What Is Tokenization?

**Tokenization** splits text into smaller units (tokens):
- Words
- Subwords (pieces of words)
- Characters
- Sentences

### Word Tokenization

```python
class WordTokenizer:
    """
    Simple word-based tokenizer.
    """
    
    def __init__(self, lowercase: bool = True):
        self.lowercase = lowercase
        self.vocab = set()
        self.token_to_id = {}
        self.id_to_token = {}
        
    def tokenize(self, text: str) -> List[str]:
        """
        Split text into word tokens.
        
        Args:
            text: Input text string
            
        Returns:
            List of word tokens
        """
        if self.lowercase:
            text = text.lower()
            
        # Split on whitespace and punctuation
        tokens = re.findall(r'\b\w+\b|[^\w\s]', text)
        
        return tokens
    
    def build_vocab(self, texts: List[str], min_freq: int = 1):
        """
        Build vocabulary from texts.
        
        Args:
            texts: List of text strings
            min_freq: Minimum frequency for inclusion
        """
        word_freq = {}
        
        for text in texts:
            tokens = self.tokenize(text)
            for token in tokens:
                word_freq[token] = word_freq.get(token, 0) + 1
        
        # Build vocab with minimum frequency
        self.vocab = {word for word, freq in word_freq.items() 
                      if freq >= min_freq}
        
        # Create mappings
        self.token_to_id = {token: idx + 1 for idx, token 
                           in enumerate(sorted(self.vocab))}
        self.token_to_id['<PAD>'] = 0
        
        self.id_to_token = {idx: token for token, idx 
                          in self.token_to_id.items()}
    
    def encode(self, text: str, max_length: int = None) -> np.ndarray:
        """Convert text to token IDs."""
        tokens = self.tokenize(text)
        
        # Convert to IDs
        ids = [self.token_to_id.get(token, 0) for token in tokens]
        
        # Pad or truncate
        if max_length:
            if len(ids) > max_length:
                ids = ids[:max_length]
            else:
                ids = ids + [0] * (max_length - len(ids))
                
        return np.array(ids)
    
    def decode(self, ids: np.ndarray) -> str:
        """Convert IDs back to text."""
        tokens = [self.id_to_token.get(idx, '<UNK>') for idx in ids]
        return ' '.join(tokens)
```

### Subword Tokenization (BPE)

```python
class BPETokenizer:
    """
    Byte Pair Encoding (BPE) subword tokenizer.
    
    Based on "Neural Machine Translation of Rare Words with Subword Units"
    (Sennrich et al., 2016)
    """
    
    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size
        self.merges = []
        self.vocab = {}
        
    def get_frequencies(self, tokens: List[str]) -> dict:
        """Count frequency of consecutive pairs."""
        pairs = {}
        
        for token in tokens:
            # Add end-of-word marker
            chars = list(token) + ['</w>']
            
            # Count pairs
            for i in range(len(chars) - 1):
                pair = (chars[i], chars[i+1])
                pairs[pair] = pairs.get(pair, 0) + 1
                
        return pairs
    
    def merge_pair(self, tokens: List[str], pair: tuple) -> List[str]:
        """Merge all occurrences of a pair."""
        result = []
        
        for token in tokens:
            chars = list(token) + ['</w>']
            merged = []
            i = 0
            
            while i < len(chars):
                if i < len(chars) - 1 and (chars[i], chars[i+1]) == pair:
                    merged.append(pair[0] + pair[1])
                    i += 2
                else:
                    merged.append(chars[i])
                    i += 1
                    
            result.append(''.join(merged))
            
        return result
    
    def train(self, texts: List[str], min_freq: int = 2):
        """
        Train BPE tokenizer.
        
        Args:
            texts: List of training texts
            min_freq: Minimum pair frequency for merge
        """
        # Initial tokenization (character-level)
        tokenized = [self._char_tokenize(text) for text in texts]
        
        # Build initial vocab
        vocab_set = set()
        for tokens in tokenized:
            for token in tokens:
                for char in list(token) + ['</w>']:
                    vocab_set.add(char)
                    
        self.vocab = {char: idx for idx, char in enumerate(sorted(vocab_set))}
        
        # Iteratively merge
        for _ in range(self.vocab_size - len(vocab_set)):
            # Count pairs
            pair_freqs = {}
            for tokens in tokenized:
                pairs = self.get_frequencies(tokens)
                for pair, freq in pairs.items():
                    pair_freqs[pair] = pair_freqs.get(pair, 0) + freq
            
            # Find most frequent pair above threshold
            best_pair = None
            best_freq = min_freq - 1
            
            for pair, freq in pair_freqs.items():
                if freq > best_freq:
                    best_freq = freq
                    best_pair = pair
                    
            if best_pair is None:
                break
                
            # Merge
            self.merges.append(best_pair)
            tokenized = [self.merge_pair(tokens, best_pair) for tokens in tokenized]
            
            # Add to vocab
            merged_token = best_pair[0] + best_pair[1]
            self.vocab[merged_token] = len(self.vocab)
            
    def _char_tokenize(self, text: str) -> List[str]:
        """Character-level tokenization."""
        return list(text)
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize using learned merges."""
        tokens = self._char_tokenize(text)
        
        # Apply merges in order
        for pair in self.merges:
            tokens = self.merge_pair([tokens], pair)
            tokens = tokens[0].split('</w>')
            tokens = [t for t in tokens if t]
            tokens = [t + '</w>' for t in tokens]
            
        return [t.replace('</w>', '') for t in tokens]
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        tokens = self.tokenize(text)
        
        return [self.vocab.get(token, 0) for token in tokens]
```

### Modern Tokenizers (Hugging Face style)

In practice, use established libraries:

```python
# Using Hugging Face tokenizers
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# Create BPE tokenizer
tokenizer = Tokenizer(BPE())
tokenizer.pre_tokenizer = Whitespace()

# Train
trainer = BpeTrainer(vocab_size=10000, min_frequency=2)
tokenizer.train(["file1.txt", "file2.txt"], trainer)

# Use
output = tokenizer.encode("Hello world!")
print(output.tokens)
print(output.ids)
```

---

## 7.4 Text Vectorization

### Bag of Words (BoW)

```python
class BagOfWords:
    """
    Bag of Words vectorizer.
    
    Each document becomes a vector of word counts.
    """
    
    def __init__(self, max_features: int = None, min_df: int = 1):
        self.max_features = max_features
        self.min_df = min_df
        self.vocab = {}
        self.idf = None
        
    def fit(self, texts: List[str]):
        """Build vocabulary from texts."""
        word_doc_freq = {}
        n_docs = len(texts)
        
        # Count document frequencies
        for text in texts:
            words = set(text.lower().split())
            for word in words:
                word_doc_freq[word] = word_doc_freq.get(word, 0) + 1
        
        # Filter by minimum document frequency
        vocab = {word: freq for word, freq in word_doc_freq.items() 
                 if freq >= self.min_df}
        
        # Limit to max_features
        if self.max_features:
            vocab = dict(sorted(vocab.items(), key=lambda x: x[1], 
                               reverse=True)[:self.max_features])
        
        # Create mapping
        self.vocab = {word: idx for idx, word in enumerate(vocab.keys())}
        
        # Compute IDF
        self.idf = {}
        for word, idx in self.vocab.items():
            df = word_doc_freq[word]
            self.idf[idx] = np.log(n_docs / (1 + df)) + 1
            
    def transform(self, texts: List[str]) -> np.ndarray:
        """Convert texts to BoW vectors."""
        n_samples = len(texts)
        n_features = len(self.vocab)
        
        # Create sparse matrix
        vectors = np.zeros((n_samples, n_features))
        
        for i, text in enumerate(texts):
            words = text.lower().split()
            for word in words:
                if word in self.vocab:
                    vectors[i, self.vocab[word]] += 1
                    
        return vectors
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(texts)
        return self.transform(texts)
```

### TF-IDF Vectorization

```python
class TFIDFVectorizer:
    """
    Term Frequency - Inverse Document Frequency vectorizer.
    
    TF-IDF weights words by:
    - How often they appear in document (TF)
    - How rare they are across documents (IDF)
    
    Formula: TF-IDF = TF × IDF
    """
    
    def __init__(self, max_features: int = None, min_df: int = 1):
        self.max_features = max_features
        self.min_df = min_df
        self.vocab = {}
        self.idf = None
        
    def fit(self, texts: List[str]):
        """Compute IDF values."""
        n_docs = len(texts)
        word_doc_freq = {}
        
        for text in texts:
            words = set(text.lower().split())
            for word in words:
                word_doc_freq[word] = word_doc_freq.get(word, 0) + 1
        
        # Filter
        vocab = {word: freq for word, freq in word_doc_freq.items() 
                 if freq >= self.min_df}
        
        if self.max_features:
            vocab = dict(sorted(vocab.items(), key=lambda x: x[1], 
                               reverse=True)[:self.max_features])
        
        self.vocab = {word: idx for idx, word in enumerate(vocab.keys())}
        
        # IDF: log(N / df) + 1
        self.idf = {}
        for word, idx in self.vocab.items():
            df = word_doc_freq[word]
            self.idf[idx] = np.log(n_docs / (1 + df)) + 1
            
    def transform(self, texts: List[str]) -> np.ndarray:
        """Convert texts to TF-IDF vectors."""
        n_samples = len(texts)
        n_features = len(self.vocab)
        
        vectors = np.zeros((n_samples, n_features))
        
        for i, text in enumerate(texts):
            words = text.lower().split()
            word_counts = {}
            
            # Term frequency
            for word in words:
                if word in self.vocab:
                    word_counts[word] = word_counts.get(word, 0) + 1
            
            # TF-IDF
            for word, count in word_counts.items():
                idx = self.vocab[word]
                tf = count / len(words) if len(words) > 0 else 0
                vectors[i, idx] = tf * self.idf[idx]
                
        return vectors
```

---

## 7.5 Complete NLP Pipeline Example

```python
class NLPipeline:
    """
    Complete NLP pipeline from text to features.
    """
    
    def __init__(self, 
                 lowercase: bool = True,
                 remove_punctuation: bool = True,
                 min_df: int = 2,
                 max_features: int = 5000,
                 vectorizer_type: str = 'tfidf'):
        """
        Initialize NLP pipeline.
        
        Args:
            lowercase: Convert to lowercase
            remove_punctuation: Remove punctuation
            min_df: Minimum document frequency
            max_features: Maximum vocabulary size
            vectorizer_type: 'bow' or 'tfidf'
        """
        self.preprocessor = TextPreprocessor(
            lowercase=lowercase,
            remove_punctuation=remove_punctuation
        )
        
        self.min_df = min_df
        self.max_features = max_features
        
        if vectorizer_type == 'tfidf':
            self.vectorizer = TFIDFVectorizer(
                max_features=max_features,
                min_df=min_df
            )
        else:
            self.vectorizer = BagOfWords(
                max_features=max_features,
                min_df=min_df
            )
            
    def fit(self, texts: List[str]):
        """Fit pipeline on texts."""
        # Preprocess
        processed = self.preprocessor.preprocess_batch(texts)
        
        # Fit vectorizer
        self.vectorizer.fit(processed)
        
    def transform(self, texts: List[str]) -> np.ndarray:
        """Transform texts to feature vectors."""
        processed = self.preprocessor.preprocess_batch(texts)
        return self.vectorizer.transform(processed)
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Fit and transform."""
        self.fit(texts)
        return self.transform(texts)


# Example Usage
if __name__ == "__main__":
    # Sample documents
    documents = [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with many layers.",
        "Natural language processing deals with text data.",
        "Computer vision processes and analyzes images.",
        "Reinforcement learning trains agents through rewards.",
        "Machine learning models can be supervised or unsupervised."
    ]
    
    # Create and fit pipeline
    pipeline = NLPipeline(
        lowercase=True,
        remove_punctuation=True,
        min_df=1,
        max_features=20,
        vectorizer_type='tfidf'
    )
    
    # Transform
    features = pipeline.fit_transform(documents)
    
    print(f"Feature matrix shape: {features.shape}")
    print(f"Vocabulary: {list(pipeline.vectorizer.vocab.keys())}")
    
    # Show first document's features
    print(f"\nDocument 0: {documents[0]}")
    print(f"Features: {features[0]}")
```

---

## 📝 Summary

### Key Takeaways

1. **Text preprocessing**: Clean, normalize, remove noise
2. **Tokenization**: Split text into meaningful units
3. **BoW**: Simple bag of words (count-based)
4. **TF-IDF**: Weighted by term importance across documents
5. **BPE**: Subword tokenization for handling rare words
6. **Pipeline**: Combine all components for end-to-end processing

### Modern Practice

- Use **transformer tokenizers** (BERT, GPT) for state-of-the-art
- Pre-trained tokenizers handle edge cases
- Vocabulary sizes: 30K-50K tokens typical

### Next Steps

- Word embeddings (Word2Vec, GloVe)
- RNNs for sequence modeling
- Transformers for modern NLP

---

## ❓ Quick Check

1. Why do we need to normalize text?
2. What's the difference between word and subword tokenization?
3. What does TF-IDF measure?
4. Why is BPE useful for NLP?

*Answers at end of chapter*