"""
Arabic NLP Utilities
====================

Complete toolkit for Arabic language processing including:
- Text normalization
- Dialect detection
- Tokenization helpers
- Dataset preparation
"""

import re
from typing import List, Dict, Tuple


class ArabicTextNormalizer:
    """
    Comprehensive Arabic text normalization
    
    Handles:
    - Alif normalization (أ, إ, آ → ا)
    - Alif Maqsura (ى → ي)
    - Ta Marbuta (ة → ه)
    - Hamza variations (ؤ, ئ → ء)
    - Diacritics removal (tashkeel)
    - Unicode normalization
    - Whitespace normalization
    """
    
    def __init__(self, normalize_diacritics: bool = True,
                 normalize_hamza: bool = True,
                 normalize_alif: bool = True):
        self.normalize_diacritics = normalize_diacritics
        self.normalize_hamza = normalize_hamza
        self.normalize_alif = normalize_alif
    
    def normalize(self, text: str) -> str:
        """
        Normalize Arabic text
        
        Args:
            text: Input Arabic text
            
        Returns:
            Normalized Arabic text
        """
        if self.normalize_alif:
            # Normalize Alif forms (أ, إ, آ → ا)
            text = re.sub("[إأآا]", "ا", text)
        
        # Normalize Alif Maqsura (ى → ي)
        text = re.sub("ى", "ي", text)
        
        # Normalize Ta Marbuta (ة → ه)
        text = re.sub("ة", "ه", text)
        
        if self.normalize_hamza:
            # Normalize Hamza variations
            text = re.sub("ؤ", "ء", text)
            text = re.sub("ئ", "ء", text)
        
        if self.normalize_diacritics:
            # Remove diacritics (tashkeel)
            # Unicode range: U+0617-U+061A, U+064B-U+0652
            text = re.sub(r'[\u0617-\u061A\u064B-\u0652]', '', text)
        
        # Normalize Unicode
        import unicodedata
        text = unicodedata.normalize("NFKC", text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def normalize_batch(self, texts: List[str]) -> List[str]:
        """Normalize batch of texts"""
        return [self.normalize(text) for text in texts]


class ArabicDialectDetector:
    """
    Simple Arabic dialect detection
    
    Detects:
    - Modern Standard Arabic (MSA)
    - Egyptian
    - Levantine (Syrian, Lebanese, Jordanian, Palestinian)
    - Gulf (Saudi, Emirati, Qatari, Kuwaiti)
    - Maghrebi (Moroccan, Algerian, Tunisian)
    - Iraqi
    - Yemeni
    
    Note: For production use, consider CAMeL Tools or MARBERT
    """
    
    def __init__(self):
        self.dialect_indicators = {
            'egyptian': [
                'إزك', 'إيه', 'أوي', 'خلاص', 'دلوقتي', 'إيه',
                'مش', 'عايز', 'قولك', 'يعني', 'حاجة'
            ],
            'levantine': [
                'شو', 'كيفك', 'هلق', 'أبدا', 'هيك', 'هون',
                'إلن', 'عنا', 'فيي', 'منيح', 'كتير'
            ],
            'gulf': [
                'شلونك', 'يا', 'الحين', 'زين', 'أبي', 'يالله',
                'بخير', 'ويس', 'ليش', 'تو', 'داب'
            ],
            'maghrebi': [
                'علاش', 'دابا', 'باش', 'ماشي', 'برشا', 'عندي',
                'واش', 'هاكا', 'غادي', 'كي', 'شكون'
            ],
            'iraqi': [
                'شكون', 'بيّه', 'عيني', 'هسة', 'كلش', 'ياخي',
                'أكو', 'ماكو', 'شلون', 'وين'
            ],
            'yemeni': [
                'منتو', 'أبى', 'الحين', 'زين', 'شفيك', 'ماشي',
                'قد', 'إلا', 'ويش', 'توني'
            ]
        }
    
    def detect(self, text: str) -> str:
        """
        Detect dialect of Arabic text
        
        Args:
            text: Arabic text
            
        Returns:
            Dialect label ('msa', 'egyptian', 'levantine', etc.)
        """
        text_lower = text.lower()
        
        scores = {}
        for dialect, indicators in self.dialect_indicators.items():
            score = sum(1 for indicator in indicators if indicator in text_lower)
            scores[dialect] = score
        
        # Get highest scoring dialect
        if max(scores.values()) == 0:
            return 'msa'  # Modern Standard Arabic

        best_dialect = max(scores, key=scores.get)
        return best_dialect
    
    def detect_batch(self, texts: List[str]) -> List[str]:
        """Detect dialects for batch of texts"""
        return [self.detect(text) for text in texts]


class ArabicTokenizer:
    """
    Arabic-aware tokenizer
    
    Features:
    - Optional normalization before tokenization
    - Support for different tokenization strategies
    - Compatible with Hugging Face tokenizers
    """
    
    def __init__(self, tokenizer, normalizer: ArabicTextNormalizer = None,
                 do_normalize: bool = True):
        """
        Initialize Arabic tokenizer
        
        Args:
            tokenizer: Hugging Face tokenizer
            normalizer: ArabicTextNormalizer instance
            do_normalize: Whether to normalize before tokenization
        """
        self.tokenizer = tokenizer
        self.normalizer = normalizer or ArabicTextNormalizer()
        self.do_normalize = do_normalize
    
    def encode(self, text: str, **kwargs):
        """Encode text to tokens"""
        if self.do_normalize:
            text = self.normalizer.normalize(text)
        
        return self.tokenizer.encode(text, **kwargs)
    
    def decode(self, token_ids: List[int], **kwargs):
        """Decode tokens to text"""
        return self.tokenizer.decode(token_ids, **kwargs)
    
    def __call__(self, texts: List[str], **kwargs):
        """Tokenize batch of texts"""
        if self.do_normalize:
            texts = self.normalizer.normalize_batch(texts)
        
        return self.tokenizer(texts, **kwargs)


def prepare_arabic_dataset(dataset, tokenizer, max_length: int = 512,
                          text_column: str = 'text') -> Dict:
    """
    Prepare Arabic dataset for fine-tuning
    
    Args:
        dataset: Hugging Face dataset
        tokenizer: Tokenizer (or ArabicTokenizer)
        max_length: Maximum sequence length
        text_column: Name of text column in dataset
        
    Returns:
        Tokenized dataset
    """
    from datasets import DatasetDict
    
    def tokenize(examples):
        texts = examples[text_column]
        
        # Handle ArabicTokenizer or regular tokenizer
        if isinstance(tokenizer, ArabicTokenizer):
            tokenized = tokenizer(texts, truncation=True, max_length=max_length, padding='max_length')
        else:
            # Apply normalization manually
            normalizer = ArabicTextNormalizer()
            normalized_texts = [normalizer.normalize(text) for text in texts]
            tokenized = tokenizer(normalized_texts, truncation=True, max_length=max_length, padding='max_length')
        
        return tokenized
    
    tokenized_dataset = dataset.map(
        tokenize,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    return tokenized_dataset


# Example usage
if __name__ == "__main__":
    # Text normalization example
    normalizer = ArabicTextNormalizer()
    
    arabic_text = "اللُّغَةُ العَرَبِيَّةُ جميلةٌ جداً"
    normalized = normalizer.normalize(arabic_text)
    
    print(f"Original:  {arabic_text}")
    print(f"Normalized: {normalized}")
    print()
    
    # Dialect detection example
    detector = ArabicDialectDetector()
    
    texts = [
        "اللغة العربية جميلة",  # MSA
        "إزك يا باشا؟ أنا بخير",  # Egyptian
        "شو حالك؟ كيفك؟",  # Levantine
        "شلونك؟ زين الله",  # Gulf
        "علاش ما جيتيش دابا؟",  # Maghrebi
    ]
    
    print("Dialect Detection:")
    print("-" * 50)
    for text in texts:
        dialect = detector.detect(text)
        print(f"{text:30s} → {dialect}")
    print()
    
    # Tokenization example (requires transformers)
    try:
        from transformers import AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained("aubmindlab/bert-base-arabertv2")
        arabic_tokenizer = ArabicTokenizer(tokenizer)
        
        text = "الذكاء الاصطناعي يغير العالم"
        tokens = arabic_tokenizer.encode(text, return_tensors='pt')
        
        print(f"Text:   {text}")
        print(f"Tokens: {tokens}")
        print(f"Shape:  {tokens.shape}")
    except ImportError:
        print("Install transformers: pip install transformers")
