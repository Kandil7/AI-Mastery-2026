"""
NLP Text Preprocessing Module.

This module provides comprehensive text preprocessing utilities,
including stemming, lemmatization, stop words removal, n-grams, and TF-IDF.

Features:
- Multiple stemming algorithms (Porter, Lancaster, Snowball)
- Lemmatization with POS tagging
- Stop words removal (multiple languages)
- N-gram generation
- TF-IDF vectorization
- Text normalization
- Feature extraction

Example Usage:
    >>> from text_preprocessing import Stemmer, Lemmatizer, StopWordsRemover
    >>> from text_preprocessing import NGramGenerator, TFIDFVectorizer
    >>> 
    >>> # Stemming
    >>> stemmer = Stemmer(algorithm='porter')
    >>> stemmed = stemmer.stem("running")
    >>> 
    >>> # TF-IDF
    >>> vectorizer = TFIDFVectorizer()
    >>> documents = ["hello world", "hello there", "world peace"]
    >>> tfidf_matrix = vectorizer.fit_transform(documents)
"""

from typing import Union, List, Dict, Tuple, Optional, Set, Iterator, Callable
import numpy as np
from numpy.typing import ArrayLike
import logging
import re
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import os

logger = logging.getLogger(__name__)


class Stemmer:
    """
    Text stemming algorithms.
    
    Supported algorithms:
    - Porter: Most common, moderate aggressiveness
    - Lancaster: More aggressive, can over-stem
    - Snowball: Improved Porter algorithm
    
    Example:
        >>> stemmer = Stemmer(algorithm='porter')
        >>> stemmer.stem("running")
        'run'
        >>> stemmer.stem("beautiful")
        'beauti'
    """
    
    def __init__(self, algorithm: str = 'porter'):
        """
        Initialize Stemmer.
        
        Args:
            algorithm: Stemming algorithm ('porter', 'lancaster', 'snowball').
        """
        self.algorithm = algorithm.lower()
        
        # Porter stemmer step definitions
        self.step1a_rules = [
            ('sses', 'ss'),
            ('ies', 'i'),
            ('ss', 'ss'),
            ('s', ''),
        ]
        
        self.step1b_rules = [
            ('eed', 'ee', 'ee'),
            ('ed', '', None),
            ('ing', '', None),
        ]
        
        logger.debug(f"Stemmer initialized: algorithm={algorithm}")
    
    def stem(self, word: str) -> str:
        """
        Stem a single word.
        
        Args:
            word: Input word.
        
        Returns:
            str: Stemmed word.
        """
        word = word.lower()
        
        if self.algorithm == 'porter':
            return self._porter_stem(word)
        elif self.algorithm == 'lancaster':
            return self._lancaster_stem(word)
        elif self.algorithm == 'snowball':
            return self._snowball_stem(word)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
    
    def stem_batch(self, words: List[str]) -> List[str]:
        """
        Stem multiple words.
        
        Args:
            words: List of words.
        
        Returns:
            List[str]: Stemmed words.
        """
        return [self.stem(word) for word in words]
    
    def _porter_stem(self, word: str) -> str:
        """Porter stemming algorithm."""
        if len(word) <= 2:
            return word
        
        # Step 1a
        for suffix, replacement in self.step1a_rules:
            if word.endswith(suffix):
                word = word[:-len(suffix)] + replacement
                break
        
        # Step 1b
        for suffix, replacement, condition in self.step1b_rules:
            if word.endswith(suffix):
                stem = word[:-len(suffix)]
                if condition == 'ee' or self._contains_vowel(stem):
                    word = stem + replacement
                    if replacement == '' and self._ends_with_cvc(stem):
                        word += 'e'
                break
        
        # Step 1c
        if word.endswith('y') and self._contains_vowel(word[:-1]):
            word = word[:-1] + 'i'
        
        # Additional simplifications
        word = re.sub(r'(.)\1+', r'\1', word)  # Remove double letters
        
        return word
    
    def _lancaster_stem(self, word: str) -> str:
        """Lancaster stemming algorithm (simplified)."""
        # Lancaster rules (simplified version)
        rules = [
            ('ing', ''),
            ('ed', ''),
            ('ly', ''),
            ('ies', 'y'),
            ('ful', ''),
            ('less', ''),
            ('ness', ''),
            ('tion', 't'),
            ('ment', ''),
            ('able', ''),
            ('ible', ''),
            ('ant', ''),
            ('ent', ''),
            ('ism', ''),
            ('est', ''),
            ('s', ''),
        ]
        
        for suffix, replacement in rules:
            if word.endswith(suffix) and len(word) > len(suffix) + 1:
                word = word[:-len(suffix)] + replacement
        
        return word
    
    def _snowball_stem(self, word: str) -> str:
        """Snowball stemming (improved Porter)."""
        # Use Porter as base with improvements
        word = self._porter_stem(word)
        
        # Additional Snowball improvements
        if word.endswith('ational'):
            word = word[:-5] + 'e'
        elif word.endswith('tional'):
            word = word[:-2]
        elif word.endswith('enci'):
            word = word[:-1] + 'e'
        elif word.endswith('anci'):
            word = word[:-1] + 'e'
        
        return word
    
    def _contains_vowel(self, word: str) -> bool:
        """Check if word contains a vowel."""
        return any(c in 'aeiou' for c in word)
    
    def _ends_with_cvc(self, word: str) -> bool:
        """Check if word ends with consonant-vowel-consonant pattern."""
        if len(word) < 3:
            return False
        
        consonants = set('bcdfghjklmnpqrstvwxyz')
        vowels = set('aeiou')
        
        return (word[-1] in consonants and 
                word[-2] in vowels and 
                word[-3] in consonants and
                word[-1] not in 'wxy')


class Lemmatizer:
    """
    Text lemmatization with basic POS tagging.
    
    Lemmatization returns the dictionary form (lemma) of a word,
    considering its part of speech.
    
    Example:
        >>> lemmatizer = Lemmatizer()
        >>> lemmatizer.lemmatize("running", pos='verb')
        'run'
        >>> lemmatizer.lemmatize("better", pos='adjective')
        'good'
    """
    
    def __init__(self):
        """Initialize Lemmatizer with rule-based transformations."""
        # Common irregular forms
        self.irregular_verbs = {
            'am': 'be', 'is': 'be', 'are': 'be', 'was': 'be', 'were': 'be',
            'been': 'be', 'being': 'be',
            'have': 'have', 'has': 'have', 'had': 'have', 'having': 'have',
            'do': 'do', 'does': 'do', 'did': 'do', 'doing': 'do',
            'go': 'go', 'goes': 'go', 'went': 'go', 'gone': 'go', 'going': 'go',
            'come': 'come', 'comes': 'come', 'came': 'come', 'coming': 'come',
            'take': 'take', 'takes': 'take', 'took': 'take', 'taken': 'take',
            'make': 'make', 'makes': 'make', 'made': 'make', 'making': 'make',
            'get': 'get', 'gets': 'get', 'got': 'get', 'gotten': 'get',
            'give': 'give', 'gives': 'give', 'gave': 'give', 'given': 'give',
            'find': 'find', 'finds': 'find', 'found': 'find',
            'think': 'think', 'thinks': 'think', 'thought': 'think',
            'see': 'see', 'sees': 'see', 'saw': 'see', 'seen': 'see',
            'know': 'know', 'knows': 'know', 'knew': 'know', 'known': 'know',
            'run': 'run', 'runs': 'run', 'ran': 'run', 'running': 'run',
            'eat': 'eat', 'eats': 'eat', 'ate': 'eat', 'eaten': 'eat',
            'write': 'write', 'writes': 'write', 'wrote': 'write', 'written': 'write',
            'read': 'read', 'reads': 'read',  # pronounced differently
            'sing': 'sing', 'sings': 'sing', 'sang': 'sing', 'sung': 'sing',
            'swim': 'swim', 'swims': 'swim', 'swam': 'swim', 'swum': 'swim',
            'begin': 'begin', 'begins': 'begin', 'began': 'begin', 'begun': 'begin',
            'break': 'break', 'breaks': 'break', 'broke': 'break', 'broken': 'break',
            'choose': 'choose', 'chooses': 'choose', 'chose': 'choose', 'chosen': 'choose',
            'drive': 'drive', 'drives': 'drive', 'drove': 'drive', 'driven': 'drive',
            'fall': 'fall', 'falls': 'fall', 'fell': 'fall', 'fallen': 'fall',
            'fly': 'fly', 'flies': 'fly', 'flew': 'fly', 'flown': 'fly',
            'forget': 'forget', 'forgets': 'forget', 'forgot': 'forget', 'forgotten': 'forget',
            'forgive': 'forgive', 'forgives': 'forgive', 'forgave': 'forgive', 'forgiven': 'forgive',
            'freeze': 'freeze', 'freezes': 'freeze', 'froze': 'freeze', 'frozen': 'freeze',
            'grow': 'grow', 'grows': 'grow', 'grew': 'grow', 'grown': 'grow',
            'hide': 'hide', 'hides': 'hide', 'hid': 'hide', 'hidden': 'hide',
            'hold': 'hold', 'holds': 'hold', 'held': 'hold',
            'keep': 'keep', 'keeps': 'keep', 'kept': 'keep',
            'lay': 'lay', 'lays': 'lay', 'laid': 'lay',
            'lead': 'lead', 'leads': 'lead', 'led': 'lead',
            'leave': 'leave', 'leaves': 'leave', 'left': 'leave',
            'lend': 'lend', 'lends': 'lend', 'lent': 'lend',
            'let': 'let', 'lets': 'let',
            'lie': 'lie', 'lies': 'lie', 'lay': 'lie', 'lain': 'lie',
            'lose': 'lose', 'loses': 'lose', 'lost': 'lose',
            'mean': 'mean', 'means': 'mean', 'meant': 'mean',
            'meet': 'meet', 'meets': 'meet', 'met': 'meet',
            'pay': 'pay', 'pays': 'pay', 'paid': 'pay',
            'put': 'put', 'puts': 'put',
            'ride': 'ride', 'rides': 'ride', 'rode': 'ride', 'ridden': 'ride',
            'ring': 'ring', 'rings': 'ring', 'rang': 'ring', 'rung': 'ring',
            'rise': 'rise', 'rises': 'rise', 'rose': 'rise', 'risen': 'rise',
            'say': 'say', 'says': 'say', 'said': 'say',
            'sell': 'sell', 'sells': 'sell', 'sold': 'sell',
            'send': 'send', 'sends': 'send', 'sent': 'send',
            'set': 'set', 'sets': 'set',
            'shake': 'shake', 'shakes': 'shake', 'shook': 'shake', 'shaken': 'shake',
            'shine': 'shine', 'shines': 'shine', 'shone': 'shine',
            'shoot': 'shoot', 'shoots': 'shoot', 'shot': 'shoot',
            'show': 'show', 'shows': 'show', 'showed': 'show', 'shown': 'show',
            'shut': 'shut', 'shuts': 'shut',
            'sit': 'sit', 'sits': 'sit', 'sat': 'sit',
            'sleep': 'sleep', 'sleeps': 'sleep', 'slept': 'sleep',
            'speak': 'speak', 'speaks': 'speak', 'spoke': 'speak', 'spoken': 'speak',
            'spend': 'spend', 'spends': 'spend', 'spent': 'spend',
            'split': 'split', 'splits': 'split',
            'spread': 'spread', 'spreads': 'spread',
            'stand': 'stand', 'stands': 'stand', 'stood': 'stand',
            'steal': 'steal', 'steals': 'steal', 'stole': 'steal', 'stolen': 'steal',
            'stick': 'stick', 'sticks': 'stick', 'stuck': 'stick',
            'strike': 'strike', 'strikes': 'strike', 'struck': 'strike',
            'swear': 'swear', 'swears': 'swear', 'swore': 'swear', 'sworn': 'swear',
            'sweep': 'sweep', 'sweeps': 'sweep', 'swept': 'sweep',
            'teach': 'teach', 'teaches': 'teach', 'taught': 'teach',
            'tear': 'tear', 'tears': 'tear', 'tore': 'tear', 'torn': 'tear',
            'tell': 'tell', 'tells': 'tell', 'told': 'tell',
            'throw': 'throw', 'throws': 'throw', 'threw': 'throw', 'thrown': 'throw',
            'understand': 'understand', 'understands': 'understand', 'understood': 'understand',
            'wake': 'wake', 'wakes': 'wake', 'woke': 'wake', 'woken': 'wake',
            'wear': 'wear', 'wears': 'wear', 'wore': 'wear', 'worn': 'wear',
            'win': 'win', 'wins': 'win', 'won': 'win',
        }
        
        self.irregular_nouns = {
            'children': 'child',
            'men': 'man',
            'women': 'woman',
            'feet': 'foot',
            'teeth': 'tooth',
            'geese': 'goose',
            'mice': 'mouse',
            'lice': 'louse',
            'oxen': 'ox',
            'people': 'person',
            'dice': 'die',
        }
        
        self.irregular_adjectives = {
            'better': 'good',
            'best': 'good',
            'worse': 'bad',
            'worst': 'bad',
            'more': 'much',
            'most': 'much',
            'less': 'little',
            'least': 'little',
            'farther': 'far',
            'farthest': 'far',
            'further': 'far',
            'furthest': 'far',
            'older': 'old',
            'oldest': 'old',
            'elder': 'old',
            'eldest': 'old',
        }
        
        logger.debug("Lemmatizer initialized")
    
    def lemmatize(self, word: str, pos: str = 'noun') -> str:
        """
        Lemmatize a word.
        
        Args:
            word: Input word.
            pos: Part of speech ('noun', 'verb', 'adjective', 'adverb').
        
        Returns:
            str: Lemmatized word.
        """
        word = word.lower()
        
        # Check irregular forms first
        if pos == 'verb' and word in self.irregular_verbs:
            return self.irregular_verbs[word]
        
        if pos == 'noun' and word in self.irregular_nouns:
            return self.irregular_nouns[word]
        
        if pos == 'adjective' and word in self.irregular_adjectives:
            return self.irregular_adjectives[word]
        
        # Apply regular rules
        if pos == 'verb':
            return self._lemmatize_verb(word)
        elif pos == 'noun':
            return self._lemmatize_noun(word)
        elif pos == 'adjective':
            return self._lemmatize_adjective(word)
        elif pos == 'adverb':
            return self._lemmatize_adverb(word)
        
        return word
    
    def _lemmatize_verb(self, word: str) -> str:
        """Lemmatize verb."""
        # Remove -ing
        if word.endswith('ing') and len(word) > 4:
            stem = word[:-3]
            if stem.endswith('e'):
                return stem
            elif len(stem) > 1 and stem[-1] == stem[-2]:
                return stem[:-1]
            return stem
        
        # Remove -ed
        if word.endswith('ed') and len(word) > 3:
            stem = word[:-2]
            if stem.endswith('e'):
                return stem
            elif stem.endswith('i'):
                return stem[:-1] + 'y'
            return stem
        
        # Remove -es
        if word.endswith('es') and len(word) > 3:
            if word.endswith('ies'):
                return word[:-3] + 'y'
            if word.endswith('ses') or word.endswith('zes') or word.endswith('xes'):
                return word[:-1]
            return word[:-2]
        
        # Remove -s
        if word.endswith('s') and not word.endswith('ss') and len(word) > 2:
            return word[:-1]
        
        return word
    
    def _lemmatize_noun(self, word: str) -> str:
        """Lemmatize noun."""
        # Plural -ies -> -y
        if word.endswith('ies') and len(word) > 3:
            return word[:-3] + 'y'
        
        # Plural -es
        if word.endswith('es') and len(word) > 3:
            if word.endswith('ses') or word.endswith('zes') or word.endswith('xes'):
                return word[:-1]
            if word.endswith('ches') or word.endswith('shes'):
                return word[:-2]
        
        # Plural -s
        if word.endswith('s') and not word.endswith('ss') and len(word) > 2:
            return word[:-1]
        
        return word
    
    def _lemmatize_adjective(self, word: str) -> str:
        """Lemmatize adjective."""
        # -est -> base
        if word.endswith('est') and len(word) > 4:
            return word[:-3]
        
        # -er -> base
        if word.endswith('er') and len(word) > 3:
            return word[:-2]
        
        # -ly -> base (for adverbs used as adjectives)
        if word.endswith('ly') and len(word) > 3:
            return word[:-2]
        
        return word
    
    def _lemmatize_adverb(self, word: str) -> str:
        """Lemmatize adverb."""
        # -ly -> adjective
        if word.endswith('ly') and len(word) > 3:
            return word[:-2]
        
        return word
    
    def lemmatize_batch(
        self,
        words: List[str],
        pos_tags: Optional[List[str]] = None
    ) -> List[str]:
        """
        Lemmatize multiple words.
        
        Args:
            words: List of words.
            pos_tags: Optional list of POS tags.
        
        Returns:
            List[str]: Lemmatized words.
        """
        if pos_tags is None:
            pos_tags = ['noun'] * len(words)
        
        return [self.lemmatize(w, p) for w, p in zip(words, pos_tags)]


class StopWordsRemover:
    """
    Stop words removal for multiple languages.
    
    Example:
        >>> remover = StopWordsRemover(language='english')
        >>> tokens = ["the", "quick", "brown", "fox"]
        >>> filtered = remover.remove(tokens)
        >>> filtered
        ['quick', 'brown', 'fox']
    """
    
    # English stop words
    ENGLISH_STOP_WORDS = {
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
        'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
        'to', 'was', 'were', 'will', 'with', 'about', 'above', 'after',
        'again', 'against', 'all', 'am', 'any', 'because', 'been', 'before',
        'below', 'between', 'both', 'but', 'can', 'could', 'did', 'do',
        'does', 'doing', 'down', 'during', 'each', 'few', 'further', 'had',
        'her', 'here', 'hers', 'herself', 'him', 'himself', 'his', 'how',
        'i', 'if', 'into', 'just', 'me', 'more', 'most', 'my', 'myself',
        'no', 'nor', 'not', 'now', 'only', 'or', 'other', 'our', 'ours',
        'ourselves', 'out', 'over', 'own', 'same', 'she', 'should', 'so',
        'some', 'such', 'than', 'their', 'theirs', 'them', 'themselves',
        'then', 'there', 'these', 'they', 'this', 'those', 'through',
        'too', 'under', 'until', 'up', 'very', 'what', 'when', 'where',
        'which', 'while', 'who', 'whom', 'why', 'would', 'you', 'your',
        'yours', 'yourself', 'yourselves', 'i\'m', 'i\'ll', 'i\'ve',
        'don\'t', 'doesn\'t', 'didn\'t', 'won\'t', 'wouldn\'t', 'couldn\'t',
        'shouldn\'t', 'can\'t', 'isn\'t', 'aren\'t', 'wasn\'t', 'weren\'t',
        'let\'s', 'that\'s', 'who\'s', 'what\'s', 'where\'s', 'when\'s',
        'why\'s', 'how\'s',
    }
    
    # Spanish stop words
    SPANISH_STOP_WORDS = {
        'el', 'la', 'los', 'las', 'de', 'del', 'en', 'un', 'una', 'unos',
        'unas', 'y', 'o', 'pero', 'que', 'si', 'no', 'es', 'son', 'ser',
        'estar', 'esta', 'este', 'estos', 'estas', 'con', 'por', 'para',
        'como', 'cuando', 'donde', 'quien', 'cual', 'cuales', 'muy', 'mas',
        'menos', 'mucho', 'poco', 'todo', 'todos', 'todas', 'mi', 'mis',
        'tu', 'tus', 'su', 'sus', 'nuestro', 'nuestra', 'vuestro', 'vuestra',
    }
    
    # French stop words
    FRENCH_STOP_WORDS = {
        'le', 'la', 'les', 'un', 'une', 'des', 'du', 'de', 'et', 'est',
        'en', 'que', 'qui', 'dans', 'ce', 'il', 'elle', 'son', 'sa', 'ses',
        'se', 'ne', 'pas', 'plus', 'par', 'sur', 'pour', 'avec', 'tout',
        'faire', 'sont', 'mais', 'nous', 'vous', 'leur', 'leurs', 'cette',
        'ces', 'aux', 'au', 'ou', 'donc', 'dont', 'ici', 'y', 'a', 'ai',
    }
    
    # German stop words
    GERMAN_STOP_WORDS = {
        'der', 'die', 'das', 'den', 'dem', 'des', 'ein', 'eine', 'einer',
        'einem', 'einen', 'und', 'ist', 'sind', 'war', 'waren', 'sein',
        'seine', 'seiner', 'mit', 'von', 'zu', 'zum', 'zur', 'als', 'auch',
        'aber', 'oder', 'wenn', 'nicht', 'noch', 'nur', 'schon', 'so',
        'wie', 'was', 'wer', 'wo', 'da', 'hier', 'dort', 'dann', 'nun',
    }
    
    def __init__(self, language: str = 'english', custom_stop_words: Optional[Set[str]] = None):
        """
        Initialize StopWordsRemover.
        
        Args:
            language: Language for stop words.
            custom_stop_words: Optional custom stop words to add.
        """
        self.language = language.lower()
        
        if self.language == 'english':
            self.stop_words = self.ENGLISH_STOP_WORDS.copy()
        elif self.language == 'spanish':
            self.stop_words = self.SPANISH_STOP_WORDS.copy()
        elif self.language == 'french':
            self.stop_words = self.FRENCH_STOP_WORDS.copy()
        elif self.language == 'german':
            self.stop_words = self.GERMAN_STOP_WORDS.copy()
        else:
            self.stop_words = set()
            logger.warning(f"Unknown language: {language}, using empty stop words")
        
        if custom_stop_words:
            self.stop_words.update(custom_stop_words)
        
        logger.debug(f"StopWordsRemover: language={language}, "
                    f"{len(self.stop_words)} stop words")
    
    def remove(self, tokens: List[str]) -> List[str]:
        """
        Remove stop words from tokens.
        
        Args:
            tokens: List of tokens.
        
        Returns:
            List[str]: Filtered tokens.
        """
        return [t for t in tokens if t.lower() not in self.stop_words]
    
    def is_stop_word(self, word: str) -> bool:
        """
        Check if word is a stop word.
        
        Args:
            word: Word to check.
        
        Returns:
            bool: True if stop word.
        """
        return word.lower() in self.stop_words
    
    def add_stop_words(self, words: List[str]) -> None:
        """
        Add custom stop words.
        
        Args:
            words: Words to add.
        """
        self.stop_words.update(w.lower() for w in words)
    
    def remove_stop_words(self, words: List[str]) -> None:
        """
        Remove words from stop words list.
        
        Args:
            words: Words to remove.
        """
        self.stop_words.difference_update(w.lower() for w in words)


class NGramGenerator:
    """
    N-gram generation from text.
    
    Example:
        >>> generator = NGramGenerator(n=2)
        >>> tokens = ["the", "quick", "brown", "fox"]
        >>> ngrams = generator.generate(tokens)
        >>> ngrams
        [('the', 'quick'), ('quick', 'brown'), ('brown', 'fox')]
    """
    
    def __init__(
        self,
        n: int = 2,
        pad_left: bool = False,
        pad_right: bool = False,
        pad_symbol: str = '<PAD>'
    ):
        """
        Initialize NGramGenerator.
        
        Args:
            n: N-gram size.
            pad_left: Add padding at start.
            pad_right: Add padding at end.
            pad_symbol: Padding symbol.
        """
        self.n = n
        self.pad_left = pad_left
        self.pad_right = pad_right
        self.pad_symbol = pad_symbol
    
    def generate(self, tokens: List[str]) -> List[Tuple[str, ...]]:
        """
        Generate n-grams from tokens.
        
        Args:
            tokens: Input tokens.
        
        Returns:
            List[Tuple]: List of n-grams.
        """
        if self.n <= 0:
            raise ValueError("n must be positive")
        
        if self.n == 1:
            return [(t,) for t in tokens]
        
        # Add padding if requested
        padded = tokens.copy()
        if self.pad_left:
            padded = [self.pad_symbol] * (self.n - 1) + padded
        if self.pad_right:
            padded = padded + [self.pad_symbol] * (self.n - 1)
        
        if len(padded) < self.n:
            return []
        
        # Generate n-grams
        ngrams = []
        for i in range(len(padded) - self.n + 1):
            ngram = tuple(padded[i:i + self.n])
            ngrams.append(ngram)
        
        return ngrams
    
    def generate_from_text(
        self,
        text: str,
        tokenizer: Optional[Callable[[str], List[str]]] = None
    ) -> List[Tuple[str, ...]]:
        """
        Generate n-grams from text.
        
        Args:
            text: Input text.
            tokenizer: Optional tokenizer function.
        
        Returns:
            List[Tuple]: List of n-grams.
        """
        if tokenizer is None:
            tokens = text.lower().split()
        else:
            tokens = tokenizer(text)
        
        return self.generate(tokens)
    
    def count_ngrams(
        self,
        texts: List[str],
        normalize: bool = False
    ) -> Counter:
        """
        Count n-grams across multiple texts.
        
        Args:
            texts: List of texts.
            normalize: Normalize counts to frequencies.
        
        Returns:
            Counter: N-gram counts.
        """
        counter = Counter()
        
        for text in texts:
            ngrams = self.generate_from_text(text)
            counter.update(ngrams)
        
        if normalize:
            total = sum(counter.values())
            if total > 0:
                counter = Counter({k: v / total for k, v in counter.items()})
        
        return counter


class TFIDFVectorizer:
    """
    TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer.
    
    TF-IDF measures word importance in a document relative to a corpus.
    
    TF(t, d) = count of t in d / total terms in d
    IDF(t) = log(N / (1 + df(t)))
    TF-IDF(t, d) = TF(t, d) * IDF(t)
    
    Example:
        >>> vectorizer = TFIDFVectorizer()
        >>> documents = ["hello world", "hello there", "world peace"]
        >>> tfidf_matrix = vectorizer.fit_transform(documents)
        >>> tfidf_matrix.shape
        (3, 4)
    """
    
    def __init__(
        self,
        max_features: Optional[int] = None,
        min_df: int = 1,
        max_df: float = 1.0,
        ngram_range: Tuple[int, int] = (1, 1),
        sublinear_tf: bool = False,
        smooth_idf: bool = True,
        norm: str = 'l2'
    ):
        """
        Initialize TFIDFVectorizer.
        
        Args:
            max_features: Maximum number of features.
            min_df: Minimum document frequency.
            max_df: Maximum document frequency (fraction or count).
            ngram_range: (min_n, max_n) for n-grams.
            sublinear_tf: Use log(1 + tf) instead of tf.
            smooth_idf: Add 1 to document frequencies.
            norm: Normalization ('l1', 'l2', or None).
        """
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.ngram_range = ngram_range
        self.sublinear_tf = sublinear_tf
        self.smooth_idf = smooth_idf
        self.norm = norm
        
        self.vocabulary_: Dict[str, int] = {}
        self.idf_: Optional[np.ndarray] = None
        self.n_docs_: int = 0
        
        logger.debug(f"TFIDFVectorizer initialized: ngram_range={ngram_range}")
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        return text.lower().split()
    
    def _get_ngrams(self, tokens: List[str]) -> List[str]:
        """Get n-grams from tokens."""
        ngrams = []
        min_n, max_n = self.ngram_range
        
        for n in range(min_n, max_n + 1):
            generator = NGramGenerator(n=n)
            ngrams.extend([' '.join(ng) for ng in generator.generate(tokens)])
        
        return ngrams
    
    def fit(
        self,
        documents: List[str],
        y: Optional[np.ndarray] = None
    ) -> 'TFIDFVectorizer':
        """
        Fit vectorizer to documents.
        
        Args:
            documents: List of documents.
            y: Ignored (for sklearn compatibility).
        
        Returns:
            self: Fitted vectorizer.
        """
        self.n_docs_ = len(documents)
        
        # Count document frequencies
        doc_freq = Counter()
        term_counts = []
        
        for doc in documents:
            tokens = self._tokenize(doc)
            ngrams = self._get_ngrams(tokens)
            
            # Count terms in this document
            term_count = Counter(ngrams)
            term_counts.append(term_count)
            
            # Update document frequency
            doc_freq.update(term_count.keys())
        
        # Filter by document frequency
        min_df_count = self.min_df if isinstance(self.min_df, int) else int(self.min_df * self.n_docs_)
        max_df_count = self.max_df if isinstance(self.max_df, int) else int(self.max_df * self.n_docs_)
        
        valid_terms = {
            term for term, df in doc_freq.items()
            if min_df_count <= df <= max_df_count
        }
        
        # Limit features
        if self.max_features is not None:
            sorted_terms = sorted(valid_terms, key=lambda t: -doc_freq[t])
            valid_terms = set(sorted_terms[:self.max_features])
        
        # Build vocabulary
        self.vocabulary_ = {term: idx for idx, term in enumerate(sorted(valid_terms))}
        
        # Compute IDF
        n_features = len(self.vocabulary_)
        self.idf_ = np.zeros(n_features)
        
        for term, idx in self.vocabulary_.items():
            df = doc_freq[term]
            if self.smooth_idf:
                self.idf_[idx] = np.log((self.n_docs_ + 1) / (df + 1)) + 1
            else:
                self.idf_[idx] = np.log(self.n_docs_ / df)
        
        logger.info(f"TFIDFVectorizer fitted: {n_features} features")
        return self
    
    def transform(self, documents: List[str]) -> np.ndarray:
        """
        Transform documents to TF-IDF matrix.
        
        Args:
            documents: List of documents.
        
        Returns:
            np.ndarray: TF-IDF matrix (n_docs, n_features).
        """
        if not self.vocabulary_:
            raise ValueError("Vectorizer not fitted. Call fit() first.")
        
        n_docs = len(documents)
        n_features = len(self.vocabulary_)
        
        tfidf_matrix = np.zeros((n_docs, n_features))
        
        for doc_idx, doc in enumerate(documents):
            tokens = self._tokenize(doc)
            ngrams = self._get_ngrams(tokens)
            term_count = Counter(ngrams)
            
            # Compute TF
            total_terms = len(ngrams) if ngrams else 1
            
            for term, count in term_count.items():
                if term in self.vocabulary_:
                    feat_idx = self.vocabulary_[term]
                    
                    # TF
                    if self.sublinear_tf:
                        tf = 1 + np.log(count) if count > 0 else 0
                    else:
                        tf = count / total_terms
                    
                    # TF-IDF
                    tfidf_matrix[doc_idx, feat_idx] = tf * self.idf_[feat_idx]
        
        # Normalize
        if self.norm == 'l2':
            norms = np.linalg.norm(tfidf_matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1
            tfidf_matrix = tfidf_matrix / norms
        elif self.norm == 'l1':
            norms = np.sum(np.abs(tfidf_matrix), axis=1, keepdims=True)
            norms[norms == 0] = 1
            tfidf_matrix = tfidf_matrix / norms
        
        return tfidf_matrix
    
    def fit_transform(
        self,
        documents: List[str],
        y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Fit and transform documents.
        
        Args:
            documents: List of documents.
            y: Ignored.
        
        Returns:
            np.ndarray: TF-IDF matrix.
        """
        self.fit(documents, y)
        return self.transform(documents)
    
    def get_feature_names(self) -> List[str]:
        """Get feature names (terms)."""
        return sorted(self.vocabulary_.keys(), key=lambda k: self.vocabulary_[k])
    
    def get_top_terms(
        self,
        tfidf_vector: np.ndarray,
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Get top terms from TF-IDF vector.
        
        Args:
            tfidf_vector: TF-IDF vector.
            top_k: Number of top terms.
        
        Returns:
            List[Tuple]: (term, score) pairs.
        """
        feature_names = self.get_feature_names()
        indices = np.argsort(tfidf_vector)[::-1][:top_k]
        
        return [(feature_names[i], tfidf_vector[i]) for i in indices if tfidf_vector[i] > 0]


class TextNormalizer:
    """
    Text normalization utilities.
    
    Example:
        >>> normalizer = TextNormalizer()
        >>> text = "Hello, WORLD! 123"
        >>> normalized = normalizer.normalize(text)
    """
    
    def __init__(
        self,
        lowercase: bool = True,
        remove_punctuation: bool = True,
        remove_numbers: bool = False,
        remove_extra_whitespace: bool = True,
        normalize_whitespace: bool = True
    ):
        """
        Initialize TextNormalizer.
        
        Args:
            lowercase: Convert to lowercase.
            remove_punctuation: Remove punctuation.
            remove_numbers: Remove numbers.
            remove_extra_whitespace: Remove extra whitespace.
            normalize_whitespace: Normalize whitespace to single spaces.
        """
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_numbers = remove_numbers
        self.remove_extra_whitespace = remove_extra_whitespace
        self.normalize_whitespace = normalize_whitespace
    
    def normalize(self, text: str) -> str:
        """
        Normalize text.
        
        Args:
            text: Input text.
        
        Returns:
            str: Normalized text.
        """
        if self.lowercase:
            text = text.lower()
        
        if self.remove_punctuation:
            text = re.sub(r'[^\w\s]', '', text)
        
        if self.remove_numbers:
            text = re.sub(r'\d+', '', text)
        
        if self.normalize_whitespace:
            text = re.sub(r'\s+', ' ', text)
        
        if self.remove_extra_whitespace:
            text = text.strip()
        
        return text
    
    def normalize_batch(self, texts: List[str]) -> List[str]:
        """
        Normalize multiple texts.
        
        Args:
            texts: List of texts.
        
        Returns:
            List[str]: Normalized texts.
        """
        return [self.normalize(t) for t in texts]


if __name__ == "__main__":
    # Example usage and demonstrations
    print("=" * 60)
    print("Text Preprocessing Module - Demonstration")
    print("=" * 60)
    
    # Stemming
    print("\n1. Stemming:")
    stemmer = Stemmer(algorithm='porter')
    words = ["running", "runs", "ran", "beautiful", "beautifully", "connection"]
    for word in words:
        print(f"   {word:15} -> {stemmer.stem(word)}")
    
    # Lemmatization
    print("\n2. Lemmatization:")
    lemmatizer = Lemmatizer()
    words_with_pos = [
        ("running", "verb"),
        ("runs", "verb"),
        ("better", "adjective"),
        ("children", "noun"),
        ("mice", "noun"),
        ("went", "verb"),
    ]
    for word, pos in words_with_pos:
        print(f"   {word:15} ({pos:9}) -> {lemmatizer.lemmatize(word, pos)}")
    
    # Stop words removal
    print("\n3. Stop Words Removal:")
    remover = StopWordsRemover(language='english')
    tokens = ["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]
    filtered = remover.remove(tokens)
    print(f"   Original: {tokens}")
    print(f"   Filtered: {filtered}")
    
    # N-grams
    print("\n4. N-grams:")
    generator = NGramGenerator(n=2)
    tokens = ["the", "quick", "brown", "fox"]
    bigrams = generator.generate(tokens)
    print(f"   Tokens: {tokens}")
    print(f"   Bigrams: {bigrams}")
    
    generator3 = NGramGenerator(n=3)
    trigrams = generator3.generate(tokens)
    print(f"   Trigrams: {trigrams}")
    
    # TF-IDF
    print("\n5. TF-IDF Vectorization:")
    documents = [
        "the cat sat on the mat",
        "the dog sat on the log",
        "cats and dogs are pets",
        "the cat and the dog",
    ]
    
    vectorizer = TFIDFVectorizer(ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(documents)
    
    print(f"   Documents: {len(documents)}")
    print(f"   Features: {len(vectorizer.get_feature_names())}")
    print(f"   TF-IDF matrix shape: {tfidf_matrix.shape}")
    
    # Show top terms for first document
    top_terms = vectorizer.get_top_terms(tfidf_matrix[0], top_k=5)
    print(f"   Top terms for doc 1: {top_terms}")
    
    # Text normalization
    print("\n6. Text Normalization:")
    normalizer = TextNormalizer(
        lowercase=True,
        remove_punctuation=True,
        remove_numbers=True
    )
    text = "Hello, WORLD! 123 This is a TEST..."
    normalized = normalizer.normalize(text)
    print(f"   Original: {text}")
    print(f"   Normalized: {normalized}")
    
    print("\n" + "=" * 60)
