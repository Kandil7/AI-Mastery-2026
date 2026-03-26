"""
Advanced Arabic NLP Examples
=============================

Complete examples for:
- Named Entity Recognition (NER)
- Part-of-Speech (POS) Tagging
- Sentiment Analysis
- Dialect Translation
- Text Summarization
- Question Answering
"""

import re
from typing import List, Dict, Tuple
from dataclasses import dataclass


# ============================================================================
# Part 1: Arabic Named Entity Recognition (NER)
# ============================================================================

@dataclass
class Entity:
    """Named entity"""
    text: str
    label: str
    start: int
    end: int
    confidence: float = 1.0


class ArabicNER:
    """
    Arabic Named Entity Recognition
    
    Recognizes:
    - PERSON (شخص)
    - LOCATION (موقع)
    - ORGANIZATION (منظمة)
    - DATE (تاريخ)
    - MONEY (مال)
    - PERCENT (نسبة)
    """
    
    def __init__(self):
        # Arabic entity patterns
        self.patterns = {
            'PERSON': [
                r'(?:السيد|الأستاذ|الدكتور|الشيخ)\s+(\w+)',
                r'(\w+(?: بن | آل | ال)\w+)',
            ],
            'LOCATION': [
                r'(?:في|إلى|من)\s+(\w+(?:ستان|يا|ة|ان))',
                r'(العاصمة|المدينة|القرية)\s+(\w+)',
            ],
            'ORGANIZATION': [
                r'(شركة|مؤسسة|وزارة|جامعة)\s+(\w+)',
                r'(الأمم المتحدة|الجامعة العربية)',
            ],
            'DATE': [
                r'(\d{1,2}\s+(?:يناير|فبراير|مارس|أبريل|مايو|يونيو|يوليو|أغسطس|سبتمبر|أكتوبر|نوفمبر|ديسمبر)\s+\d{4})',
                r'(اليوم|غداً|أمس|الأسبوع\s+الماضي|الشهر\s+الماضي)',
            ],
            'MONEY': [
                r'(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:دولار|ريال|درهم|دينار|جنيه)',
            ],
            'PERCENT': [
                r'(\d+(?:\.\d+)?)\s*%',
                r'(\d+(?:\.\d+)?)\s*(?:في\s+المائة|بالمائة)',
            ]
        }
    
    def extract_entities(self, text: str) -> List[Entity]:
        """Extract named entities from text"""
        entities = []
        
        for label, patterns in self.patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text)
                for match in matches:
                    entity = Entity(
                        text=match.group(0),
                        label=label,
                        start=match.start(),
                        end=match.end(),
                        confidence=0.9
                    )
                    entities.append(entity)
        
        # Sort by position
        entities.sort(key=lambda e: e.start)
        
        return entities
    
    def extract_entities_batch(self, texts: List[str]) -> List[List[Entity]]:
        """Extract entities from multiple texts"""
        return [self.extract_entities(text) for text in texts]


# ============================================================================
# Part 2: Arabic Part-of-Speech (POS) Tagging
# ============================================================================

class ArabicPOSTagger:
    """
    Arabic Part-of-Speech Tagging
    
    Tags:
    - NOUN (اسم)
    - VERB (فعل)
    - ADJ (صفة)
    - ADV (ظرف)
    - PRON (ضمير)
    - PREP (حرف جر)
    - CONJ (حرف عطف)
    - DET (أداة تعريف)
    """
    
    def __init__(self):
        # Arabic POS patterns
        self.patterns = {
            'VERB': [
                r'\b(?:ذهب|قال|فعل|كتب|قرأ|أكل|شرب|نام|استيقظ|عمل)\b',
                r'\b(?:يذهب|يقول|يفعل|يكتب|يقرأ|يأكل|يشرب|ينام)\b',
            ],
            'NOUN': [
                r'\b(?:كتاب|قلم|مدرسة|جامعة|مدينة|بيت|سيارة|طائرة)\b',
                r'\b(?:رجل|امرأة|طفل|طالب|معلم|طبيب|مهندس)\b',
            ],
            'ADJ': [
                r'\b(?:كبير|صغير|جميل|قبيح|جيد|سيء|سريع|بطيء)\b',
                r'\b(?:عربي|إسلامي|وطني|إنساني|علمي)\b',
            ],
            'PRON': [
                r'\b(?:أنا|أنت|هو|هي|نحن|أنتم|هم|هن|أنتما|هما)\b',
                r'\b(?:أنا|ني|ي|ك|ه|نا|كم|هم)\b',
            ],
            'PREP': [
                r'\b(?:في|إلى|من|على|عن|بـ|لـ|كـ)\b',
            ],
            'CONJ': [
                r'\b(?:و|أو|لكن|فإن|أن|لن|كي)\b',
            ],
            'DET': [
                r'\bال\b',
            ]
        }
    
    def tag(self, text: str) -> List[Tuple[str, str]]:
        """
        Tag words with POS tags
        
        Returns:
            List of (word, tag) tuples
        """
        words = text.split()
        tagged = []
        
        for word in words:
            tag = 'UNKNOWN'
            
            for pos_tag, patterns in self.patterns.items():
                for pattern in patterns:
                    if re.search(pattern, word):
                        tag = pos_tag
                        break
                if tag != 'UNKNOWN':
                    break
            
            tagged.append((word, tag))
        
        return tagged
    
    def tag_sentence(self, text: str) -> str:
        """Tag sentence and return formatted output"""
        tagged = self.tag(text)
        return ' '.join([f"{word}/{tag}" for word, tag in tagged])


# ============================================================================
# Part 3: Arabic Sentiment Analysis
# ============================================================================

class ArabicSentimentAnalyzer:
    """
    Arabic Sentiment Analysis
    
    Analyzes sentiment as:
    - POSITIVE (إيجابي)
    - NEGATIVE (سلبي)
    - NEUTRAL (محايد)
    """
    
    def __init__(self):
        # Sentiment lexicons
        self.positive_words = {
            'جيد', 'جيدة', 'عظيم', 'رائع', 'ممتاز', 'جميل', 'جميل',
            'سعيد', 'فرح', 'ناجح', 'فوز', 'حب', 'أحب', 'يعجب',
            'مفيد', 'قيمة', 'جودة', 'توصية', 'أنصح', 'أفضل',
            'شكر', 'thanks', 'merci', 'Merci'
        }
        
        self.negative_words = {
            'سيء', 'سيئة', 'رهيب', 'فظيع', 'تعبان', 'حزين',
            'فاشل', 'فشل', 'خسارة', 'خسر', 'كره', 'أكره',
            'غير', 'لا', 'مش', 'مو', 'مفيش', 'بدون',
            'مشكلة', 'خطأ', 'خطيئة', 'سيء', 'أسوأ'
        }
        
        self.intensifiers = {
            'جداً': 2.0,
            'كثيراً': 1.5,
            'للغاية': 2.0,
            'أوي': 1.8,  # Egyptian
            'كتير': 1.5,  # Levantine
        }
        
        self.negators = {
            'لا', 'ليس', 'مش', 'مو', 'مفيش', 'ما', 'لن', 'لم'
        }
    
    def analyze(self, text: str) -> Dict:
        """
        Analyze sentiment of text
        
        Returns:
            Dict with sentiment label and scores
        """
        words = text.lower().split()
        
        positive_score = 0.0
        negative_score = 0.0
        intensifier = 1.0
        negated = False
        
        for i, word in enumerate(words):
            # Check for intensifiers
            if word in self.intensifiers:
                intensifier = self.intensifiers[word]
            
            # Check for negators
            if word in self.negators:
                negated = True
            
            # Check for positive words
            if word in self.positive_words:
                score = 1.0 * intensifier
                if negated:
                    negative_score += score
                else:
                    positive_score += score
                negated = False
                intensifier = 1.0
            
            # Check for negative words
            if word in self.negative_words:
                score = 1.0 * intensifier
                if negated:
                    positive_score += score
                else:
                    negative_score += score
                negated = False
                intensifier = 1.0
        
        # Determine sentiment
        total = positive_score + negative_score
        
        if total == 0:
            sentiment = 'NEUTRAL'
            confidence = 0.5
        else:
            positive_ratio = positive_score / total
            negative_ratio = negative_score / total
            
            if positive_ratio > 0.6:
                sentiment = 'POSITIVE'
            elif negative_ratio > 0.6:
                sentiment = 'NEGATIVE'
            else:
                sentiment = 'NEUTRAL'
            
            confidence = max(positive_ratio, negative_ratio)
        
        return {
            'sentiment': sentiment,
            'positive_score': positive_score,
            'negative_score': negative_score,
            'confidence': confidence,
            'label_ar': self._get_arabic_label(sentiment)
        }
    
    def _get_arabic_label(self, sentiment: str) -> str:
        """Get Arabic label for sentiment"""
        labels = {
            'POSITIVE': 'إيجابي',
            'NEGATIVE': 'سلبي',
            'NEUTRAL': 'محايد'
        }
        return labels.get(sentiment, 'محايد')
    
    def analyze_batch(self, texts: List[str]) -> List[Dict]:
        """Analyze sentiment for multiple texts"""
        return [self.analyze(text) for text in texts]


# ============================================================================
# Part 4: Arabic Dialect Translation
# ============================================================================

class ArabicDialectTranslator:
    """
    Arabic Dialect Translation
    
    Translates between:
    - Modern Standard Arabic (MSA)
    - Egyptian
    - Levantine
    - Gulf
    - Maghrebi
    """
    
    def __init__(self):
        # Dialect dictionaries
        self.msa_to_egyptian = {
            'كيف': 'إزك',
            'حالك': 'عاملك إيه',
            'أريد': 'عايز',
            'أنا': 'أنا',
            'نعم': 'أيوه',
            'لا': 'لأ',
            'شكراً': 'شكراً',
            'عفواً': 'عفواً',
            'الآن': 'دلوقتي',
            'جداً': 'أوي',
            'جيد': 'كويّس',
            'سيء': 'وحش',
        }
        
        self.msa_to_levantine = {
            'كيف': 'كيفك',
            'حالك': 'حالك',
            'أريد': 'بدي',
            'أنا': 'أنا',
            'نعم': 'أيوة',
            'لا': 'لأ',
            'شكراً': 'شكراً',
            'الآن': 'هلق',
            'جداً': 'كتير',
            'جيد': 'منيح',
            'سيء': 'مش منيح',
        }
        
        self.msa_to_gulf = {
            'كيف': 'شلونك',
            'حالك': 'حالك',
            'أريد': 'أبي',
            'أنا': 'أنا',
            'نعم': 'نعم',
            'لا': 'لا',
            'شكراً': 'يعطيك العافية',
            'الآن': 'الحين',
            'جداً': 'يا',
            'جيد': 'زين',
            'سيء': 'مو زين',
        }
    
    def translate_to_egyptian(self, msa_text: str) -> str:
        """Translate MSA to Egyptian"""
        words = msa_text.split()
        translated = []
        
        for word in words:
            translated_word = self.msa_to_egyptian.get(word, word)
            translated.append(translated_word)
        
        return ' '.join(translated)
    
    def translate_to_levantine(self, msa_text: str) -> str:
        """Translate MSA to Levantine"""
        words = msa_text.split()
        translated = []
        
        for word in words:
            translated_word = self.msa_to_levantine.get(word, word)
            translated.append(translated_word)
        
        return ' '.join(translated)
    
    def translate_to_gulf(self, msa_text: str) -> str:
        """Translate MSA to Gulf"""
        words = msa_text.split()
        translated = []
        
        for word in words:
            translated_word = self.msa_to_gulf.get(word, word)
            translated.append(translated_word)
        
        return ' '.join(translated)
    
    def translate(self, msa_text: str, target_dialect: str) -> str:
        """
        Translate MSA to target dialect
        
        Args:
            msa_text: MSA text
            target_dialect: 'egyptian', 'levantine', 'gulf', 'maghrebi'
        """
        if target_dialect == 'egyptian':
            return self.translate_to_egyptian(msa_text)
        elif target_dialect == 'levantine':
            return self.translate_to_levantine(msa_text)
        elif target_dialect == 'gulf':
            return self.translate_to_gulf(msa_text)
        else:
            return msa_text  # Return MSA if dialect not found


# ============================================================================
# Part 5: Arabic Text Summarization
# ============================================================================

class ArabicTextSummarizer:
    """
    Arabic Text Summarization
    
    Extractive summarization based on:
    - Sentence frequency
    - Position importance
    - Keyword density
    """
    
    def __init__(self):
        # Arabic stop words
        self.stop_words = {
            'في', 'من', 'إلى', 'على', 'عن', 'مع', 'لـ', 'بـ', 'كـ',
            'ال', 'هذا', 'هذه', 'ذلك', 'تلك', 'أن', 'إن', 'لأن',
            'و', 'أو', 'لكن', 'فإن', 'قد', 'لن', 'لم', 'لا',
            'هو', 'هي', 'هم', 'هن', 'أنا', 'نحن', 'أنت', 'أنتم'
        }
    
    def summarize(self, text: str, num_sentences: int = 3) -> str:
        """
        Summarize Arabic text
        
        Args:
            text: Input text
            num_sentences: Number of sentences in summary
        """
        # Split into sentences
        sentences = re.split(r'[.!?]\s*', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) <= num_sentences:
            return text
        
        # Calculate word frequencies
        word_freq = self._calculate_word_frequencies(text)
        
        # Score sentences
        sentence_scores = []
        for i, sentence in enumerate(sentences):
            score = self._score_sentence(sentence, word_freq, i, len(sentences))
            sentence_scores.append((i, sentence, score))
        
        # Get top sentences
        sentence_scores.sort(key=lambda x: x[2], reverse=True)
        top_sentences = sentence_scores[:num_sentences]
        
        # Sort by original position
        top_sentences.sort(key=lambda x: x[0])
        
        # Build summary
        summary = ' '.join([s[1] for s in top_sentences])
        
        return summary
    
    def _calculate_word_frequencies(self, text: str) -> Dict[str, int]:
        """Calculate word frequencies"""
        words = text.lower().split()
        word_freq = {}
        
        for word in words:
            # Remove punctuation
            word = re.sub(r'[^\w\s]', '', word)
            
            # Skip stop words
            if word in self.stop_words:
                continue
            
            word_freq[word] = word_freq.get(word, 0) + 1
        
        return word_freq
    
    def _score_sentence(self, sentence: str, word_freq: Dict[str, int],
                       position: int, total_sentences: int) -> float:
        """Score a sentence"""
        words = sentence.lower().split()
        
        # Word frequency score
        freq_score = sum(word_freq.get(word, 0) for word in words)
        
        # Position score (first and last sentences are important)
        position_score = 0
        if position == 0:
            position_score = 3.0
        elif position == total_sentences - 1:
            position_score = 2.0
        elif position < total_sentences * 0.2:
            position_score = 1.5
        
        # Length score (prefer medium-length sentences)
        length = len(words)
        length_score = 1.0
        if 10 <= length <= 30:
            length_score = 1.5
        elif length < 5 or length > 50:
            length_score = 0.5
        
        # Combined score
        total_score = freq_score + position_score + length_score
        
        return total_score


# ============================================================================
# Part 6: Arabic Question Answering
# ============================================================================

class ArabicQuestionAnswerer:
    """
    Arabic Question Answering
    
    Simple extractive QA system
    """
    
    def __init__(self):
        self.question_words = {
            'ما': 'what',
            'ماذا': 'what',
            'من': 'who',
            'أين': 'where',
            'متى': 'when',
            'كيف': 'how',
            'لماذا': 'why',
            'كم': 'how much',
            'هل': 'yes/no'
        }
    
    def answer(self, question: str, context: str) -> Dict:
        """
        Answer question based on context
        
        Args:
            question: Question text
            context: Context text
        
        Returns:
            Dict with answer and confidence
        """
        # Detect question type
        q_type = self._detect_question_type(question)
        
        # Find relevant sentences
        relevant_sentences = self._find_relevant_sentences(question, context)
        
        if not relevant_sentences:
            return {
                'answer': 'لا أعرف',
                'confidence': 0.0,
                'question_type': q_type
            }
        
        # Extract answer
        answer = self._extract_answer(question, relevant_sentences[0], q_type)
        
        return {
            'answer': answer,
            'confidence': 0.8,
            'question_type': q_type,
            'source': relevant_sentences[0][:100]
        }
    
    def _detect_question_type(self, question: str) -> str:
        """Detect question type"""
        for q_word, q_type in self.question_words.items():
            if question.startswith(q_word):
                return q_type
        return 'unknown'
    
    def _find_relevant_sentences(self, question: str, context: str) -> List[str]:
        """Find sentences relevant to question"""
        sentences = re.split(r'[.!?]\s*', context)
        
        question_words = set(question.lower().split())
        
        scored_sentences = []
        for sentence in sentences:
            # Calculate overlap
            sentence_words = set(sentence.lower().split())
            overlap = len(question_words & sentence_words)
            scored_sentences.append((overlap, sentence))
        
        # Sort by score
        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        
        # Return top sentences
        return [s[1] for s in scored_sentences if s[0] > 0]
    
    def _extract_answer(self, question: str, sentence: str, q_type: str) -> str:
        """Extract answer from sentence"""
        # Simple extraction (in production, use ML model)
        return sentence


# ============================================================================
# Part 7: Example Usage
# ============================================================================

def demo_all_features():
    """Demo all Arabic NLP features"""
    
    print("=" * 60)
    print("Advanced Arabic NLP Demo")
    print("=" * 60)
    
    # Sample texts
    texts = {
        'ner': "السيد أحمد محمد يعمل في شركة التقنية المتقدمة في الرياض منذ عام 2020.",
        'pos': "الطالب المجتهد يذهب إلى الجامعة كل يوم ليدرس بجد.",
        'sentiment': "المنتج رائع جداً وجيد، أنا سعيد به وأنصح به بشدة.",
        'dialect': "أنا أريد الذهاب إلى المدرسة الآن.",
        'summary': """الذكاء الاصطناعي هو فرع من علوم الحاسوب.
        يهدف إلى إنشاء أنظمة ذكية.
        التعلم الآلي هو جزء من الذكاء الاصطناعي.
        التعلم العميق يستخدم الشبكات العصبية.
        التطبيقات تشمل السيارات ذاتية القيادة والمساعدين الافتراضيين.""",
        'qa': {
            'question': "ما هو الذكاء الاصطناعي؟",
            'context': "الذكاء الاصطناعي هو مجال من علوم الحاسوب. يهدف إلى إنشاء آلات ذكية. التعلم الآلي جزء مهم منه."
        }
    }
    
    # 1. NER Demo
    print("\n1. Named Entity Recognition")
    print("-" * 60)
    ner = ArabicNER()
    entities = ner.extract_entities(texts['ner'])
    print(f"Text: {texts['ner']}")
    print(f"Entities found: {len(entities)}")
    for entity in entities:
        print(f"  - {entity.text} ({entity.label})")
    
    # 2. POS Tagging Demo
    print("\n2. Part-of-Speech Tagging")
    print("-" * 60)
    pos_tagger = ArabicPOSTagger()
    tagged = pos_tagger.tag_sentence(texts['pos'])
    print(f"Text: {texts['pos']}")
    print(f"Tagged: {tagged}")
    
    # 3. Sentiment Analysis Demo
    print("\n3. Sentiment Analysis")
    print("-" * 60)
    sentiment = ArabicSentimentAnalyzer()
    result = sentiment.analyze(texts['sentiment'])
    print(f"Text: {texts['sentiment']}")
    print(f"Sentiment: {result['sentiment']} ({result['label_ar']})")
    print(f"Confidence: {result['confidence']:.2f}")
    
    # 4. Dialect Translation Demo
    print("\n4. Dialect Translation")
    print("-" * 60)
    translator = ArabicDialectTranslator()
    print(f"MSA: {texts['dialect']}")
    print(f"Egyptian: {translator.translate_to_egyptian(texts['dialect'])}")
    print(f"Levantine: {translator.translate_to_levantine(texts['dialect'])}")
    print(f"Gulf: {translator.translate_to_gulf(texts['dialect'])}")
    
    # 5. Summarization Demo
    print("\n5. Text Summarization")
    print("-" * 60)
    summarizer = ArabicTextSummarizer()
    summary = summarizer.summarize(texts['summary'], num_sentences=2)
    print(f"Original: {texts['summary']}")
    print(f"Summary: {summary}")
    
    # 6. Question Answering Demo
    print("\n6. Question Answering")
    print("-" * 60)
    qa = ArabicQuestionAnswerer()
    answer = qa.answer(
        texts['qa']['question'],
        texts['qa']['context']
    )
    print(f"Question: {texts['qa']['question']}")
    print(f"Answer: {answer['answer']}")
    print(f"Confidence: {answer['confidence']}")
    
    print("\n" + "=" * 60)
    print("✓ All demos completed!")
    print("=" * 60)


if __name__ == "__main__":
    demo_all_features()
