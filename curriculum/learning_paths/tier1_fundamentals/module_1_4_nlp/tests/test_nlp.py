"""
Tests for Module 1.4: NLP.
"""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from part1_fundamentals.module_1_4_nlp.tokenization import (
    WordTokenizer, CharTokenizer, BPETokenizer, WordPieceTokenizer,
    get_tokenizer,
)
from part1_fundamentals.module_1_4_nlp.embeddings import (
    Word2Vec, GloVe, PositionalEncoding, TokenEmbeddings,
)
from part1_fundamentals.module_1_4_nlp.sequence_models import (
    RNNCell, LSTMCell, GRUCell, RNN, LSTM, GRU,
)
from part1_fundamentals.module_1_4_nlp.text_preprocessing import (
    Stemmer, Lemmatizer, StopWordsRemover, NGramGenerator,
    TFIDFVectorizer, TextNormalizer,
)


class TestTokenization(unittest.TestCase):
    """Tests for tokenization."""
    
    def test_word_tokenizer(self):
        """Test word tokenization."""
        tokenizer = WordTokenizer()
        text = "Hello, World! How are you?"
        tokens = tokenizer.tokenize(text)
        
        self.assertGreater(len(tokens), 0)
        self.assertTrue(all(t.islower() for t in tokens))
    
    def test_char_tokenizer(self):
        """Test character tokenization."""
        tokenizer = CharTokenizer()
        text = "Hello"
        tokens = tokenizer.tokenize(text)
        
        self.assertEqual(len(tokens), 5)
        self.assertEqual(tokens, ['h', 'e', 'l', 'l', 'o'])
    
    def test_bpe_tokenizer(self):
        """Test BPE tokenization."""
        tokenizer = BPETokenizer(vocab_size=100, min_frequency=1)
        texts = ["hello world", "hello there", "world peace"] * 10
        tokenizer.train(texts, show_progress=False)
        
        tokens = tokenizer.tokenize("hello world")
        self.assertGreater(len(tokens), 0)
    
    def test_bpe_encode_decode(self):
        """Test BPE encode/decode."""
        tokenizer = BPETokenizer(vocab_size=100, min_frequency=1)
        texts = ["hello world", "hello there"] * 10
        tokenizer.train(texts, show_progress=False)
        
        text = "hello world"
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        
        # Decoded should contain original words
        self.assertIn("hello", decoded.lower())
    
    def test_get_tokenizer(self):
        """Test tokenizer factory."""
        word_tokenizer = get_tokenizer('word')
        self.assertIsInstance(word_tokenizer, WordTokenizer)
        
        char_tokenizer = get_tokenizer('char')
        self.assertIsInstance(char_tokenizer, CharTokenizer)


class TestEmbeddings(unittest.TestCase):
    """Tests for embeddings."""
    
    def test_word2vec_training(self):
        """Test Word2Vec training."""
        texts = [
            "the cat sat on the mat",
            "the dog sat on the log",
            "cats and dogs are pets",
        ] * 50
        
        w2v = Word2Vec(embedding_dim=50, epochs=10, min_count=1)
        w2v.train(texts, show_progress=False)
        
        self.assertGreater(len(w2v.word_to_idx), 0)
    
    def test_word2vec_get_vector(self):
        """Test Word2Vec word vector retrieval."""
        texts = ["hello world", "hello there", "world peace"] * 50
        
        w2v = Word2Vec(embedding_dim=50, epochs=10, min_count=1)
        w2v.train(texts, show_progress=False)
        
        vector = w2v.get_vector("hello")
        self.assertIsNotNone(vector)
        self.assertEqual(len(vector), 50)
    
    def test_word2vec_most_similar(self):
        """Test Word2Vec similarity."""
        texts = [
            "cat kitten feline",
            "dog puppy canine",
            "cat meow pet",
            "dog bark pet",
        ] * 50
        
        w2v = Word2Vec(embedding_dim=50, epochs=20, min_count=1)
        w2v.train(texts, show_progress=False)
        
        similar = w2v.most_similar("cat", top_k=3)
        self.assertGreater(len(similar), 0)
    
    def test_glove_training(self):
        """Test GloVe training."""
        texts = [
            "the cat sat on the mat",
            "the dog sat on the log",
        ] * 50
        
        glove = GloVe(embedding_dim=50, epochs=20, min_count=1)
        glove.train(texts, show_progress=False)
        
        vector = glove.get_vector("cat")
        self.assertIsNotNone(vector)
    
    def test_positional_encoding(self):
        """Test positional encoding."""
        pos_enc = PositionalEncoding(d_model=128, max_len=100)
        
        x = np.zeros((2, 10, 128))
        output = pos_enc.forward(x)
        
        self.assertEqual(output.shape, (2, 10, 128))
        
        # Different positions should have different encodings
        self.assertFalse(np.allclose(output[0, 0, :], output[0, 1, :]))
    
    def test_token_embeddings(self):
        """Test token embeddings."""
        embeddings = TokenEmbeddings(vocab_size=1000, d_model=256)
        
        token_ids = np.random.randint(0, 1000, (4, 20))
        output = embeddings.forward(token_ids)
        
        self.assertEqual(output.shape, (4, 20, 256))


class TestSequenceModels(unittest.TestCase):
    """Tests for sequence models."""
    
    def test_rnn_cell(self):
        """Test RNN cell."""
        cell = RNNCell(input_size=32, hidden_size=64)
        x = np.random.randn(4, 32)
        h_prev = np.zeros((4, 64))
        
        h_next = cell.forward(x, h_prev)
        
        self.assertEqual(h_next.shape, (4, 64))
    
    def test_lstm_cell(self):
        """Test LSTM cell."""
        cell = LSTMCell(input_size=32, hidden_size=64)
        x = np.random.randn(4, 32)
        h_prev = np.zeros((4, 64))
        c_prev = np.zeros((4, 64))
        
        h_next, c_next = cell.forward(x, h_prev, c_prev)
        
        self.assertEqual(h_next.shape, (4, 64))
        self.assertEqual(c_next.shape, (4, 64))
    
    def test_gru_cell(self):
        """Test GRU cell."""
        cell = GRUCell(input_size=32, hidden_size=64)
        x = np.random.randn(4, 32)
        h_prev = np.zeros((4, 64))
        
        h_next = cell.forward(x, h_prev)
        
        self.assertEqual(h_next.shape, (4, 64))
    
    def test_lstm_forward(self):
        """Test LSTM forward pass."""
        lstm = LSTM(input_size=32, hidden_size=64, num_layers=2)
        x = np.random.randn(4, 10, 32)
        
        output, (h_n, c_n) = lstm.forward(x)
        
        self.assertEqual(output.shape, (4, 10, 64))
        self.assertEqual(h_n.shape[0], 2)  # num_layers
    
    def test_bidirectional_lstm(self):
        """Test bidirectional LSTM."""
        lstm = LSTM(input_size=32, hidden_size=64, bidirectional=True)
        x = np.random.randn(4, 10, 32)
        
        output, (h_n, c_n) = lstm.forward(x)
        
        # Output dimension should be 2 * hidden_size
        self.assertEqual(output.shape, (4, 10, 128))
    
    def test_gru_forward(self):
        """Test GRU forward pass."""
        gru = GRU(input_size=32, hidden_size=64, num_layers=2)
        x = np.random.randn(4, 10, 32)
        
        output, h_n = gru.forward(x)
        
        self.assertEqual(output.shape, (4, 10, 64))


class TestTextPreprocessing(unittest.TestCase):
    """Tests for text preprocessing."""
    
    def test_stemmer_porter(self):
        """Test Porter stemmer."""
        stemmer = Stemmer(algorithm='porter')
        
        self.assertEqual(stemmer.stem("running"), "run")
        self.assertEqual(stemmer.stem("runs"), "run")
    
    def test_lemmatizer(self):
        """Test lemmatizer."""
        lemmatizer = Lemmatizer()
        
        self.assertEqual(lemmatizer.lemmatize("running", pos='verb'), "run")
        self.assertEqual(lemmatizer.lemmatize("better", pos='adjective'), "good")
        self.assertEqual(lemmatizer.lemmatize("children", pos='noun'), "child")
    
    def test_stop_words_removal(self):
        """Test stop words removal."""
        remover = StopWordsRemover(language='english')
        tokens = ["the", "quick", "brown", "fox"]
        
        filtered = remover.remove(tokens)
        
        self.assertNotIn("the", filtered)
        self.assertIn("quick", filtered)
    
    def test_ngram_generator(self):
        """Test n-gram generation."""
        generator = NGramGenerator(n=2)
        tokens = ["the", "quick", "brown", "fox"]
        
        bigrams = generator.generate(tokens)
        
        self.assertEqual(len(bigrams), 3)
        self.assertEqual(bigrams[0], ("the", "quick"))
    
    def test_tfidf_vectorizer(self):
        """Test TF-IDF vectorization."""
        documents = [
            "hello world",
            "hello there",
            "world peace",
        ]
        
        vectorizer = TFIDFVectorizer()
        tfidf_matrix = vectorizer.fit_transform(documents)
        
        self.assertEqual(tfidf_matrix.shape[0], 3)
        self.assertGreater(tfidf_matrix.shape[1], 0)
    
    def test_tfidf_feature_names(self):
        """Test TF-IDF feature names."""
        documents = ["hello world", "hello there"]
        
        vectorizer = TFIDFVectorizer()
        vectorizer.fit_transform(documents)
        
        feature_names = vectorizer.get_feature_names()
        
        self.assertIn("hello", feature_names)
        self.assertIn("world", feature_names)
    
    def test_text_normalizer(self):
        """Test text normalization."""
        normalizer = TextNormalizer(
            lowercase=True,
            remove_punctuation=True,
            remove_numbers=True
        )
        
        text = "Hello, WORLD! 123"
        normalized = normalizer.normalize(text)
        
        self.assertEqual(normalized, "hello world")


class TestIntegration(unittest.TestCase):
    """Integration tests for NLP pipeline."""
    
    def test_full_pipeline(self):
        """Test full NLP preprocessing pipeline."""
        # Tokenization
        tokenizer = WordTokenizer()
        text = "The quick brown fox jumps over the lazy dog"
        tokens = tokenizer.tokenize(text)
        
        # Stop words removal
        stop_remover = StopWordsRemover()
        filtered = stop_remover.remove(tokens)
        
        # Stemming
        stemmer = Stemmer()
        stemmed = [stemmer.stem(t) for t in filtered]
        
        # TF-IDF
        vectorizer = TFIDFVectorizer()
        tfidf = vectorizer.fit_transform([" ".join(stemmed)])
        
        self.assertEqual(tfidf.shape[0], 1)
        self.assertGreater(tfidf.shape[1], 0)
    
    def test_word2vec_with_tokenization(self):
        """Test Word2Vec with tokenized input."""
        texts = [
            "the cat sat on the mat",
            "the dog sat on the log",
        ] * 50
        
        tokenizer = WordTokenizer()
        tokenized = [" ".join(tokenizer.tokenize(t)) for t in texts]
        
        w2v = Word2Vec(embedding_dim=50, epochs=10, min_count=1)
        w2v.train(tokenized, show_progress=False)
        
        vector = w2v.get_vector("cat")
        self.assertIsNotNone(vector)


if __name__ == '__main__':
    unittest.main()
