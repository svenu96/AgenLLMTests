"""
tokenization_demo.py - Understanding Tokenization (Offline Demo)

This example teaches tokenization concepts without requiring internet access.
We'll build a simple tokenizer to understand how text becomes tokens.

Learning Objectives:
- Understand what tokenization is and why it's needed
- See different tokenization approaches
- Build a simple tokenizer from scratch
- Learn about common tokenization challenges
"""

import re
import json
from collections import Counter
from typing import List, Dict

class SimpleTokenizer:
    """A basic tokenizer to demonstrate tokenization concepts"""
    
    def __init__(self):
        self.vocab = {}
        self.word_freq = Counter()
        self.special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1, 
            '<START>': 2,
            '<END>': 3
        }
        
    def basic_word_tokenize(self, text: str) -> List[str]:
        """Simple word-level tokenization"""
        # Convert to lowercase and split by spaces/punctuation
        text = text.lower()
        # Split on whitespace and punctuation
        tokens = re.findall(r'\b\w+\b|[^\w\s]', text)
        return tokens
    
    def character_tokenize(self, text: str) -> List[str]:
        """Character-level tokenization"""
        return list(text)
    
    def subword_tokenize(self, text: str) -> List[str]:
        """Simplified subword tokenization (demonstration only)"""
        words = self.basic_word_tokenize(text)
        subwords = []
        
        for word in words:
            if len(word) <= 3:
                subwords.append(word)
            else:
                # Split longer words into chunks
                for i in range(0, len(word), 3):
                    chunk = word[i:i+3]
                    if i > 0:
                        chunk = "##" + chunk  # Mark as continuation
                    subwords.append(chunk)
        
        return subwords
    
    def build_vocabulary(self, texts: List[str]) -> Dict[str, int]:
        """Build vocabulary from training texts"""
        all_tokens = []
        
        for text in texts:
            tokens = self.basic_word_tokenize(text)
            all_tokens.extend(tokens)
            
        # Count token frequencies
        token_counts = Counter(all_tokens)
        
        # Build vocabulary (start with special tokens)
        vocab = self.special_tokens.copy()
        
        # Add most frequent tokens
        for token, count in token_counts.most_common():
            if token not in vocab:
                vocab[token] = len(vocab)
                
        return vocab
    
    def encode(self, text: str, vocab: Dict[str, int]) -> List[int]:
        """Convert text to token IDs"""
        tokens = self.basic_word_tokenize(text)
        token_ids = []
        
        for token in tokens:
            if token in vocab:
                token_ids.append(vocab[token])
            else:
                token_ids.append(vocab['<UNK>'])  # Unknown token
                
        return token_ids
    
    def decode(self, token_ids: List[int], vocab: Dict[str, int]) -> str:
        """Convert token IDs back to text"""
        # Create reverse vocabulary
        id_to_token = {id: token for token, id in vocab.items()}
        
        tokens = []
        for token_id in token_ids:
            if token_id in id_to_token:
                token = id_to_token[token_id]
                if token not in self.special_tokens:
                    tokens.append(token)
                    
        return ' '.join(tokens)

def demo_tokenization_approaches():
    """Demonstrate different tokenization approaches"""
    print("🔤 Tokenization Approaches Demo")
    print("=" * 40)
    
    # Sample text
    text = "Hello world! This is a tokenization example. Can you understand how it works?"
    print(f"Original text: '{text}'")
    print()
    
    tokenizer = SimpleTokenizer()
    
    # 1. Word-level tokenization
    word_tokens = tokenizer.basic_word_tokenize(text)
    print(f"📝 Word tokens: {word_tokens}")
    print(f"   Count: {len(word_tokens)} tokens")
    print()
    
    # 2. Character-level tokenization
    char_tokens = tokenizer.character_tokenize(text)
    print(f"🔤 Character tokens: {char_tokens[:20]}...")  # Show first 20
    print(f"   Count: {len(char_tokens)} tokens")
    print()
    
    # 3. Subword tokenization
    subword_tokens = tokenizer.subword_tokenize(text)
    print(f"🧩 Subword tokens: {subword_tokens}")
    print(f"   Count: {len(subword_tokens)} tokens")
    print()

def demo_vocabulary_building():
    """Show how to build a vocabulary"""
    print("📚 Vocabulary Building Demo")
    print("=" * 35)
    
    # Sample training texts
    training_texts = [
        "The cat sits on the mat",
        "A dog runs in the park", 
        "The bird flies in the sky",
        "Fish swim in the water",
        "The sun shines bright"
    ]
    
    print("Training texts:")
    for i, text in enumerate(training_texts, 1):
        print(f"  {i}. {text}")
    print()
    
    tokenizer = SimpleTokenizer()
    vocab = tokenizer.build_vocabulary(training_texts)
    
    print("📖 Built vocabulary:")
    for token, id in sorted(vocab.items(), key=lambda x: x[1]):
        print(f"  {id:2d}: '{token}'")
    print(f"\nVocabulary size: {len(vocab)} tokens")

def demo_encoding_decoding():
    """Demonstrate encoding and decoding"""
    print("\n🔄 Encoding/Decoding Demo")
    print("=" * 30)
    
    # Build vocabulary from sample texts
    training_texts = [
        "hello world", "world peace", "peace and love", 
        "love is beautiful", "beautiful day"
    ]
    
    tokenizer = SimpleTokenizer()
    vocab = tokenizer.build_vocabulary(training_texts)
    
    # Test text
    test_text = "hello beautiful world"
    print(f"Original: '{test_text}'")
    
    # Encode
    token_ids = tokenizer.encode(test_text, vocab)
    print(f"Encoded:  {token_ids}")
    
    # Decode
    decoded = tokenizer.decode(token_ids, vocab)
    print(f"Decoded:  '{decoded}'")
    
    # Test with unknown word
    test_unknown = "hello amazing world"
    print(f"\nWith unknown word: '{test_unknown}'")
    token_ids_unk = tokenizer.encode(test_unknown, vocab)
    print(f"Encoded: {token_ids_unk}")
    decoded_unk = tokenizer.decode(token_ids_unk, vocab)
    print(f"Decoded: '{decoded_unk}' (notice missing 'amazing')")

def interactive_tokenization():
    """Let users experiment with tokenization"""
    print("\n🎯 Interactive Tokenization")
    print("-" * 30)
    
    tokenizer = SimpleTokenizer()
    
    # Build a small vocabulary
    sample_texts = [
        "the quick brown fox jumps over the lazy dog",
        "artificial intelligence is amazing technology",
        "machine learning helps solve complex problems"
    ]
    
    vocab = tokenizer.build_vocabulary(sample_texts)
    print(f"Built vocabulary with {len(vocab)} tokens")
    print("Try typing some text to see how it gets tokenized!")
    
    while True:
        user_input = input("\nYour text (or 'quit'): ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
            
        if not user_input:
            continue
        
        print(f"\nText: '{user_input}'")
        
        # Show different tokenizations
        word_tokens = tokenizer.basic_word_tokenize(user_input)
        print(f"Word tokens: {word_tokens}")
        
        subword_tokens = tokenizer.subword_tokenize(user_input)
        print(f"Subword tokens: {subword_tokens}")
        
        # Encode/decode
        token_ids = tokenizer.encode(user_input, vocab)
        print(f"Token IDs: {token_ids}")
        
        decoded = tokenizer.decode(token_ids, vocab)
        print(f"Decoded: '{decoded}'")
        
        # Count known vs unknown
        known_tokens = sum(1 for token in word_tokens if token in vocab)
        unknown_tokens = len(word_tokens) - known_tokens
        print(f"Known: {known_tokens}, Unknown: {unknown_tokens}")

def tokenization_challenges():
    """Show common tokenization challenges"""
    print("\n⚠️  Tokenization Challenges")
    print("=" * 35)
    
    tokenizer = SimpleTokenizer()
    
    challenges = [
        ("Punctuation", "Hello, world! How are you?"),
        ("Numbers", "I have 123 apples and $45.67"),
        ("Contractions", "Don't you think it's working?"),
        ("URLs", "Visit https://example.com for more info"),
        ("Emojis", "Great work! 😊 👍 🚀"),
        ("Mixed case", "iPhone and JavaScript are CamelCase")
    ]
    
    for challenge_type, text in challenges:
        print(f"\n📋 {challenge_type}:")
        print(f"   Text: '{text}'")
        tokens = tokenizer.basic_word_tokenize(text)
        print(f"   Tokens: {tokens}")

def main():
    """Run all tokenization demonstrations"""
    print("🎓 Tokenization Masterclass")
    print("=" * 35)
    print("Learn how text becomes numbers that AI can understand!")
    print()
    
    # Basic demonstrations
    demo_tokenization_approaches()
    demo_vocabulary_building()
    demo_encoding_decoding()
    tokenization_challenges()
    
    print("\n🎯 Key Takeaways:")
    print("- Tokenization converts text to numbers")
    print("- Different approaches have different trade-offs")
    print("- Vocabulary size affects model complexity")
    print("- Unknown words are handled with <UNK> tokens")
    print("- Real tokenizers are more sophisticated than our demo")
    
    # Interactive session
    try_interactive = input("\nTry interactive tokenization? (y/n): ").strip().lower()
    if try_interactive in ['y', 'yes']:
        interactive_tokenization()
    
    print("\n🎉 You now understand tokenization basics!")
    print("\n📚 What's next?")
    print("- Try examples/01_first_llm.py (requires internet)")
    print("- Read docs/01_tokenization.md for more details")
    print("- Explore real tokenizers when you have internet access")

if __name__ == "__main__":
    main()