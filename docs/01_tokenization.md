# Understanding Tokenization in LLMs

## What is Tokenization?

Tokenization is the process of breaking down text into smaller units called "tokens" that language models can understand and process. Think of it as converting human language into a format that computers can work with.

## Why Do We Need Tokenization?

- **Computers don't understand words directly** - they work with numbers
- **Standardization** - ensures consistent input format across different texts
- **Efficiency** - allows models to process text in manageable chunks
- **Vocabulary management** - handles unknown words and various languages

## Types of Tokenization

### 1. Word-Level Tokenization
```
Text: "Hello world!"
Tokens: ["Hello", "world", "!"]
```
**Pros**: Intuitive, preserves word boundaries
**Cons**: Large vocabulary, struggles with rare words

### 2. Character-Level Tokenization
```
Text: "Hello"
Tokens: ["H", "e", "l", "l", "o"]
```
**Pros**: Small vocabulary, handles any text
**Cons**: Long sequences, loses word meaning

### 3. Subword Tokenization (Most Common)
```
Text: "unhappiness"
Tokens: ["un", "happy", "ness"]
```
**Pros**: Balanced vocabulary size, handles rare words
**Cons**: Requires training, can split words oddly

## Popular Tokenization Algorithms

### BPE (Byte Pair Encoding)
- Used by GPT models
- Merges most frequent character pairs iteratively
- Good balance between vocabulary size and sequence length

### WordPiece
- Used by BERT models
- Similar to BPE but optimized for likelihood
- Adds "##" prefix for subword tokens

### SentencePiece
- Used by T5, XLNet
- Works directly on raw text (no pre-tokenization)
- Language-agnostic approach

## Practical Example

Let's see how different models tokenize the same text:

```python
from transformers import AutoTokenizer

text = "The quick brown fox jumps over the lazy dog"

# GPT-2 (BPE)
gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
gpt2_tokens = gpt2_tokenizer.tokenize(text)
print(f"GPT-2: {gpt2_tokens}")
# Output: ['The', 'Ġquick', 'Ġbrown', 'Ġfox', 'Ġjumps', 'Ġover', 'Ġthe', 'Ġlazy', 'Ġdog']

# BERT (WordPiece)
bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert_tokens = bert_tokenizer.tokenize(text)
print(f"BERT: {bert_tokens}")
# Output: ['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog']
```

## Key Concepts

### Special Tokens
- **[CLS]**: Classification token (BERT)
- **[SEP]**: Separator token 
- **[PAD]**: Padding token
- **[UNK]**: Unknown token
- **[MASK]**: Masked token (for training)

### Token IDs
Each token gets converted to a unique number:
```python
tokens = ["Hello", "world"]
token_ids = [15496, 995]  # Example IDs
```

### Vocabulary Size
- Determines model complexity
- Common sizes: 32K (GPT-2), 50K (GPT-3), 100K+ (newer models)
- Larger vocabulary = more parameters

## Best Practices

1. **Understand your model's tokenizer** - different models use different approaches
2. **Check token limits** - most models have maximum context lengths
3. **Handle special characters** - emojis, code, math symbols may tokenize unexpectedly
4. **Consider multilingual texts** - some tokenizers work better for specific languages

## Common Issues and Solutions

### Issue: Text gets split strangely
```python
# Problem
tokens = tokenizer.tokenize("GPT-4")
# Might become: ["G", "PT", "-", "4"]

# Solution: Use consistent naming/formatting
```

### Issue: Running out of context
```python
# Check token count before processing
tokens = tokenizer.encode(text)
if len(tokens) > model.config.max_position_embeddings:
    # Truncate or split the text
    text = text[:max_chars]
```

### Issue: Inconsistent results across models
```python
# Always specify which tokenizer you're using
tokenizer = AutoTokenizer.from_pretrained("specific-model-name")
```

## Try It Yourself

Create a simple script to explore tokenization:

```python
from transformers import AutoTokenizer

def explore_tokenization(text, model_name="gpt2"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Tokenize
    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.encode(text)
    
    print(f"Original text: {text}")
    print(f"Tokens: {tokens}")
    print(f"Token IDs: {token_ids}")
    print(f"Number of tokens: {len(tokens)}")
    
    # Decode back
    decoded = tokenizer.decode(token_ids)
    print(f"Decoded: {decoded}")

# Test with different texts
explore_tokenization("Hello, world!")
explore_tokenization("artificial intelligence")
explore_tokenization("🤖 AI is amazing! 🚀")
```

## Next Steps

- Try the tokenization example in `examples/tokenization_demo.py`
- Experiment with different models and see how they tokenize text
- Learn about the impact of tokenization on model performance
- Explore multilingual tokenization challenges

## Further Reading

- [Hugging Face Tokenizers Documentation](https://huggingface.co/docs/tokenizers/)
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [BPE Paper](https://arxiv.org/abs/1508.07909)
- [SentencePiece Paper](https://arxiv.org/abs/1808.06226)