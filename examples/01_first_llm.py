"""
01_first_llm.py - Your First LLM Experience

This example introduces you to working with Large Language Models using Hugging Face Transformers.
We'll start with a simple text generation task to understand the basics.

Learning Objectives:
- Load a pre-trained LLM
- Generate text using the model
- Understand basic parameters like temperature and max_length
- See how different prompts affect output
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def main():
    print("🚀 Welcome to your first LLM experience!")
    print("=" * 50)
    
    # Step 1: Choose and load a model
    print("\n📦 Loading model and tokenizer...")
    print("We're using GPT-2, a smaller model that's great for learning")
    
    model_name = "gpt2"  # Small, fast model good for learning
    
    try:
        # Load the tokenizer (converts text to numbers the model understands)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load the actual model
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Add padding token if it doesn't exist
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        print("✅ Model loaded successfully!")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("💡 Try running: pip install transformers torch")
        return
    
    # Step 2: Basic text generation
    print("\n🎯 Basic Text Generation")
    print("-" * 30)
    
    # Simple prompt
    prompt = "The future of artificial intelligence is"
    print(f"Prompt: '{prompt}'")
    
    # Tokenize the input
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Generate text
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=50,           # Maximum length of generated text
            temperature=0.7,         # Controls randomness (0.0 = deterministic, 1.0 = creative)
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,          # Enable sampling for more diverse outputs
            num_return_sequences=1   # How many different completions to generate
        )
    
    # Decode the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated: '{generated_text}'")
    
    # Step 3: Understanding temperature
    print("\n🌡️ Understanding Temperature")
    print("-" * 35)
    
    temperatures = [0.1, 0.7, 1.2]
    prompt = "Once upon a time, in a magical forest,"
    
    for temp in temperatures:
        print(f"\nTemperature: {temp}")
        inputs = tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_length=40,
                temperature=temp,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                num_return_sequences=1
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        completion = generated_text[len(prompt):].strip()
        print(f"Output: {completion}")
    
    # Step 4: Interactive example
    print("\n💬 Try Your Own Prompts!")
    print("-" * 30)
    print("Enter a prompt to see how the model completes it (or 'quit' to exit)")
    
    while True:
        user_prompt = input("\nYour prompt: ").strip()
        
        if user_prompt.lower() in ['quit', 'exit', 'q']:
            break
            
        if not user_prompt:
            continue
            
        try:
            inputs = tokenizer(user_prompt, return_tensors="pt")
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_length=len(inputs.input_ids[0]) + 30,  # Add 30 tokens to the prompt
                    temperature=0.8,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=True,
                    num_return_sequences=1
                )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            completion = generated_text[len(user_prompt):].strip()
            print(f"Model completion: {completion}")
            
        except Exception as e:
            print(f"Error generating text: {e}")
    
    print("\n🎉 Great job! You've completed your first LLM exercise!")
    print("\n📚 What you learned:")
    print("- How to load and use a pre-trained model")
    print("- The effect of temperature on text generation")
    print("- How prompts influence model outputs")
    print("- Basic text generation pipeline")
    print("\n➡️  Next: Try examples/02_prompt_engineering.py to learn about better prompting!")

if __name__ == "__main__":
    main()